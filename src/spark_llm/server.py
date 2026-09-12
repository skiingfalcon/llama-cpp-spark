"""Build llama-server argv and manage process lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console

from spark_llm.config import Settings, get_settings, repo_root
from spark_llm.download import resolve_local
from spark_llm.registry import Defaults, ModelKind, ModelSpec, Registry, load_registry

console = Console(stderr=True)


@dataclass
class ServeOverrides:
    """CLI-layer overrides applied last in the merge."""

    host: str | None = None
    port: int | None = None
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    model_path: Path | None = None
    hf: str | None = None
    extra_args: list[str] = field(default_factory=list)


def kind_flags(kind: ModelKind) -> list[str]:
    if kind is ModelKind.embedding:
        return ["--embeddings"]
    if kind is ModelKind.rerank:
        return ["--reranking"]
    if kind is ModelKind.multimodal:
        return ["--jinja"]
    # chat
    return []


def binary_path(settings: Settings) -> Path:
    return settings.vendor_dir / "build" / "bin" / "llama-server"


def runtime_env(settings: Settings) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = str(settings.vendor_dir / "build" / "bin")
    compat = "/usr/local/cuda-13/compat"
    if not Path(compat).is_dir():
        compat = "/usr/local/cuda/compat"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{compat}:{existing}".rstrip(":")
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return env


def build_argv(
    spec: ModelSpec | None,
    defaults: Defaults,
    settings: Settings,
    overrides: ServeOverrides | None = None,
) -> list[str]:
    """Merge defaults <- kind flags <- model overrides <- CLI overrides into argv.

    Layers never hard-code knowledge of a specific model name.
    """
    overrides = overrides or ServeOverrides()
    server = binary_path(settings)
    argv: list[str] = [str(server)]

    host = overrides.host or (spec.host if spec else None) or defaults.host or settings.host
    port = (
        overrides.port
        if overrides.port is not None
        else (spec.port if spec and spec.port is not None else settings.base_port)
    )
    ngl = (
        overrides.n_gpu_layers
        if overrides.n_gpu_layers is not None
        else (
            spec.n_gpu_layers if spec and spec.n_gpu_layers is not None else defaults.n_gpu_layers
        )
    )
    flash = (spec.flash_attn if spec and spec.flash_attn else None) or defaults.flash_attn
    batch = (
        spec.batch_size if spec and spec.batch_size is not None else None
    ) or defaults.batch_size
    ubatch = (
        spec.ubatch_size if spec and spec.ubatch_size is not None else None
    ) or defaults.ubatch_size
    ctx = overrides.ctx_size
    if ctx is None and spec is not None:
        ctx = spec.ctx_size

    # Model source: CLI path > CLI -hf > local file > model hf_ref
    if overrides.model_path is not None:
        argv.extend(["-m", str(overrides.model_path)])
    elif overrides.hf:
        argv.extend(["-hf", overrides.hf])
    elif spec is not None:
        local = resolve_local(spec, settings.models_dir) if spec.file else None
        if local is not None:
            argv.extend(["-m", str(local)])
        else:
            ref = spec.hf_ref()
            if ref:
                argv.extend(["-hf", ref])
            elif spec.file:
                argv.extend(["-m", str(settings.models_dir / spec.file)])
            else:
                raise ValueError(f"model {spec.name!r} has no resolvable weights path")
    else:
        raise ValueError("need a ModelSpec or --model-path / --hf")

    argv.extend(["--host", host, "--port", str(port)])
    argv.extend(["--n-gpu-layers", str(ngl)])
    argv.extend(["--flash-attn", str(flash)])
    argv.extend(["--batch-size", str(batch), "--ubatch-size", str(ubatch)])
    if ctx is not None:
        argv.extend(["--ctx-size", str(ctx)])

    if spec is not None:
        argv.extend(kind_flags(spec.kind))
        if spec.mmproj:
            mm = Path(spec.mmproj)
            if not mm.is_file():
                mm = settings.models_dir / spec.mmproj
            argv.extend(["--mmproj", str(mm)])
        if spec.sampling:
            s = spec.sampling
            if s.temp is not None:
                argv.extend(["--temp", str(s.temp)])
            if s.top_p is not None:
                argv.extend(["--top-p", str(s.top_p)])
            if s.top_k is not None:
                argv.extend(["--top-k", str(s.top_k)])
            if s.min_p is not None:
                argv.extend(["--min-p", str(s.min_p)])
        argv.extend(spec.extra_args)

    # Deduplicate kind flags that may also appear in extra_args
    seen: set[str] = set()
    deduped: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        # keep valued flags paired
        if tok in {
            "-m",
            "-hf",
            "--host",
            "--port",
            "--n-gpu-layers",
            "--flash-attn",
            "--batch-size",
            "--ubatch-size",
            "--ctx-size",
            "--mmproj",
            "--temp",
            "--top-p",
            "--top-k",
            "--min-p",
        }:
            deduped.extend(argv[i : i + 2])
            i += 2
            continue
        if (
            tok.startswith("-")
            and tok in seen
            and tok
            in {
                "--embeddings",
                "--reranking",
                "--jinja",
            }
        ):
            i += 1
            continue
        if tok.startswith("-"):
            seen.add(tok)
        deduped.append(tok)
        i += 1
    argv = deduped

    argv.extend(overrides.extra_args)
    return argv


def state_file(settings: Settings) -> Path:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return settings.state_dir / "servers.json"


def load_state(settings: Settings) -> dict[str, dict]:
    path = state_file(settings)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_state(settings: Settings, state: dict[str, dict]) -> None:
    state_file(settings).write_text(json.dumps(state, indent=2) + "\n")


def wait_healthy(port: int, timeout_s: float, poll_s: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                r = client.get(url)
                if r.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001 — probe until timeout
                last_err = exc
            time.sleep(poll_s)
    raise TimeoutError(f"server on :{port} not healthy within {timeout_s}s ({last_err})")


def start_server(
    argv: list[str],
    name: str,
    port: int,
    settings: Settings | None = None,
    foreground: bool = False,
) -> subprocess.Popen[bytes]:
    settings = settings or get_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.state_dir / f"{name}.log"
    console.print(f"[cyan]starting[/cyan] {name} on :{port}")
    console.print("  " + " ".join(argv))

    env = runtime_env(settings)
    if foreground:
        proc = subprocess.Popen(argv, env=env)
        return proc

    log_fh = log_path.open("ab")
    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state = load_state(settings)
    state[name] = {
        "pid": proc.pid,
        "port": port,
        "argv": argv,
        "log": str(log_path),
    }
    save_state(settings, state)
    try:
        wait_healthy(port, settings.health_timeout_s, settings.health_poll_s)
    except TimeoutError:
        console.print(f"[red]health check failed[/red]; see {log_path}")
        raise
    console.print(f"[green]ready[/green] http://127.0.0.1:{port}  (pid {proc.pid})")
    return proc


def stop_servers(names: list[str] | None = None, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    state = load_state(settings)
    targets = names or list(state.keys())
    for name in targets:
        info = state.get(name)
        if not info:
            console.print(f"[yellow]not tracked[/yellow] {name}")
            continue
        pid = int(info["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]stopped[/green] {name} (pid {pid})")
        except ProcessLookupError:
            console.print(f"[yellow]already dead[/yellow] {name} (pid {pid})")
        state.pop(name, None)
    save_state(settings, state)


def argv_for_named_model(
    name: str,
    overrides: ServeOverrides | None = None,
    settings: Settings | None = None,
    registry: Registry | None = None,
) -> tuple[list[str], int]:
    settings = settings or get_settings()
    registry = registry or load_registry(settings=settings)
    spec = registry.get(name)
    overrides = overrides or ServeOverrides()
    argv = build_argv(spec, registry.defaults, settings, overrides)
    port = overrides.port if overrides.port is not None else (spec.port or settings.base_port)
    return argv, port


def ephemeral_spec_for_hf(hf: str, port: int, settings: Settings) -> ModelSpec:
    """Build a minimal ModelSpec for an unregistered -hf reference."""
    repo, _, quant = hf.partition(":")
    return ModelSpec(
        name=hf.replace("/", "_").replace(":", "_"),
        repo=repo or None,
        quant=quant or None,
        kind=ModelKind.chat,
        port=port,
        extra_args=["--jinja"],
    )


def project_build_script() -> Path:
    return repo_root() / "scripts" / "build.sh"
