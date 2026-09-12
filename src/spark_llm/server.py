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
from spark_llm.download import resolve_weights
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


@dataclass
class RuntimeParams:
    """The merged runtime configuration shared by llama-server and llama-bench.

    Produced by :func:`merge_runtime`; both ``build_argv`` (serve) and ``bench.bench_argv``
    consume it so that what gets benchmarked is exactly what gets served.
    """

    host: str
    port: int
    n_gpu_layers: int
    flash_attn: str
    batch_size: int
    ubatch_size: int
    ctx_size: int | None
    n_parallel: int | None
    cache_type_k: str | None
    cache_type_v: str | None
    model_path: Path | None
    hf_ref: str | None
    kind: ModelKind

    def as_dict(self) -> dict[str, object]:
        d = self.__dict__.copy()
        d["model_path"] = str(self.model_path) if self.model_path else None
        d["kind"] = self.kind.value
        return d


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


def _first(*values):  # type: ignore[no-untyped-def]
    for v in values:
        if v is not None:
            return v
    return None


def resolve_model_source(
    spec: ModelSpec | None,
    settings: Settings,
    overrides: ServeOverrides | None = None,
) -> tuple[Path | None, str | None]:
    """(local_path, hf_ref): CLI path > CLI -hf > local weights (file/snapshot) > model hf_ref."""
    overrides = overrides or ServeOverrides()
    if overrides.model_path is not None:
        return overrides.model_path, None
    if overrides.hf:
        return None, overrides.hf
    if spec is None:
        raise ValueError("need a ModelSpec or --model-path / --hf")
    local = resolve_weights(spec, settings.models_dir)
    if local is not None:
        return local, None
    ref = spec.hf_ref()
    if ref:
        return None, ref
    if spec.file:
        return settings.models_dir / spec.file, None
    raise ValueError(f"model {spec.name!r} has no resolvable weights path")


def merge_runtime(
    spec: ModelSpec | None,
    defaults: Defaults,
    settings: Settings,
    overrides: ServeOverrides | None = None,
) -> RuntimeParams:
    """Merge defaults <- per-model overrides <- CLI overrides into one RuntimeParams.

    Layers never hard-code knowledge of a specific model name.
    """
    overrides = overrides or ServeOverrides()
    s = spec
    model_path, hf_ref = resolve_model_source(spec, settings, overrides)
    return RuntimeParams(
        host=_first(overrides.host, s.host if s else None, defaults.host, settings.host),
        port=_first(overrides.port, s.port if s else None, settings.base_port),
        n_gpu_layers=_first(
            overrides.n_gpu_layers, s.n_gpu_layers if s else None, defaults.n_gpu_layers
        ),
        flash_attn=_first(s.flash_attn if s else None, defaults.flash_attn),
        batch_size=_first(s.batch_size if s else None, defaults.batch_size),
        ubatch_size=_first(s.ubatch_size if s else None, defaults.ubatch_size),
        ctx_size=_first(overrides.ctx_size, s.ctx_size if s else None),
        n_parallel=_first(s.n_parallel if s else None, defaults.n_parallel),
        cache_type_k=_first(s.cache_type_k if s else None, defaults.cache_type_k),
        cache_type_v=_first(s.cache_type_v if s else None, defaults.cache_type_v),
        model_path=model_path,
        hf_ref=hf_ref,
        kind=s.kind if s else ModelKind.chat,
    )


_VALUED_FLAGS = {
    "-m",
    "-hf",
    "--host",
    "--port",
    "--n-gpu-layers",
    "--flash-attn",
    "--batch-size",
    "--ubatch-size",
    "--ctx-size",
    "--parallel",
    "--cache-type-k",
    "--cache-type-v",
    "--mmproj",
    "--temp",
    "--top-p",
    "--top-k",
    "--min-p",
}
_DEDUP_FLAGS = {"--embeddings", "--reranking", "--jinja"}


def _dedupe(argv: list[str]) -> list[str]:
    """Drop repeated boolean kind flags (they may also appear in extra_args)."""
    seen: set[str] = set()
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _VALUED_FLAGS:
            out.extend(argv[i : i + 2])
            i += 2
            continue
        if tok in _DEDUP_FLAGS and tok in seen:
            i += 1
            continue
        if tok.startswith("-"):
            seen.add(tok)
        out.append(tok)
        i += 1
    return out


def build_argv(
    spec: ModelSpec | None,
    defaults: Defaults,
    settings: Settings,
    overrides: ServeOverrides | None = None,
) -> list[str]:
    """Merge defaults <- kind flags <- model overrides <- CLI overrides into llama-server argv."""
    overrides = overrides or ServeOverrides()
    rt = merge_runtime(spec, defaults, settings, overrides)
    argv: list[str] = [str(binary_path(settings))]

    if rt.model_path is not None:
        argv.extend(["-m", str(rt.model_path)])
    else:
        argv.extend(["-hf", str(rt.hf_ref)])

    argv.extend(["--host", rt.host, "--port", str(rt.port)])
    argv.extend(["--n-gpu-layers", str(rt.n_gpu_layers)])
    argv.extend(["--flash-attn", str(rt.flash_attn)])
    argv.extend(["--batch-size", str(rt.batch_size), "--ubatch-size", str(rt.ubatch_size)])
    if rt.ctx_size is not None:
        argv.extend(["--ctx-size", str(rt.ctx_size)])
    if rt.n_parallel is not None:
        argv.extend(["--parallel", str(rt.n_parallel)])
    if rt.cache_type_k:
        argv.extend(["--cache-type-k", rt.cache_type_k])
    if rt.cache_type_v:
        argv.extend(["--cache-type-v", rt.cache_type_v])

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

    argv = _dedupe(argv)
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


def wait_healthy(port: int, timeout_s: float, poll_s: float, host: str = "127.0.0.1") -> None:
    url = f"http://{host}:{port}/health"
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
    raise TimeoutError(f"server on {host}:{port} not healthy within {timeout_s}s ({last_err})")


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
