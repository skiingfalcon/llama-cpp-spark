"""Typer CLI: build, models, download, serve, chat, bench, doctor, stop."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from spark_llm.bench import print_bench_output, run_bench
from spark_llm.client import chat_once
from spark_llm.config import get_settings, repo_root
from spark_llm.download import download_hf_ref, download_model, resolve_local
from spark_llm.registry import ModelKind, ModelSpec, load_registry
from spark_llm.server import (
    ServeOverrides,
    argv_for_named_model,
    binary_path,
    build_argv,
    load_state,
    project_build_script,
    start_server,
    stop_servers,
)

app = typer.Typer(
    name="spark-llm",
    help="llama.cpp CUDA inference on NVIDIA DGX Spark (GB10 / sm_121).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _passthrough(ctx: typer.Context) -> list[str]:
    return list(ctx.args) if ctx.args else []


@app.command()
def build() -> None:
    """Clone/pin and compile llama.cpp with CUDA for sm_121a."""
    script = project_build_script()
    if not script.is_file():
        console.print(f"[red]missing[/red] {script}")
        raise typer.Exit(1)
    raise typer.Exit(subprocess.call(["bash", str(script)]))


@app.command("models")
def models_cmd() -> None:
    """List models registered in models.toml."""
    settings = get_settings()
    reg = load_registry(settings=settings)
    table = Table(title="models.toml")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("port")
    table.add_column("source")
    table.add_column("local")
    for name, spec in sorted(reg.models.items()):
        src = spec.file or spec.hf_ref() or "-"
        local = resolve_local(spec, settings.models_dir)
        table.add_row(
            name,
            spec.kind.value,
            str(spec.port or "-"),
            src,
            "yes" if local else "no",
        )
    console.print(table)


@app.command()
def download(
    name: str = typer.Argument(..., help="Registered model name from models.toml"),
) -> None:
    """Download a registered model's GGUF(s) into SPARK_LLM_MODELS_DIR."""
    settings = get_settings()
    reg = load_registry(settings=settings)
    path = download_model(reg.get(name), settings)
    console.print(f"[green]ok[/green] {path}")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def serve(
    ctx: typer.Context,
    names: list[str] | None = typer.Argument(
        None, help="Registered model name(s); omit when using --hf / --model-path"
    ),
    hf: str | None = typer.Option(
        None, "--hf", help="Unregistered HuggingFace GGUF ref, e.g. org/repo:Q4_K_M"
    ),
    model_path: Path | None = typer.Option(
        None, "--model-path", help="Local .gguf path (bypass registry)"
    ),
    port: int | None = typer.Option(None, "--port", help="Override listen port"),
    host: str | None = typer.Option(None, "--host", help="Override bind host"),
    n_gpu_layers: int | None = typer.Option(None, "--n-gpu-layers", "-ngl"),
    ctx_size: int | None = typer.Option(None, "--ctx-size", "-c"),
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Stay attached (single model only)"
    ),
) -> None:
    """Start llama-server for one or more models (per-model ports).

    Extra args after `--` are appended verbatim to llama-server.
    """
    settings = get_settings()
    extra = _passthrough(ctx)
    names = names or []

    if not binary_path(settings).is_file():
        console.print("[red]llama-server missing[/red]; run: spark-llm build")
        raise typer.Exit(1)

    if hf or model_path:
        if len(names) > 1:
            console.print("[red]--hf / --model-path serve one model at a time[/red]")
            raise typer.Exit(1)

        # Resolve --hf via huggingface_hub; llama-server native -hf needs OpenSSL HTTPS.
        if hf:
            resolved_path = download_hf_ref(hf, settings)
            label = names[0] if names else hf.replace("/", "_").replace(":", "_")
        else:
            assert model_path is not None
            resolved_path = model_path
            label = names[0] if names else model_path.name

        if not resolved_path.is_file():
            console.print(f"[red]not found[/red] {resolved_path}")
            raise typer.Exit(1)

        overrides = ServeOverrides(
            host=host,
            port=port or settings.base_port,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            model_path=resolved_path,
            extra_args=extra,
        )
        anon = ModelSpec(
            name=label,
            kind=ModelKind.chat,
            port=overrides.port,
            file=resolved_path.name,
            extra_args=["--jinja"] if not any(a == "--jinja" for a in extra) else [],
        )
        reg = load_registry(settings=settings)
        argv = build_argv(anon, reg.defaults, settings, overrides)
        listen = overrides.port or settings.base_port
        start_server(argv, label, listen, settings, foreground=foreground)
        if foreground:
            raise typer.Exit(0)
        return

    if not names:
        console.print("[red]provide a model name, or --hf / --model-path[/red]")
        raise typer.Exit(1)

    if foreground and len(names) > 1:
        console.print("[red]--foreground only supports a single model[/red]")
        raise typer.Exit(1)

    for name in names:
        overrides = ServeOverrides(
            host=host,
            port=port if len(names) == 1 else None,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            extra_args=extra if len(names) == 1 else [],
        )
        argv, listen = argv_for_named_model(name, overrides, settings)
        start_server(argv, name, listen, settings, foreground=foreground and len(names) == 1)


@app.command()
def stop(
    names: list[str] | None = typer.Argument(
        None, help="Server name(s) to stop; default: all tracked"
    ),
) -> None:
    """Stop background llama-server processes tracked in state/servers.json."""
    stop_servers(names)


@app.command()
def chat(
    name: str = typer.Argument("gpt-oss-20b", help="Registered model (for port lookup)"),
    message: str = typer.Option(..., "--message", "-m", help="User message"),
    port: int | None = typer.Option(None, "--port"),
    system: str | None = typer.Option(None, "--system"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
) -> None:
    """Send a chat completion to a running server."""
    settings = get_settings()
    if port is None:
        reg = load_registry(settings=settings)
        port = reg.get(name).port or settings.base_port
    result = chat_once(port, message, system=system, model=name, stream=stream)
    if stream:
        for chunk in result:  # type: ignore[union-attr]
            console.print(chunk, end="")
        console.print()
    else:
        console.print(result)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def bench(
    ctx: typer.Context,
    name: str = typer.Argument("gpt-oss-20b"),
) -> None:
    """Run llama-bench for a registered model."""
    settings = get_settings()
    reg = load_registry(settings=settings)
    raw = run_bench(reg.get(name), settings, extra_args=_passthrough(ctx))
    print_bench_output(raw)


@app.command()
def doctor() -> None:
    """Check toolchain, binaries, GPU, and conflicting workloads."""
    settings = get_settings()
    ok = True

    def check(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        mark = "[green]ok[/green]" if good else "[red]FAIL[/red]"
        if not good:
            ok = False
        console.print(f"{mark}  {label}: {detail}")

    check("arch", os.uname().machine == "aarch64", os.uname().machine)
    nvcc = shutil.which("nvcc")
    check("nvcc", bool(nvcc), nvcc or "not found")
    if nvcc:
        ver = subprocess.check_output([nvcc, "--version"], text=True)
        line = [ln for ln in ver.splitlines() if "release" in ln.lower()]
        check("cuda toolkit", bool(line), line[-1].strip() if line else ver.strip())

    smi = shutil.which("nvidia-smi")
    check("nvidia-smi", bool(smi), smi or "not found")
    if smi:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        check("gpu", "GB10" in out or "12.1" in out, out)
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        tracked_pids = {int(info["pid"]) for info in load_state(settings).values()}
        foreign = []
        ours = []
        if procs:
            for line in procs.splitlines():
                pid_s = line.split(",", 1)[0].strip()
                try:
                    pid = int(pid_s)
                except ValueError:
                    foreign.append(line)
                    continue
                (ours if pid in tracked_pids else foreign).append(line)
        if foreign:
            check("gpu free", False, "; ".join(foreign))
        elif ours:
            console.print(f"[yellow]note[/yellow]  gpu in use by spark-llm: {'; '.join(ours)}")
        else:
            check("gpu free", True, "no compute apps")

    cmake = shutil.which("cmake")
    check("cmake", bool(cmake), cmake or "not found (uv tool install 'cmake>=3.30')")
    server = binary_path(settings)
    check("llama-server", server.is_file(), str(server))
    if server.is_file():
        from spark_llm.server import runtime_env

        try:
            ver = subprocess.check_output(
                [str(server), "--version"],
                text=True,
                stderr=subprocess.STDOUT,
                env=runtime_env(settings),
            )
            check("llama-server version", True, ver.splitlines()[0][:120])
        except Exception as exc:  # noqa: BLE001
            check("llama-server version", False, str(exc))

    check("models.toml", settings.models_toml.is_file(), str(settings.models_toml))
    check("models_dir", settings.models_dir.is_dir(), str(settings.models_dir))
    check("repo", (repo_root() / "pyproject.toml").is_file(), str(repo_root()))

    state = load_state(settings)
    if state:
        console.print(f"[cyan]tracked servers[/cyan] {', '.join(state)}")
        for name, info in state.items():
            try:
                r = httpx.get(f"http://127.0.0.1:{info['port']}/health", timeout=1.0)
                healthy = r.status_code == 200
            except Exception:
                healthy = False
            check(f"server:{name}", healthy, f"pid={info['pid']} port={info['port']}")

    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
