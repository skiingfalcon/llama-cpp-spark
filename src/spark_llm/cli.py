"""Typer CLI: build, models, download, serve, chat, bench, eval, doctor, stop.

Installed as ``local-llm`` (and the deprecated alias ``spark-llm``). Platform-specific behaviour
(build vs zip install, process control, telemetry, doctor checks) is delegated to
:mod:`spark_llm.platforms`; this module stays the same on the Spark and the Halo box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from spark_llm.bench import BenchOptions, compare_runs, print_bench_run, run_bench
from spark_llm.client import chat_once
from spark_llm.config import Settings, get_settings, repo_root
from spark_llm.download import download_hf_ref, download_model, resolve_weights
from spark_llm.evals.cli import eval_app
from spark_llm.evals.runs import BACKEND_ALIASES
from spark_llm.platforms import current as current_platform
from spark_llm.platforms.base import BuildOptions
from spark_llm.registry import ModelKind, ModelSpec, load_registry
from spark_llm.server import (
    ServeOverrides,
    argv_for_named_model,
    binary_path,
    build_argv,
    load_state,
    start_server,
    stop_servers,
)

app = typer.Typer(
    name="local-llm",
    help="llama.cpp inference and eval harness for local boxes (DGX Spark, AMD Strix Halo).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
app.add_typer(eval_app, name="eval")

BACKEND_OPT = typer.Option(
    None,
    "--backend",
    help="GPU backend of the llama.cpp binary: Spark cuda; Halo vulkan | rocm (LOCAL_LLM_BACKEND)",
)


@app.callback()
def _root() -> None:
    if Path(sys.argv[0]).stem == "spark-llm":
        console.print("[dim]note: spark-llm is now local-llm; the old name keeps working[/dim]")


def _passthrough(ctx: typer.Context) -> list[str]:
    return list(ctx.args) if ctx.args else []


def _with_backend(settings: Settings, backend: str | None) -> Settings:
    """Apply --backend for this invocation and validate the effective backend (flag or env)."""
    plat = current_platform()
    if backend is not None:
        settings = settings.model_copy(update={"backend": backend})
    effective = settings.backend
    if effective is not None:
        effective = BACKEND_ALIASES.get(effective.lower(), effective.lower())
        settings = settings.model_copy(update={"backend": effective})
    if effective is not None and effective not in plat.supported_backends():
        console.print(
            f"[red]backend {effective!r} not available on {plat.name}[/red]; "
            f"choose from {', '.join(plat.supported_backends())}"
        )
        raise typer.Exit(2)
    return settings


@app.command()
def build(
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Halo: vulkan | rocm | both (default both). Spark builds CUDA only.",
    ),
    source: str = typer.Option(
        "official", "--source", help="Halo rocm zip source: official (ggml-org) | lemonade"
    ),
    tag: str | None = typer.Option(None, "--tag", help="llama.cpp release tag (Halo zips)"),
    asset: str | None = typer.Option(None, "--asset", help="Exact release asset name (Halo)"),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already present"),
) -> None:
    """Build llama.cpp (Spark: pinned CMake/CUDA source build) or install prebuilt zips (Halo)."""
    settings = get_settings()
    backends = [b.strip() for b in backend.split(",") if b.strip()] if backend else []
    opts = BuildOptions(backends=backends, tag=tag, source=source, asset=asset, force=force)
    raise typer.Exit(current_platform().build(settings, opts))


@app.command("models")
def models_cmd() -> None:
    """List models registered in the platform's models*.toml."""
    settings = get_settings()
    reg = load_registry(settings=settings)
    table = Table(title=settings.models_toml.name)
    table.add_column("name")
    table.add_column("kind")
    table.add_column("port")
    table.add_column("source")
    table.add_column("local")
    for name, spec in sorted(reg.models.items()):
        src = spec.file or spec.hf_ref() or "-"
        local = resolve_weights(spec, settings.models_dir)
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
    name: str = typer.Argument(..., help="Registered model name from the registry"),
) -> None:
    """Download a registered model's GGUF(s) into LOCAL_LLM_MODELS_DIR."""
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
    backend: str | None = BACKEND_OPT,
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Stay attached (single model only)"
    ),
) -> None:
    """Start llama-server for one or more models (per-model ports).

    Extra args after `--` are appended verbatim to llama-server.
    """
    settings = _with_backend(get_settings(), backend)
    extra = _passthrough(ctx)
    names = names or []

    if not binary_path(settings).is_file():
        console.print(
            f"[red]llama-server missing[/red] ({binary_path(settings)}); run: local-llm build"
        )
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
    name: str = typer.Argument(..., help="Registered model (for port lookup)"),
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


def _int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def bench(
    ctx: typer.Context,
    names: list[str] | None = typer.Argument(None, help="Registered model name(s)"),
    pp: str = typer.Option("512,4096", "--pp", help="Prompt sizes (llama-bench -p)"),
    tg: str = typer.Option("128", "--tg", help="Generation sizes (llama-bench -n)"),
    depth: str = typer.Option(
        "0,16384,65536", "--depth", help="KV depths (llama-bench -d); filtered by ctx_size"
    ),
    reps: int = typer.Option(5, "--reps", help="Repetitions per test (llama-bench -r)"),
    force: bool = typer.Option(False, "--force", help="Bench even if the GPU is busy"),
    backend: str | None = BACKEND_OPT,
    compare: list[Path] | None = typer.Option(
        None, "--compare", help="Two saved state/bench/*.json files to diff (no bench run)"
    ),
) -> None:
    """Kernel-throughput smoke test (llama-bench) using the exact served configuration.

    Measures pp/tg tokens per second only. For task quality and serving latency use
    `local-llm eval`. Results are saved to state/bench/<timestamp>.json with build
    provenance. Extra args after `--` are appended verbatim to llama-bench.
    """
    if compare:
        if len(compare) != 2:
            console.print("[red]--compare takes exactly two files[/red]")
            raise typer.Exit(1)
        console.print(compare_runs(compare[0], compare[1]))
        return
    if not names:
        console.print("[red]provide at least one registered model name[/red]")
        raise typer.Exit(1)
    settings = _with_backend(get_settings(), backend)
    opts = BenchOptions(
        pp=_int_list(pp),
        tg=_int_list(tg),
        depth=_int_list(depth),
        reps=reps,
        force=force,
        extra_args=_passthrough(ctx),
    )
    try:
        run = run_bench(names, settings, opts)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    print_bench_run(run)


@app.command()
def doctor() -> None:
    """Check toolchain, binaries, GPU, and conflicting workloads for the detected platform."""
    settings = get_settings()
    plat = current_platform()
    ok = True

    def check(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        mark = "[green]ok[/green]" if good else "[red]FAIL[/red]"
        if not good:
            ok = False
        console.print(f"{mark}  {label}: {detail}")

    try:
        backend = plat.backend(settings)
    except ValueError as exc:
        backend = f"invalid ({exc})"
        ok = False
    console.print(f"[cyan]platform[/cyan] {plat.name} (backend {backend})")
    for c in plat.doctor_checks(settings):
        if c.note:
            console.print(f"[yellow]note[/yellow]  {c.label}: {c.detail}")
        else:
            check(c.label, c.ok, c.detail)

    check(settings.models_toml.name, settings.models_toml.is_file(), str(settings.models_toml))
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
