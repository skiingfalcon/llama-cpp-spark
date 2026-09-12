"""Wrap llama-bench and present results with rich."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from spark_llm.config import Settings, get_settings
from spark_llm.download import resolve_local
from spark_llm.registry import ModelSpec
from spark_llm.server import runtime_env

console = Console()


def bench_binary(settings: Settings) -> Path:
    return settings.vendor_dir / "build" / "bin" / "llama-bench"


def run_bench(
    spec: ModelSpec,
    settings: Settings | None = None,
    extra_args: list[str] | None = None,
) -> str:
    settings = settings or get_settings()
    binary = bench_binary(settings)
    if not binary.is_file():
        raise FileNotFoundError(f"llama-bench not found at {binary}; run make build")

    model_path = resolve_local(spec, settings.models_dir)
    if model_path is None and spec.file:
        model_path = settings.models_dir / spec.file
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            f"model weights missing for {spec.name}; run: spark-llm download {spec.name}"
        )

    ngl = spec.n_gpu_layers if spec.n_gpu_layers is not None else settings.n_gpu_layers
    argv = [
        str(binary),
        "-m",
        str(model_path),
        "-ngl",
        str(ngl),
        "-fa",
        "1",
    ]
    if extra_args:
        argv.extend(extra_args)

    console.print("[cyan]bench[/cyan] " + " ".join(argv))
    result = subprocess.run(
        argv,
        env=runtime_env(settings),
        check=False,
        capture_output=True,
        text=True,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        console.print(out)
        raise RuntimeError(f"llama-bench exited {result.returncode}")
    return out


def print_bench_output(raw: str) -> None:
    """Best-effort parse of llama-bench pipe/table output into a rich table."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    # Prefer lines that look like the summary table (contain | or start with model)
    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        console.print(raw)
        return
    table = Table(title="llama-bench")
    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    for h in headers:
        table.add_column(h)
    for ln in table_lines[1:]:
        if set(ln.strip()) <= {"|", "-", " "}:
            continue
        cols = [c.strip() for c in ln.split("|") if c.strip() or True]
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) == len(headers):
            table.add_row(*cols)
    console.print(table)
    # Also dump raw for anything we missed
    console.print("\n[dim]raw output[/dim]")
    console.print(raw)
