"""Kernel-throughput smoke test: wrap llama-bench with the *served* configuration.

This measures raw prompt-processing / token-generation speed for a GGUF on this build.
It says nothing about task quality or serving behaviour under load; use ``local-llm eval``
for those. What it does guarantee is that the numbers are taken with the same batch,
ubatch, GPU-layer, flash-attention and KV-cache settings that ``local-llm serve`` uses,
and that every run is persisted with build provenance under ``state/bench/``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from spark_llm.config import Settings, get_settings
from spark_llm.console import err
from spark_llm.console import out as console
from spark_llm.gpu import busy_reasons
from spark_llm.platforms import current as current_platform
from spark_llm.provenance import Provenance, collect
from spark_llm.registry import Defaults, ModelKind, ModelSpec, load_registry
from spark_llm.server import RuntimeParams, merge_runtime, runtime_env


@dataclass
class BenchOptions:
    pp: list[int] = field(default_factory=lambda: [512, 4096])
    tg: list[int] = field(default_factory=lambda: [128])
    depth: list[int] = field(default_factory=lambda: [0, 16384, 65536])
    reps: int = 5
    force: bool = False
    extra_args: list[str] = field(default_factory=list)


def bench_binary(settings: Settings) -> Path:
    return current_platform().bench_binary(settings)


def _flash_attn_flag(value: str) -> str:
    v = value.strip().lower()
    if v in {"on", "1", "true", "yes"}:
        return "1"
    if v in {"off", "0", "false", "no"}:
        return "0"
    return v  # "auto" on builds that support it


def usable_depths(
    depths: list[int], pp: list[int], ctx_size: int | None
) -> tuple[list[int], list[int]]:
    """Split depths into (kept, skipped) so depth + max(pp) never exceeds a fixed ctx.

    ``ctx_size`` of None/0 means "model maximum"; nothing is filtered in that case.
    """
    if not ctx_size:
        return list(depths), []
    limit = ctx_size - max(pp or [0])
    kept = [d for d in depths if d <= max(limit, 0)]
    skipped = [d for d in depths if d not in kept]
    return kept, skipped


def bench_argv(
    spec: ModelSpec,
    defaults: Defaults,
    settings: Settings,
    opts: BenchOptions | None = None,
) -> tuple[list[str], RuntimeParams, list[int]]:
    """llama-bench argv derived from the same merged runtime as llama-server.

    Returns (argv, merged params, skipped depths).
    """
    opts = opts or BenchOptions()
    rt = merge_runtime(spec, defaults, settings)
    if rt.model_path is None or not rt.model_path.is_file():
        raise FileNotFoundError(
            f"model weights missing for {spec.name}; run: local-llm download {spec.name}"
        )
    depths, skipped = usable_depths(opts.depth, opts.pp, rt.ctx_size)
    argv = [
        str(bench_binary(settings)),
        "-m",
        str(rt.model_path),
        "-ngl",
        str(rt.n_gpu_layers),
        "-fa",
        _flash_attn_flag(rt.flash_attn),
        "-b",
        str(rt.batch_size),
        "-ub",
        str(rt.ubatch_size),
        "-p",
        ",".join(str(x) for x in opts.pp),
        "-n",
        ",".join(str(x) for x in opts.tg),
        "-d",
        ",".join(str(x) for x in depths),
        "-r",
        str(opts.reps),
        "-o",
        "json",
    ]
    if rt.cache_type_k:
        argv.extend(["-ctk", rt.cache_type_k])
    if rt.cache_type_v:
        argv.extend(["-ctv", rt.cache_type_v])
    if spec.kind is ModelKind.embedding:
        argv.extend(["-embd", "1"])
    if spec.kind is ModelKind.multimodal and spec.mmproj:
        err.print("[yellow]note[/yellow] llama-bench ignores --mmproj; text path only")
    argv.extend(opts.extra_args)
    return argv, rt, skipped


def parse_bench_json(stdout: str) -> list[dict]:
    """llama-bench -o json emits a JSON array on stdout; tolerate leading log noise."""
    text = stdout.strip()
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array in llama-bench output")
    return json.loads(text[start:])


@dataclass
class ModelBench:
    name: str
    argv: list[str]
    runtime: dict
    skipped_depths: list[int]
    results: list[dict]
    stderr_tail: str


@dataclass
class BenchRun:
    provenance: Provenance
    options: dict
    models: list[ModelBench]
    path: Path | None = None

    def to_json(self) -> dict:
        return {
            "provenance": self.provenance.model_dump(),
            "options": self.options,
            "models": [m.__dict__ for m in self.models],
        }


def bench_dir(settings: Settings) -> Path:
    d = settings.state_dir / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_bench(
    names: list[str],
    settings: Settings | None = None,
    opts: BenchOptions | None = None,
) -> BenchRun:
    settings = settings or get_settings()
    opts = opts or BenchOptions()
    binary = bench_binary(settings)
    if not binary.is_file():
        raise FileNotFoundError(f"llama-bench not found at {binary}; run make build")

    reasons = busy_reasons(settings)
    if reasons and not opts.force:
        raise RuntimeError(
            "GPU is not idle; results would not be comparable:\n  - "
            + "\n  - ".join(reasons)
            + "\n(use --force to bench anyway)"
        )

    registry = load_registry(settings=settings)
    env = runtime_env(settings)
    prov = collect(settings, binary=binary, env=env)
    run = BenchRun(
        provenance=prov,
        options={k: v for k, v in opts.__dict__.items() if k != "force"},
        models=[],
    )
    for name in names:
        spec = registry.get(name)
        argv, rt, skipped = bench_argv(spec, registry.defaults, settings, opts)
        if skipped:
            console.print(
                f"[yellow]skip[/yellow] {name}: depths {skipped} exceed ctx_size={rt.ctx_size}"
            )
        console.print("[cyan]bench[/cyan] " + " ".join(argv))
        proc = subprocess.run(argv, env=env, check=False, capture_output=True, text=True)
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        if proc.returncode != 0:
            console.print(proc.stderr or proc.stdout)
            raise RuntimeError(f"llama-bench exited {proc.returncode} for {name}")
        results = parse_bench_json(proc.stdout)
        run.models.append(
            ModelBench(
                name=name,
                argv=argv,
                runtime=rt.as_dict(),
                skipped_depths=skipped,
                results=results,
                stderr_tail=tail,
            )
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = bench_dir(settings) / f"{stamp}.json"
    path.write_text(json.dumps(run.to_json(), indent=2) + "\n")
    run.path = path
    return run


def _test_label(r: dict) -> str:
    label = f"pp{r.get('n_prompt', 0)}" if r.get("n_prompt") else f"tg{r.get('n_gen', 0)}"
    if r.get("n_prompt") and r.get("n_gen"):
        label = f"pp{r['n_prompt']}+tg{r['n_gen']}"
    if r.get("n_depth"):
        label += f" @d{r['n_depth']}"
    return label


def print_bench_run(run: BenchRun) -> None:
    p = run.provenance
    console.print(
        f"[dim]llama.cpp {p.llama_cpp_checkout or p.llama_cpp_pinned or '?'}"
        f"  build={p.build_id or 'unknown'}  gpu={p.gpu.get('name', '?')}"
        f"  driver={p.gpu.get('driver_version', '?')}[/dim]"
    )
    table = Table(title="llama-bench (served configuration)")
    for col in ("model", "size", "params", "ngl", "b/ub", "fa", "test", "t/s", "±"):
        table.add_column(col, justify="right" if col in {"t/s", "±"} else "left")
    for m in run.models:
        for r in m.results:
            size_gb = float(r.get("model_size", 0)) / 1e9
            params_b = float(r.get("model_n_params", 0)) / 1e9
            table.add_row(
                m.name,
                f"{size_gb:.1f} GiB" if size_gb else "-",
                f"{params_b:.1f} B" if params_b else "-",
                str(r.get("n_gpu_layers", "")),
                f"{r.get('n_batch', '')}/{r.get('n_ubatch', '')}",
                str(r.get("flash_attn", "")),
                _test_label(r),
                f"{float(r.get('avg_ts', 0)):.2f}",
                f"{float(r.get('stddev_ts', 0)):.2f}",
            )
        if m.skipped_depths:
            table.add_row(m.name, "", "", "", "", "", f"skipped depths {m.skipped_depths}", "", "")
    console.print(table)
    if run.path:
        console.print(f"[dim]saved {run.path}[/dim]")


def _build(prov: dict[str, Any]) -> Any:
    return prov.get("build_id") or prov.get("cuda_arch")  # pre-rename files say cuda_arch


def compare_runs(a_path: Path, b_path: Path) -> Table:
    """Side-by-side t/s for matching (model, test) rows across two saved runs."""
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    def index(run: dict) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for m in run["models"]:
            for r in m["results"]:
                out[(m["name"], _test_label(r))] = float(r.get("avg_ts", 0))
        return out

    ia, ib = index(a), index(b)
    table = Table(title=f"{a_path.name} vs {b_path.name}")
    for col in ("model", "test", "A t/s", "B t/s", "Δ%"):
        table.add_column(col, justify="right" if "t/s" in col or col == "Δ%" else "left")
    pa, pb = a.get("provenance", {}), b.get("provenance", {})
    if _build(pa) != _build(pb) or pa.get("llama_cpp_checkout") != pb.get("llama_cpp_checkout"):
        console.print(
            "[yellow]warning[/yellow] runs differ in build "
            f"(A: {pa.get('llama_cpp_checkout')}/{_build(pa)}, "
            f"B: {pb.get('llama_cpp_checkout')}/{_build(pb)})"
        )
    for key in sorted(set(ia) | set(ib)):
        va, vb = ia.get(key), ib.get(key)
        delta = f"{(vb - va) / va * 100:+.1f}" if va and vb else "-"
        table.add_row(
            key[0],
            key[1],
            f"{va:.2f}" if va is not None else "-",
            f"{vb:.2f}" if vb is not None else "-",
            delta,
        )
    return table
