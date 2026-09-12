"""Cross-model report over persisted eval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from spark_llm.config import Settings
from spark_llm.evals.runs import RunRecord, evals_root, list_runs, load_run

console = Console()

COLUMNS = [
    ("model", "model"),
    ("task", "task"),
    ("n", "n"),
    ("score", "score"),
    ("skipped", "skipped"),
    ("ttft p50 s", "ttft_p50_s"),
    ("prompt t/s", "prompt_tps_p50"),
    ("decode t/s", "decode_tps_p50"),
    ("tokens", "total_tokens"),
    ("ctx", "n_ctx"),
    ("build", "build"),
    ("note", "note"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def row_for(rec: RunRecord) -> dict[str, Any]:
    s = rec.summary
    return {
        "model": rec.model,
        "task": rec.task,
        "n": s.get("n", rec.n_results),
        "score": s.get("score"),
        "skipped": s.get("skipped", 0),
        "ttft_p50_s": s.get("ttft_p50_s"),
        "prompt_tps_p50": s.get("prompt_tps_p50"),
        "decode_tps_p50": s.get("decode_tps_p50"),
        "total_tokens": s.get("total_tokens"),
        "n_ctx": rec.server.get("n_ctx_per_slot"),
        "note": s.get("note") or s.get("aborted"),
        "build": f"{rec.provenance.llama_cpp_checkout or rec.provenance.llama_cpp_pinned or '?'}"
        f"/{rec.provenance.cuda_arch or '?'}",
        "config_hash": rec.config_hash,
        "run_dir": None,
    }


def latest_runs(settings: Settings, suite: str, models: list[str] | None = None) -> list[Path]:
    """Latest run per (model, task) for a suite."""
    latest: dict[tuple[str, str], Path] = {}
    for run_dir in list_runs(settings, suite):
        rec = load_run(run_dir)
        if models and rec.model not in models:
            continue
        if rec.finished is None:
            continue
        latest[(rec.model, rec.task)] = run_dir  # list_runs is sorted by timestamp
    return list(latest.values())


def build_report(run_dirs: list[Path]) -> tuple[Table, str]:
    rows = []
    for d in run_dirs:
        rec = load_run(d)
        row = row_for(rec)
        row["run_dir"] = str(d)
        rows.append(row)
    rows.sort(key=lambda r: (r["task"], -(r["score"] or 0)))

    hashes_by_task: dict[str, set[str]] = {}
    for r in rows:
        hashes_by_task.setdefault(r["task"], set()).add(r["config_hash"])

    table = Table(title="eval report")
    for title, _ in COLUMNS:
        table.add_column(title)
    md = ["| " + " | ".join(t for t, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        cells = [_fmt(r[key]) for _, key in COLUMNS]
        table.add_row(*cells)
        md.append("| " + " | ".join(cells) + " |")
    for task, hashes in hashes_by_task.items():
        if len(hashes) > 1:
            note = f"task {task}: runs use different configs ({', '.join(sorted(hashes))})"
            console.print(f"[yellow]warning[/yellow] {note}")
            md.append(f"\n> warning: {note}")
    return table, "\n".join(md) + "\n"


def write_report(settings: Settings, suite: str, markdown: str) -> Path:
    root = evals_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"report-{suite}.md"
    path.write_text(markdown)
    return path
