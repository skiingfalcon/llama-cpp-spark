"""Cross-model report over persisted eval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from spark_llm.config import Settings
from spark_llm.evals.runs import RunRecord, evals_root, list_runs, load_results, load_run

console = Console()

COLUMNS = [
    ("model", "model"),
    ("task", "task"),
    ("n", "n"),
    ("score", "score"),
    ("skipped", "skipped"),
    ("truncated", "truncated"),
    ("ttft p50 s", "ttft_p50_s"),
    ("total p50 s", "total_p50_s"),
    ("prompt t/s", "prompt_tps_p50"),
    ("decode t/s", "decode_tps_p50"),
    ("tokens", "total_tokens"),
    ("cached", "cached_prompt_tokens"),
    ("reasoning", "reasoning_tokens"),
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
    provider = rec.server.get("provider")
    build = (
        f"{provider}/api"
        if provider
        else f"{rec.provenance.llama_cpp_checkout or rec.provenance.llama_cpp_pinned or '?'}"
        f"/{rec.provenance.cuda_arch or '?'}"
    )
    return {
        "model": rec.model,
        "task": rec.task,
        "n": s.get("n", rec.n_results),
        "score": s.get("score"),
        "skipped": s.get("skipped", 0),
        "truncated": s.get("truncated"),
        "ttft_p50_s": s.get("ttft_p50_s"),
        "total_p50_s": s.get("total_p50_s"),
        "prompt_tps_p50": s.get("prompt_tps_p50"),
        "decode_tps_p50": s.get("decode_tps_p50"),
        "total_tokens": s.get("total_tokens"),
        "cached_prompt_tokens": s.get("cached_prompt_tokens"),
        "reasoning_tokens": s.get("reasoning_tokens"),
        "n_ctx": rec.server.get("n_ctx_per_slot"),
        "note": s.get("note") or s.get("aborted"),
        "build": build,
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


PAIRED_COLUMNS = [
    ("task", "task"),
    ("model", "model"),
    ("paired n", "n"),
    ("correct", "correct"),
    ("paired score", "score"),
]


def paired_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Like-for-like scores over the items every run of a task answered (none skipped).

    Raw ``score`` divides by each run's own answered set, so a model that skipped the hard
    filings (context limit, rate limit) looks better than one that attempted them. Pairing
    fixes the denominator.
    """
    by_task: dict[str, list[tuple[RunRecord, dict[str, dict[str, Any]]]]] = {}
    for d in run_dirs:
        rec = load_run(d)
        results = {r["id"]: r for r in load_results(d) if r.get("id")}
        if results:
            by_task.setdefault(rec.task, []).append((rec, results))
    rows: list[dict[str, Any]] = []
    for task, runs in sorted(by_task.items()):
        if len(runs) < 2:
            continue
        answered = [
            {i for i, r in results.items() if not r.get("skipped") and r.get("correct") is not None}
            for _, results in runs
        ]
        common = set.intersection(*answered)
        for rec, results in runs:
            correct = sum(int(bool(results[i]["correct"])) for i in common)
            rows.append(
                {
                    "task": task,
                    "model": rec.model,
                    "n": len(common),
                    "correct": correct,
                    "score": correct / len(common) if common else None,
                }
            )
    rows.sort(key=lambda r: (r["task"], -(r["score"] or 0)))
    return rows


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
    paired = paired_rows(run_dirs)
    if paired:
        md.append("\n### Paired (items answered by every run of the task)\n")
        md.append("| " + " | ".join(t for t, _ in PAIRED_COLUMNS) + " |")
        md.append("|" + "---|" * len(PAIRED_COLUMNS))
        ptable = Table(title="paired")
        for title, _ in PAIRED_COLUMNS:
            ptable.add_column(title)
        for r in paired:
            cells = [_fmt(r[key]) for _, key in PAIRED_COLUMNS]
            ptable.add_row(*cells)
            md.append("| " + " | ".join(cells) + " |")
        console.print(ptable)
    return table, "\n".join(md) + "\n"


def write_report(settings: Settings, suite: str, markdown: str) -> Path:
    root = evals_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"report-{suite}.md"
    path.write_text(markdown)
    return path
