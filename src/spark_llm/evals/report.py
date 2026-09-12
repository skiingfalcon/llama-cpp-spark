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
    ("platform", "platform"),
    ("note", "note"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def platform_key(rec: RunRecord) -> tuple[str, str | None]:
    """(platform, backend) a run was served from; API runs are platform-agnostic.

    Runs recorded before provenance carried these fields were all Spark/CUDA.
    """
    if rec.server.get("provider"):
        return ("api", None)
    prov = rec.provenance
    return (prov.platform or "spark", prov.backend or "cuda")


def platform_label(rec: RunRecord) -> str:
    plat, backend = platform_key(rec)
    return plat if backend is None else f"{plat}/{backend}"


def build_label(rec: RunRecord) -> str:
    provider = rec.server.get("provider")
    if provider:
        return f"{provider}/api"
    prov = rec.provenance
    ident = prov.llama_cpp_release or prov.llama_cpp_checkout or prov.llama_cpp_pinned or "?"
    return f"{ident}/{prov.cuda_arch or '?'}"


def row_for(rec: RunRecord) -> dict[str, Any]:
    s = rec.summary
    build = build_label(rec)
    return {
        "model": rec.model,
        "platform": platform_label(rec),
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
    """Latest run per (model, task, platform, backend) for a suite.

    Keying on the platform keeps a Halo run of gpt-oss-120b from silently replacing the Spark
    run of the same model in the table; the two show up side by side instead.
    """
    latest: dict[tuple[str, str, str, str | None], Path] = {}
    for run_dir in list_runs(settings, suite):
        rec = load_run(run_dir)
        if models and rec.model not in models:
            continue
        if rec.finished is None:
            continue
        latest[(rec.model, rec.task, *platform_key(rec))] = run_dir  # list_runs sorted by time
    return list(latest.values())


PAIRED_COLUMNS = [
    ("task", "task"),
    ("model", "model"),
    ("paired n", "n"),
    ("correct", "correct"),
    ("paired score", "score"),
]


def _answered(r: dict[str, Any]) -> bool:
    return not r.get("skipped") and r.get("correct") is not None


def _is_oss(model: str) -> bool:
    return "gpt-oss" in model


def _is_frontier(model: str) -> bool:
    return model.startswith("openai:") or "terra" in model.lower()


def _load_task_runs(
    run_dirs: list[Path],
) -> dict[str, list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]]:
    by_task: dict[str, list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]] = {}
    for d in run_dirs:
        rec = load_run(d)
        results = {r["id"]: r for r in load_results(d) if r.get("id")}
        if results:
            by_task.setdefault(rec.task, []).append((rec, d, results))
    return by_task


def paired_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Like-for-like scores over the items every run of a task answered (none skipped).

    Raw ``score`` divides by each run's own answered set, so a model that skipped the hard
    filings (context limit, rate limit) looks better than one that attempted them. Pairing
    fixes the denominator.
    """
    rows: list[dict[str, Any]] = []
    for task, runs in sorted(_load_task_runs(run_dirs).items()):
        if len(runs) < 2:
            continue
        answered = [{i for i, r in results.items() if _answered(r)} for _, _, results in runs]
        common = set.intersection(*answered)
        for rec, _, results in runs:
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


def _score_on(results: dict[str, dict[str, Any]], ids: set[str]) -> tuple[int, int, float | None]:
    hits = [results[i] for i in ids if i in results and _answered(results[i])]
    if not hits:
        return 0, 0, None
    correct = sum(int(bool(r["correct"])) for r in hits)
    return correct, len(hits), correct / len(hits)


def _frac(correct: int, n: int) -> str:
    if not n:
        return "-"
    return f"{correct}/{n} ({correct / n:.3f})"


def _verdict(r: dict[str, Any] | None) -> str:
    if r is None:
        return "-"
    if r.get("skipped") or r.get("correct") is None:
        return "skip"
    return "ok" if r["correct"] else "miss"


def _mark(r: dict[str, Any] | None) -> str:
    flag = _verdict(r)
    if flag in {"-", "skip"} or r is None:
        return flag
    mode = r.get("mode") or ""
    if r.get("fallback") or (mode and mode != "full"):
        return f"{flag}/{mode or 'fb'}"
    return flag


def _render_md(title: str, columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> str:
    lines = [f"\n### {title}\n", "| " + " | ".join(t for t, _ in columns) + " |"]
    lines.append("|" + "---|" * len(columns))
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[key]) for _, key in columns) + " |")
    return "\n".join(lines)


def _rich_table(title: str, columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> Table:
    table = Table(title=title)
    for heading, _ in columns:
        table.add_column(heading)
    for r in rows:
        table.add_row(*(_fmt(r[key]) for _, key in columns))
    return table


def comparison_group(
    run_dirs: list[Path],
) -> dict[str, list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]]:
    """Latest gpt-oss runs plus the latest OpenAI/Terra run, grouped by task (and platform).

    Emits a group only when at least two gpt-oss runs from the same platform/backend and one
    frontier run are present. With runs from several platforms the key becomes
    ``"<task> [<platform>]"`` so each box gets its own block; the frontier run is shared.
    """
    out: dict[str, list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]] = {}
    for task, runs in _load_task_runs(run_dirs).items():
        frontier = [r for r in runs if _is_frontier(r[0].model)]
        if not frontier:
            continue
        oss_by_platform: dict[str, list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]] = {}
        for r in runs:
            if _is_oss(r[0].model):
                oss_by_platform.setdefault(platform_label(r[0]), []).append(r)
        eligible = {p: rs for p, rs in oss_by_platform.items() if len(rs) >= 2}
        for plat, oss in sorted(eligible.items()):
            key = task if len(eligible) == 1 else f"{task} [{plat}]"
            out[key] = sorted(oss, key=lambda r: r[0].model) + frontier[-1:]
    return out


def _group_rows(
    runs: list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]], field: str
) -> list[dict[str, Any]]:
    keys = sorted({(r.get(field) or "?") for _, _, res in runs for r in res.values()})
    out: list[dict[str, Any]] = []
    for key in keys:
        ids = {i for _, _, res in runs for i, r in res.items() if (r.get(field) or "?") == key}
        if not ids:
            continue
        row: dict[str, Any] = {field: key, "n": len(ids)}
        for rec, _, results in runs:
            c, n, _ = _score_on(results, ids)
            row[rec.model] = _frac(c, n)
        out.append(row)
    return out


def comparison_blocks(
    run_dirs: list[Path],
) -> tuple[list[Table], str]:
    """Headline slices, per-tag / per-company scores, and disagreements for oss vs Terra."""
    tables: list[Table] = []
    md_parts: list[str] = []
    for task, runs in sorted(comparison_group(run_dirs).items()):
        models = [rec.model for rec, _, _ in runs]
        dirs = [d.name for _, d, _ in runs]
        heading = f"{task}: gpt-oss vs Terra"
        md_parts.append(f"\n### {heading}\n")
        md_parts.append(
            "Latest finished gpt-oss-20b, gpt-oss-120b, and OpenAI/Terra runs. "
            "`full` = whole filing in context; `section`/`chunked` = oversized-filing fallback. "
            "Terra's 1.05M window still sees GS/STWD in full.\n"
        )
        compared = ", ".join(f"{m} (`{d}`)" for m, d in zip(models, dirs, strict=True))
        md_parts.append(f"Compared: {compared}\n")

        all_ids = set().union(*(results.keys() for _, _, results in runs))
        answered = [{i for i, r in results.items() if _answered(r)} for _, _, results in runs]
        paired = set.intersection(*answered) if answered else set()
        oss_results = [results for rec, _, results in runs if _is_oss(rec.model)]
        full_ids = {
            i
            for i in all_ids
            if oss_results
            and all(
                (r := res.get(i)) is not None
                and not r.get("fallback")
                and (r.get("mode") or "full") == "full"
                for res in oss_results
            )
        }
        fallback_ids = {
            i
            for i in all_ids
            if any(bool((res.get(i) or {}).get("fallback")) for res in oss_results)
        }

        slice_cols = [("slice", "slice"), ("n", "n")] + [(m, m) for m in models]
        slice_rows: list[dict[str, Any]] = []
        for label, ids in (
            ("own scored (raw)", all_ids),
            ("paired (all three answered)", paired),
            ("full document (no OSS fallback)", full_ids),
            ("OSS fallback filings", fallback_ids),
        ):
            row: dict[str, Any] = {
                "slice": label,
                "n": "-" if label == "own scored (raw)" else len(ids),
            }
            for rec, _, results in runs:
                target = set(results) if label == "own scored (raw)" else ids
                c, n, _ = _score_on(results, target)
                row[rec.model] = _frac(c, n)
            slice_rows.append(row)
        tables.append(_rich_table(heading, slice_cols, slice_rows))
        md_parts.append(_render_md("Accuracy slices", slice_cols, slice_rows))

        mode_cols = [("model", "model"), ("fallback", "fallback"), ("by_mode", "by_mode")]
        mode_rows = []
        for rec, _, _ in runs:
            by_mode = rec.summary.get("by_mode") or {}
            mode_rows.append(
                {
                    "model": rec.model,
                    "fallback": rec.summary.get("fallback", 0),
                    "by_mode": ", ".join(
                        f"{k} {v.get('correct', 0)}/{v.get('n', 0)}"
                        for k, v in sorted(by_mode.items())
                    )
                    or "-",
                }
            )
        tables.append(_rich_table(f"{task} context mode", mode_cols, mode_rows))
        md_parts.append(_render_md("Context mode", mode_cols, mode_rows))

        tag_cols = [("tag", "tag"), ("n", "n")] + [(m, m) for m in models]
        tag_rows = _group_rows(runs, "tag")
        if tag_rows:
            tables.append(_rich_table(f"{task} by tag", tag_cols, tag_rows))
            md_parts.append(_render_md("By tag", tag_cols, tag_rows))

        ticker_cols = [("ticker", "ticker"), ("n", "n")] + [(m, m) for m in models]
        ticker_rows = _group_rows(runs, "ticker")
        if ticker_rows:
            tables.append(_rich_table(f"{task} by company", ticker_cols, ticker_rows))
            md_parts.append(_render_md("By company", ticker_cols, ticker_rows))

        diffs: list[dict[str, Any]] = []
        for i in sorted(all_ids):
            verdicts = {_verdict(results.get(i)) for _, _, results in runs}
            if len(verdicts) <= 1:
                continue
            sample = next(results[i] for _, _, results in runs if i in results)
            diffs.append(
                {
                    "id": i,
                    **{rec.model: _mark(results.get(i)) for rec, _, results in runs},
                    "ticker": sample.get("ticker", "?"),
                }
            )
        if diffs:
            diff_cols = [("id", "id"), ("ticker", "ticker")] + [(m, m) for m in models]
            tables.append(_rich_table(f"{task} disagreements", diff_cols, diffs))
            md_parts.append(_render_md("Disagreements", diff_cols, diffs))

    return tables, "\n".join(md_parts)


HARDWARE_COLUMNS = [
    ("platform", "platform"),
    ("score", "score"),
    ("paired", "paired"),
    ("truncated", "truncated"),
    ("ttft p50 s", "ttft_p50_s"),
    ("total p50 s", "total_p50_s"),
    ("total p95 s", "total_p95_s"),
    ("prompt t/s", "prompt_tps_p50"),
    ("decode t/s", "decode_tps_p50"),
    ("ctx", "n_ctx"),
    ("gpu", "gpu"),
    ("build", "build"),
]


def _accuracy_hardware_rows(
    runs: list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    answered = [{i for i, r in results.items() if _answered(r)} for _, _, results in runs]
    common = set.intersection(*answered) if answered else set()
    rows: list[dict[str, Any]] = []
    for rec, _, results in runs:
        s = rec.summary
        c, n, _ = _score_on(results, common)
        rows.append(
            {
                "platform": platform_label(rec),
                "score": s.get("score"),
                "paired": _frac(c, n),
                "truncated": s.get("truncated"),
                "ttft_p50_s": s.get("ttft_p50_s"),
                "total_p50_s": s.get("total_p50_s"),
                "total_p95_s": s.get("total_p95_s"),
                "prompt_tps_p50": s.get("prompt_tps_p50"),
                "decode_tps_p50": s.get("decode_tps_p50"),
                "n_ctx": rec.server.get("n_ctx_per_slot"),
                "gpu": rec.provenance.gpu.get("name") or "-",
                "build": build_label(rec),
            }
        )
    rows.sort(key=lambda r: r["platform"])
    return rows


def _perf_hardware_rows(
    runs: list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]],
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """One row per input length; one column per platform with cold TTFT / pp / tg."""
    labels = sorted({platform_label(rec) for rec, _, _ in runs})
    cols = [("length", "length")] + [(lab, lab) for lab in labels]
    by_len: dict[int, dict[str, str]] = {}
    for rec, _, results in runs:
        lab = platform_label(rec)
        for r in results.values():
            if r.get("kind") != "cold" or not r.get("fits"):
                continue
            cell = (
                f"ttft {_fmt(r.get('ttft_s'))}s / pp {_fmt(r.get('prompt_tps'))} / "
                f"tg {_fmt(r.get('decode_tps'))} t/s"
            )
            by_len.setdefault(int(r["length"]), {})[lab] = cell
    rows = [
        {"length": length, **{lab: cells.get(lab, "-") for lab in labels}}
        for length, cells in sorted(by_len.items())
    ]
    return cols, rows


def hardware_blocks(run_dirs: list[Path]) -> tuple[list[Table], str]:
    """Same model + task on more than one platform/backend: the Spark-vs-Halo tables."""
    groups: dict[tuple[str, str], list[tuple[RunRecord, Path, dict[str, dict[str, Any]]]]] = {}
    for d in run_dirs:
        rec = load_run(d)
        if rec.server.get("provider"):
            continue
        results = {r.get("id") or f"{r.get('kind')}:{r.get('length')}": r for r in load_results(d)}
        groups.setdefault((rec.model, rec.task), []).append((rec, d, results))
    tables: list[Table] = []
    md: list[str] = []
    for (model, task), runs in sorted(groups.items()):
        if len({platform_label(rec) for rec, _, _ in runs}) < 2:
            continue
        title = f"{model} {task}"
        if task == "perf":
            cols, rows = _perf_hardware_rows(runs)
        else:
            cols, rows = HARDWARE_COLUMNS, _accuracy_hardware_rows(runs)
        if not rows:
            continue
        if not md:
            md.append("\n## Hardware: same model, different box\n")
            md.append(
                "Rows are (platform/backend); `paired` scores every run on the items all of them "
                "answered. Latency includes each box's own prefill, so the "
                "gap is the hardware gap.\n"
            )
        tables.append(_rich_table(f"hardware: {title}", cols, rows))
        md.append(_render_md(title, cols, rows))
    return tables, "\n".join(md)


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
    cmp_tables, cmp_md = comparison_blocks(run_dirs)
    for t in cmp_tables:
        console.print(t)
    if cmp_md:
        md.append(cmp_md)
    hw_tables, hw_md = hardware_blocks(run_dirs)
    for t in hw_tables:
        console.print(t)
    if hw_md:
        md.append(hw_md)
    return table, "\n".join(md) + "\n"


def write_report(settings: Settings, suite: str, markdown: str) -> Path:
    root = evals_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"report-{suite}.md"
    path.write_text(markdown)
    return path
