"""Turn external harness outputs into one results.jsonl schema + summary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from spark_llm.evals.endpoint import percentile

# -- evalplus (tier 1) -------------------------------------------------------------------------

_PASS_AT_1 = re.compile(r"^(?P<ds>\w+)\s*\((?P<which>base tests|base \+ extra tests)\)\s*$", re.M)
_SCORE = re.compile(r"pass@1:\s*(?P<v>[0-9.]+)")


def parse_evalplus_stdout(stdout: str) -> dict[str, float]:
    """{'humaneval_base': 0.85, 'humaneval_plus': 0.80} from evalplus.evaluate output."""
    out: dict[str, float] = {}
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        m = _PASS_AT_1.match(line.strip())
        if not m:
            continue
        for nxt in lines[i + 1 : i + 3]:
            s = _SCORE.search(nxt)
            if s:
                suffix = "base" if m.group("which") == "base tests" else "plus"
                out[f"{m.group('ds').lower()}_{suffix}"] = float(s.group("v"))
                break
    return out


def normalize_evalplus(
    root: Path, dataset: str, stdout: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("eval_results.json")):
        if dataset not in str(path).lower():
            continue
        data = json.loads(path.read_text())
        for task_id, attempts in (data.get("eval") or {}).items():
            a = attempts[0] if attempts else {}
            base = a.get("base_status", a.get("status"))
            plus = a.get("plus_status")
            results.append(
                {
                    "id": task_id,
                    "dataset": dataset,
                    "base_pass": base == "pass",
                    "plus_pass": plus == "pass" if plus is not None else None,
                    "correct": (plus == "pass") if plus is not None else (base == "pass"),
                }
            )
    scores = parse_evalplus_stdout(stdout)
    n = len(results)
    summary = {
        "n": n,
        "score": (sum(int(bool(r["correct"])) for r in results) / n)
        if n
        else scores.get(f"{dataset}_plus", scores.get(f"{dataset}_base")),
        "base_pass_at_1": scores.get(f"{dataset}_base"),
        "plus_pass_at_1": scores.get(f"{dataset}_plus"),
        "skipped": 0,
    }
    return results, summary


# -- aider polyglot (tier 2) -------------------------------------------------------------------


def normalize_aider(bench_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(bench_dir.rglob(".aider.results.json")):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        outcomes = list(d.get("tests_outcomes") or [])
        rel = path.parent.relative_to(bench_dir)
        parts = rel.parts
        lang = parts[0] if parts else "?"
        results.append(
            {
                "id": f"{lang}/{path.parent.name}",
                "language": lang,
                "passed_first_try": bool(outcomes[0]) if outcomes else False,
                "correct": any(outcomes),
                "attempts": len(outcomes),
                "malformed_responses": int(d.get("num_malformed_responses") or 0),
                "syntax_errors": int(d.get("syntax_errors") or 0),
                "indentation_errors": int(d.get("indentation_errors") or 0),
                "lazy_comments": int(d.get("lazy_comments") or 0),
                "duration_s": d.get("duration"),
                "prompt_tokens": d.get("prompt_tokens"),
                "completion_tokens": d.get("completion_tokens"),
                "timeouts": int(d.get("test_timeouts") or 0),
            }
        )
    n = len(results)
    by_lang: dict[str, dict[str, int]] = {}
    for r in results:
        b = by_lang.setdefault(r["language"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
    summary = {
        "n": n,
        "score": (sum(int(r["correct"]) for r in results) / n) if n else None,
        "pass_rate_first_try": (sum(int(r["passed_first_try"]) for r in results) / n)
        if n
        else None,
        "format_failure_rate": (sum(int(r["malformed_responses"] > 0) for r in results) / n)
        if n
        else None,
        "syntax_error_rate": (sum(int(r["syntax_errors"] > 0) for r in results) / n) if n else None,
        "by_language": by_lang,
        "total_tokens": sum(
            (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in results
        ),
        "skipped": 0,
    }
    return results, summary


# -- SWE-bench via mini-swe-agent (tier 3) ------------------------------------------------------

_FORMAT_STATUSES = ("format", "parse", "malformed")


def normalize_swebench(
    preds: dict[str, Any],
    report: dict[str, Any] | None,
    trajectories: dict[str, dict[str, Any]],
    expected_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = set((report or {}).get("resolved_ids") or [])
    errored = set((report or {}).get("error_ids") or [])
    results: list[dict[str, Any]] = []
    for iid in expected_ids:
        pred = preds.get(iid) or {}
        traj = trajectories.get(iid) or {}
        info = traj.get("info") or {}
        exit_status = str(
            info.get("exit_status") or ("Submitted" if pred.get("model_patch") else "missing")
        )
        messages = traj.get("messages") or []
        stats = info.get("model_stats") or {}
        results.append(
            {
                "id": iid,
                "submitted": bool(pred.get("model_patch")),
                "correct": iid in resolved if report is not None else None,
                "harness_error": iid in errored,
                "exit_status": exit_status,
                "format_failure": any(k in exit_status.lower() for k in _FORMAT_STATUSES),
                "steps": max(len(messages) // 2 - 1, 0) if messages else None,
                "prompt_tokens": stats.get("prompt_tokens"),
                "completion_tokens": stats.get("completion_tokens"),
                "api_calls": stats.get("api_calls"),
            }
        )
    n = len(results)
    steps = [r["steps"] for r in results if r["steps"] is not None]
    summary = {
        "n": n,
        "submitted": sum(int(r["submitted"]) for r in results),
        "resolved": len(resolved),
        "score": (len(resolved) / n) if (n and report is not None) else None,
        "format_failure_rate": (sum(int(r["format_failure"]) for r in results) / n) if n else None,
        "not_submitted_rate": (sum(int(not r["submitted"]) for r in results) / n) if n else None,
        "exit_statuses": _counts(r["exit_status"] for r in results),
        "steps_p50": percentile(steps, 0.5) if steps else None,
        "total_tokens": sum(
            (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in results
        ),
        "skipped": 0 if report is not None else n,
    }
    return results, summary


def _counts(values) -> dict[str, int]:  # type: ignore[no-untyped-def]
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
