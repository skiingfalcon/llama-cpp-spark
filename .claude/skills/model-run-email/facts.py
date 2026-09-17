"""Print the numbers a model-run email needs, straight from committed run directories.

Usage (from the repo root):
    uv run python .claude/skills/model-run-email/facts.py <run-dir> [<run-dir> ...]
        [--fits-exclude GS,STWD] [--cold-min-tokens 80000] [--baseline <run-dir>]

One block per run: headline, "when the filing fits" count, bootstrap 95% CI (the harness's own
`bootstrap_ci`), decode/prefill, cold TTFT range on large filings, wall clock, truncations,
reasoning tokens, by-mode counts, every miss. With --baseline, also the item-level diff.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime
from pathlib import Path

from spark_llm.evals.report import bootstrap_ci


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    run = json.loads((run_dir / "run.json").read_text())
    rows = [
        json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line
    ]
    return run, rows


def minutes(run: dict) -> float | None:
    if not run.get("finished"):
        return None
    a = datetime.fromisoformat(run["started"])
    b = datetime.fromisoformat(run["finished"])
    return (b - a).total_seconds() / 60


def describe(run_dir: Path, fits_exclude: set[str], cold_min: int) -> None:
    run, rows = load(run_dir)
    s = run.get("summary") or {}
    prov = run.get("provenance") or {}
    server = run.get("server") or {}
    tc = run.get("task_config") or {}
    ok = [bool(r.get("correct")) for r in rows]
    fits = [bool(r.get("correct")) for r in rows if r.get("ticker") not in fits_exclude]
    ci = bootstrap_ci(ok)
    cold = sorted(
        r["ttft_s"]
        for r in rows
        if not r.get("warm") and (r.get("prompt_tokens") or 0) >= cold_min and r.get("ttft_s")
    )
    cold_tps = [
        r["prompt_tps"]
        for r in rows
        if not r.get("warm") and (r.get("prompt_tokens") or 0) >= cold_min and r.get("prompt_tps")
    ]
    modes: dict[str, list[int]] = {}
    for r in rows:
        m = modes.setdefault(str(r.get("mode")), [0, 0])
        m[0] += 1
        m[1] += int(bool(r.get("correct")))
    trunc = [r for r in rows if r.get("truncated")]
    non_trunc = [bool(r.get("correct")) for r in rows if not r.get("truncated")]
    comp = [r.get("completion_tokens") or 0 for r in rows]
    reasoning = [r.get("reasoning_tokens") or 0 for r in rows]
    wall = minutes(run)
    build = prov.get("llama_cpp_checkout") or prov.get("llama_cpp_release")
    cold_txt = f"{int(cold[0])}–{int(cold[-1])} s (n={len(cold)})" if cold else "n/a"

    print(f"== {run_dir}")
    print(
        f"model={run.get('model')} task={run.get('task')} platform={prov.get('platform')}/"
        f"{prov.get('backend')} build={build}"
        f" ({server.get('build_info')})"
        f" n_ctx_per_slot={server.get('n_ctx_per_slot')}"
        f" max_input_tokens={tc.get('max_input_tokens')} scoring={tc.get('scoring_version')}"
        f" quality={run.get('quality')}"
    )
    print(
        f"headline {sum(ok)}/{len(ok)} = {sum(ok) / len(ok):.1%}"
        + (f"  CI {ci[0]:.2f}–{ci[1]:.2f}" if ci else "")
        + f"  | fits ({', '.join(sorted(fits_exclude))} excluded) {sum(fits)}/{len(fits)}"
        + f"  | non-truncated {sum(non_trunc)}/{len(non_trunc)}"
    )
    print(
        f"decode p50 {s.get('decode_tps_p50') or 0:.1f} t/s | cold prefill p50 "
        f"{int(st.median(cold_tps)) if cold_tps else 'n/a'} t/s | cold TTFT on >= {cold_min} tok "
        f"{cold_txt} | "
        f"ttft p50 {s.get('ttft_p50_s') or 0:.1f} s | "
        f"total p50/p95 {s.get('total_p50_s') or 0:.0f}/"
        f"{s.get('total_p95_s') or 0:.0f} s | wall {f'{wall:.0f} min' if wall else 'n/a'}"
    )
    print(
        f"truncated {len(trunc)} | reasoning tokens {sum(reasoning)}"
        f" (p50 {int(st.median(reasoning))}) "
        f"| completion p50 {int(st.median(comp))} | cached prompt {s.get('cached_prompt_tokens')} "
        f"| fallback {s.get('fallback')} | by mode {modes}"
    )
    big = sorted(((r.get("doc_tokens") or 0, r.get("ticker")) for r in rows), reverse=True)
    seen: set[str] = set()
    tops = []
    for tok, t in big:
        if t not in seen:
            seen.add(t)
            tops.append(f"{t} {tok}")
        if len(tops) == 3:
            break
    print("largest filings (doc_tokens):", ", ".join(tops))
    print("misses:")
    for r in rows:
        if not r.get("correct"):
            print(
                f"  {r.get('id'):55} mode={r.get('mode')} answer={str(r.get('answer'))[:18]!r} "
                f"expected={r.get('expected')} truncated={bool(r.get('truncated'))}"
            )
    print()


def diff(a_dir: Path, b_dir: Path) -> None:
    _, a = load(a_dir)
    _, b = load(b_dir)
    bi = {r["id"]: r for r in b}
    print(f"== item-level diff: {a_dir.name} (A) vs {b_dir.name} (B)")
    for r in a:
        o = bi.get(r["id"])
        if o is None or bool(r.get("correct")) == bool(o.get("correct")):
            continue
        print(
            f"  {r['id']:55} A={'ok' if r.get('correct') else 'MISS'} ({r.get('mode')})  "
            f"B={'ok' if o.get('correct') else 'MISS'} ({o.get('mode')})"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument(
        "--fits-exclude", default="GS,STWD", help="tickers that do not fit a 131K window"
    )
    p.add_argument("--cold-min-tokens", type=int, default=80000)
    p.add_argument("--baseline", type=Path, help="run dir to diff each run against, item by item")
    args = p.parse_args()
    excl = {t.strip() for t in args.fits_exclude.split(",") if t.strip()}
    for run_dir in args.runs:
        describe(run_dir, excl, args.cold_min_tokens)
        if args.baseline:
            diff(run_dir, args.baseline)


if __name__ == "__main__":
    main()
