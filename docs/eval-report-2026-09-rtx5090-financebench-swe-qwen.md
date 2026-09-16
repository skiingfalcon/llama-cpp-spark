# FinanceBench and coding tier 1 — Qwen3.8-27B on an RTX 5090 (September 2026)

First runs of two suites the harness had configured but never executed: the SEC
`qa-financebench` task (150 analyst-written questions over human-selected filing excerpts) and
SWE tier 1 (evalplus HumanEval and MBPP, greedy, pass@1). Model, weights and hardware are the
ones in the [RTX 5090 extraction report](eval-report-2026-09-rtx5090-cuda-qwen.md):
`qwen3.8-27b` (Unsloth `UD-Q4_K_XL`), one RTX 5090 (32 GB), Windows 11, CUDA llama.cpp
`b10919`, `--ctx-size 131072`. The harness ran on Windows for FinanceBench and inside WSL2
Ubuntu for tier 1 (evalplus needs `signal.alarm` and `resource`, both Unix-only).

| Suite | Config | Result | Wall clock |
| --- | --- | ---: | ---: |
| **FinanceBench, 150 Qs** | reasoning on | 60/74 numeric (81%) under unit-tolerant scoring; 76 free-text unscored (no judge) | 21 min |
| **Tier 1 HumanEval / HumanEval+** | reasoning off | **93.3% / 90.9%** | ~26 min for both datasets |
| **Tier 1 MBPP / MBPP+** | reasoning off | **89.2% / 76.5%** | (same run) |
| Tier 1 HumanEval / HumanEval+ | reasoning on, 4096-token budget | 86.6% / 86.0% | 98 min for both datasets |
| Tier 1 MBPP / MBPP+ | reasoning on, 4096-token budget | 85.7% / 73.8% | (same run) |

Single run per row. These are shakedown runs as much as model results: each suite surfaced a
harness defect, listed under [What the harness needs](#what-the-harness-needs), and the fixes
are in the same branch as the run folders.

## Setup

- **Server:** the same `qwen3.8-27b` registry entry as the extraction runs (`--jinja`,
  `--flash-attn on`, batch/ubatch 2048, 131,072 context, f16 KV, 4 auto slots with unified KV).
  "Reasoning off" rows were served with `--reasoning off` added to llama-server's arguments
  (registry entry `qwen3.8-27b-code`, same port); nothing else differs. `run.json` records the
  registry entry named on the command line, so it does not show that flag: the two tier-1
  folders are distinguished by `tools.evalplus_max_new_tokens` (present only on the reasoning-on
  run) and by this report.
- **FinanceBench:** harness defaults (`temperature=0`, `seed=42`, `max_tokens=4096`), evidence
  passages as context, `financebench_merged.jsonl` from `PatronusAI/financebench` (the file the
  config pointed at no longer exists; the merged file carries the same 150 `OPEN_SOURCE` rows).
  No judge model: `gpt-oss-120b` (80 GB) does not fit on this card next to Qwen, so free-text
  answers are saved but unscored.
- **Tier 1:** `evalplus==0.3.1` via `uvx`, `--greedy`, `--backend openai` against the served
  endpoint. Reasoning-off run used evalplus's default 768-token completion budget; reasoning-on
  run used the new `swe.tier1.max_new_tokens = 4096` (see below). Tests executed inside WSL.
- **Artifacts** (all under `qwen3.8-27b-halo-vulkan` for the reason given in the extraction
  report; `server.build_info` is authoritative):
  - FinanceBench: [`state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T171214Z-qa-financebench/`](../state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T171214Z-qa-financebench/)
  - Tier 1, reasoning off: [`state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T183613Z-tier1/`](../state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T183613Z-tier1/)
    (`run.json` timestamps span 4 s because this record was regenerated from the saved
    generations after the parser fix; the first pass, whose record had `n=0`, took about 26 min
    of wall clock, observed but not recorded)
  - Tier 1, reasoning on: [`state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T205853Z-tier1/`](../state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T205853Z-tier1/)
  - Each tier-1 folder has an `evalplus/` subfolder with evalplus's own per-problem result files
    (solution text, base and plus status) and `empty_solutions.json`, the task ids whose returned
    content was empty. `results.jsonl` alone does not carry solutions.

## FinanceBench

| Slice | n | Harness score | Unit-tolerant, any-number score |
| --- | ---: | ---: | ---: |
| Numeric answers (scored by number match) | 74 | 7 (9.5%) | **60 (81.1%)** |
| of which `metrics-generated` (pure calculation) | 50 | 0 | **46 (92%)** |
| of which `domain-relevant` / `novel-generated` bucketed as numeric | 24 | 7 | 14 |
| Free-text answers (need the judge) | 76 | unscored | unscored |
| Abstained (answer contains `unknown` or an explicit "does not state/contain/provide" phrase) | 20 of 150 | | |

**The 9.5% is a scorer artifact.** The FinanceBench prompt asks for "one or two sentences", and
the harness grades those sentences with the 10-K extractor: first number in the reply, 1%
tolerance, exact units. Qwen writes its working first ("1,469,502 / 2,213,556 = 0.66") and
states units ("$1,577 million" against a reference of `$1577.00` in millions), so the scorer
grabs an input or the wrong scale and fails a correct answer. Rescoring the saved answers with
"any number in the reply within 1% of the reference, scale words forgiven" gives 60/74, and the
50 pure-calculation questions come out 46/50.

The 14 remaining numeric misses split roughly in half: seven are free-text questions that
happen to contain one number in the reference (e.g. "No. Verizon's debt decreased…" answered
"No. Total debt decreased from $150,868 to $150,639") and belong with the judge; about six are
genuine errors (an EBITDA margin computed from the wrong revenue line, dividends total instead
of per share, two ROA denominators). One is a sign-convention quibble (organic sales of
"(0.9)%" vs "shrunk by 0.9%").

Latency: p50 4.7 s per question, 21 minutes for 150, decode 75 t/s on these short prompts.

## Coding tier 1

| Dataset | Reasoning off, 768 budget | Reasoning on, 4096 budget |
| --- | ---: | ---: |
| HumanEval base pass@1 | **93.3%** (153/164) | 86.6% (142/164) |
| HumanEval+ pass@1 | **90.9%** (149/164) | 86.0% (141/164) |
| MBPP base pass@1 | **89.2%** (337/378) | 85.7% (324/378) |
| MBPP+ pass@1 | **76.5%** (289/378) | 73.8% (279/378) |
| Empty solutions | 0 / 542 | **68 / 542** (20 HumanEval, 48 MBPP) |
| Wall clock | ~26 min (observed, see Artifacts) | 98 min |

Pass counts follow evalplus's printed pass@1, which counts a problem as passing plus only if it
also passes base. Two MBPP problems (Mbpp/635, Mbpp/787) pass the plus tests but fail base, so
the harness's per-problem `correct` field (plus status alone) gives 291/378 for the reasoning-off
run in `results.jsonl`; the table uses 289.

**Thinking hurt, and every lost point is a blank.** With reasoning on, 68 of 542 problems came
back with empty content (`evalplus/empty_solutions.json`). evalplus does not store
`finish_reason`, so the mechanism is inferred: it is consistent with the 4,096-token budget being
consumed by hidden reasoning, and was reproduced by hand on HumanEval/10 (`make_palindrome`),
which returned `finish_reason=length` after 14,800 characters of reasoning and no code.

On the 474 problems both runs answered, thinking **helped**: HumanEval base 142/144 vs 139/144,
HumanEval+ 141 vs 137, MBPP base 324/330 vs 306/330, MBPP+ 279 vs 270, a 2–5 point edge for
reasoning on. The blanks more than erased it: the reasoning-off run solved 45 of the 68 problems
reasoning-on left empty (14 of 20 HumanEval, 31 of 48 MBPP). Reasoning off had no truncations
under evalplus's 768-token default.

This is the same failure the extraction suite showed on a smaller scale (one HD share-count
question spent all 4,096 tokens reasoning) and the opposite of the Spark report's hope that a
model which "reads carefully and thinks long" would pull ahead on harder tasks. On function-level
coding it is a strong direct answerer whose deliberation, when it finishes, is a little better
still, but which cannot be relied on to finish inside 4,096 tokens.

## What this says

- **Qwen3.8-27B without thinking is a good local coder:** HumanEval+ 90.9% and MBPP+ 76.5% in
  under half an hour on a consumer card, no truncations.
- **Turn thinking off for short-answer tasks** unless the budget is raised well past 4,096 or a
  thinking budget is imposed server-side (`--reasoning-budget`). Thinking is worth 2–5 points
  where it completes; at 4,096 it fails to complete one problem in eight.
- **FinanceBench separates nothing yet.** Until the numeric scorer handles sentence answers and
  the judge runs, its headline number is unusable; the pure-calculation half suggests ~90%.
- Nothing here is comparable to gpt-oss-120b yet; neither suite has a Spark run. The Spark rows
  will need the same harness fixes.

## What the harness needs

Found while running these suites; fixes for 1–3 are committed alongside the run folders.

1. **`swe/normalize.py`:** evalplus 0.3.1 writes `<model>_<backend>_temp_<t>_eval_results.json`,
   so the literal `rglob("eval_results.json")` never matched: every tier-1 record had `n=0` and
   no per-problem rows. The stdout regex also missed `humaneval+ (base + extra tests)` because of
   the `+`, so plus pass@1 was always `None`. Fixed; verified on this run's files.
2. **`evals.toml` FinanceBench path:** `financebench_open_source.jsonl` is gone from the HF
   dataset; `financebench_merged.jsonl` holds the same 150 rows. Fixed.
3. **`swe.tier1.max_new_tokens`:** evalplus hard-codes a 768-token completion budget and offers
   no CLI flag (0.3.1 is the latest release). New optional setting; when set, the pinned
   `evalplus.evaluate` entry point runs in-process with that one constructor default overridden,
   and the value is recorded in `run.json` tools. Default 4096 in `evals.toml`.
4. **FinanceBench numeric scoring** takes the first number in a sentence answer and demands exact
   units. It should either match any number in the reply (unit-tolerant) or send these items to
   the judge, whose prompt already forgives units. Not changed here; the lenient rescoring script
   used for the table above is not part of the harness.
5. **Reruns silently reuse generations.** Tier 1's work dir is fixed per model, and evalplus
   resumes from any samples it finds there, so a second run against a differently served model
   inherits the first run's answers unless the directory is moved aside. A per-run work dir (or a
   `--fresh` flag) would prevent that.
6. **Serving flags are not in provenance.** `run.json` records the registry entry, not the argv
   llama-server was started with; `--reasoning off` vs on is invisible. Recording
   `state/servers.json`'s argv for the model at run start would fix it.
7. **Windows:** `eval sec fetch` needs `PYTHONUTF8=1` (cp1252 write of a ☒ character left a 0-byte
   AAPL text file the retry skipped); `serve <model> -- <flags>` never reaches llama-server (the
   variadic names argument swallows them); tier 1 needs WSL or Linux.

## Caveats

> **Review notes (2026-09-15).** Single run per row. The FinanceBench numbers are numeric-only
> and rescored outside the harness (the lenient rule moves by ±2 depending on how a zero
> reference and percent scaling are treated); the 76 free-text answers await a judge pass (they
> are in `results.jsonl`). The two tier-1 rows differ in two variables (reasoning on/off and 768 vs
> 4096 budget), chosen because reasoning on cannot run at 768; the blank-answer analysis, not the
> headline delta, is the finding. `finish_reason` is not stored by evalplus, so "budget spent in
> reasoning" rests on empty content plus one by-hand reproduction. The reasoning-on run was interrupted twice (a WSL loopback
> failure and a session restart) and resumed from saved generations both times; generations are
> greedy, so resumption does not change the answers. No `--no-document` control for FinanceBench.

## Reproduce

```bash
# Windows, PowerShell, in the repo: FinanceBench (numeric-only without a judge)
$env:PYTHONUTF8 = "1"
uv run local-llm serve qwen3.8-27b
uv run local-llm eval sec run qwen3.8-27b --task qa-financebench
uv run local-llm stop

# WSL2 Ubuntu (mirrored networking so 127.0.0.1 reaches the Windows server), same branch:
LOCAL_LLM_PLATFORM=halo uv run local-llm eval swe check qwen3.8-27b
LOCAL_LLM_PLATFORM=halo uv run local-llm eval swe run qwen3.8-27b --tier 1     # 4096 budget from evals.toml
# reasoning off (the 768-budget row): serve the qwen3.8-27b-code entry (adds --reasoning off),
# comment out max_new_tokens in evals.toml, and move state/evals/swe/qwen3.8-27b/work-tier1
# aside first, or evalplus resumes from the previous run's generations.
```
