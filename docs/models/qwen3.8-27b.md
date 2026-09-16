# qwen3.8-27b

**Status:** measured (Spark CUDA at 131K; RTX 5090 CUDA at 131K and 262K; FinanceBench and coding tier 1 on the RTX 5090). **Careful reader; ties gpt-oss-120b at its native 262K on a 32 GB card in 24 min. Slow only on the Spark. Hidden thinking hurts it on short coding tasks.**

## Identity

| | |
| --- | --- |
| Family / vendor | Alibaba Qwen 3.8 |
| Architecture | Dense 27B, vision-language checkpoint served text-only; llama.cpp arch `qwen35`; reasoning via hidden thinking |
| Checkpoint served | `unsloth/Qwen3.8-27B-GGUF`, quant `UD-Q4_K_XL` (17.6 GB) |
| Native context | 262,144 (served at 131,072 to match the gpt-oss runs) |
| License | Apache-2.0 (verify on the model card) |
| Registry entry | `models.toml` `[models."qwen3.8-27b"]`, port 8084; same entry in `models.halo.toml` for the Windows/RTX 5090 box |

## Serving configuration used

| Knob | Spark | RTX 5090, run 1 | RTX 5090, run 2 |
| --- | --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f`, CUDA `121a-real` | release `b10919` (`d3146f2b5`), Windows CUDA 13.3 zip | same |
| ctx_size / n_parallel | 131072 / default | 131072 / default (4 auto slots, unified KV) | 262144 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| Flash attention / KV type | on / f16 | on / f16 | on / **q8_0 K and V** |
| Extra flags | `--jinja`, `n_gpu_layers = 999` | same | same |
| VRAM in use (server) | n/a (128 GB unified) | ~26 GB of 32 | ~28 GB of 32 |

FinanceBench and tier 1 used the run-1 configuration. Tier 1 "reasoning off" adds `--reasoning off`
to llama-server (registry entry `qwen3.8-27b-code`, same port); `run.json` does not record that
flag, see the [FinanceBench / tier-1 report](../eval-report-2026-09-rtx5090-financebench-swe-qwen.md).

Eval decoding: temperature 0, seed 42, `max_tokens` 4096. Tokenizer is 10–15% fatter than
gpt-oss's on the same filings (AAPL 51.4K vs 45.6K tokens; XOM 122K vs 106K).

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| **Spark** | extract-full, 10-K | **95.9% (116/121)** | 0.92–0.99 | 93 full, 20 Item-8 section, **8 BM25 chunks** (GS) | 1 | 9.8 t/s | 70–200 s per filing (cold TTFT p50 121 s) | 125 min | `state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full` |
| **RTX 5090, 131K** | extract-full, 10-K | 94.2% (114/121) | 0.90–0.98 | 91 full, 20 Item-8 section, **8 BM25 chunks** (GS, same 3/8 and same wrong values as the Spark) | 1 | 60 t/s | 38–54 s per ~100K filing (cold TTFT p50 30 s) | 22.5 min | `state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T145801Z-extract-full` |
| **RTX 5090, 262K + q8_0 KV** | extract-full, 10-K | **98.3% (119/121)** | 0.96–1.00 | 111 full (NEE, STWD now whole), **8 Item-8 section (GS, 131,589 tokens), no chunks** | 0 | 55 t/s | 40–70 s (cold TTFT p50 45 s; STWD 202K whole: 132 s) | 24.3 min | `state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T152312Z-extract-full` |
| RTX 5090, 131K | qa-financebench, 150 Qs | numeric 60/74 (81%) with unit-tolerant any-number scoring; harness raw 7/74 (scorer defect); 76 free-text unscored (no judge on a 32 GB card) | 0.72–0.89 (numeric, lenient) | evidence excerpts as context | 2 | 75 t/s | n/a (short prompts) | 21 min | `state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T171214Z-qa-financebench` |
| RTX 5090, **reasoning off** | SWE tier 1 (evalplus, greedy) | HumanEval+ **90.9%** (149/164), MBPP+ **76.5%** (289/378); base 93.3% / 89.2% | 0.86–0.95 / 0.73–0.81 | evalplus default 768-token budget | 0 blank answers | 60 t/s | n/a | ~26 min | `state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T183613Z-tier1` |
| RTX 5090, reasoning on | SWE tier 1 (evalplus, greedy) | HumanEval+ 86.0% (141/164), MBPP+ 73.8% (279/378); base 86.6% / 85.7% | 0.80–0.91 / 0.69–0.78 | 4096-token budget (`swe.tier1.max_new_tokens`) | **68 blank answers** (budget spent in hidden reasoning) | 60 t/s | n/a | 98 min | `state/evals/swe/qwen3.8-27b-halo-vulkan/20260915T205853Z-tier1` |

By mode: full 93/93, section 20/20, chunked 3/8. Reasoning ~230 hidden tokens per question
(43.6k total, about 2x gpt-oss-120b). Prompt-cache reuse 85%.

RTX 5090 extraction, by mode: run 1 full 91/93, section 20/20, chunked 3/8; run 2 full 111/113,
section 8/8. Hidden reasoning per question is unchanged across machines (p50 224–244 tokens), so
the 5x wall-clock gap to the Spark is decode and prefill speed, not less thinking. Tier-1
intervals are per dataset (HumanEval+ / MBPP+). Details: [RTX 5090 extraction
report](../eval-report-2026-09-rtx5090-cuda-qwen.md), [FinanceBench / tier-1
report](../eval-report-2026-09-rtx5090-financebench-swe-qwen.md).

## Strengths (measured)

- Perfect wherever it saw the document: 113/113 across whole filings and Item 8 sections, including the CAT debt line gpt-oss-120b missed and the HD share count Terra missed.
- No scale errors on share counts.
- Small footprint (17.6 GB): can be served next to another model, or given a much larger context window.
- **At its native 262K with an 8-bit KV cache it fits a 32 GB card with ~3 GB to spare and recovers Goldman Sachs: 8/8 on the Item 8 path, 119/121 overall, level with gpt-oss-120b on the Spark.**
- **On an RTX 5090 it is fast:** 55–60 t/s decode (about 2x gpt-oss-120b on the Spark), a ~120K filing prefilled in under a minute, the whole extraction suite in 23–24 min.
- **Good direct coder with reasoning off:** HumanEval+ 90.9%, MBPP+ 76.5%, no truncations under evalplus's 768-token default.

## Weaknesses (measured)

- On the Spark, dense architecture on a bandwidth-bound GPU: 9.8 t/s decode, about a third of gpt-oss-120b, and it reasons twice as long. Net 5x wall clock (125 vs 26 min) on the same suite. This is a Spark property: the RTX 5090 runs the same checkpoint at 55–60 t/s.
- Fatter tokenizer plus the 131K window pushed NEE to the section fallback and GS all the way to keyword chunks, where it scored 3/8. All five misses are that one filing.
- One truncation at 4096 (GS share count, all budget spent on reasoning).
- Cold prefill 2.5x slower than gpt-oss-120b (610–740 vs 1,240–1,830 t/s).
- **Cover-page share count is unstable at temperature 0:** across three extraction runs it missed `CommonStockSharesOutstanding` on GS, XOM, HD and MSFT in different combinations, always as `unknown` or a blown budget, never a wrong number. It is the entire gap between 119/121 and 121/121 at 262K.
- **Hidden thinking overruns short-answer budgets:** with reasoning on, 68 of 542 tier-1 problems ended at 4,096 tokens with no code emitted, and it scored below its own reasoning-off run on both datasets. The extra thinking bought nothing on function-level coding.
- FinanceBench cannot be read yet: the harness's numeric scorer fails correct sentence answers (7/74 raw vs 60/74 lenient) and the free-text half needs the gpt-oss-120b judge, which does not fit beside Qwen on 32 GB.

## When to use / when not to

- **Use** when reading quality matters more than wall clock and the document fits: it did not miss a question it could see.
- **Use** at its native 262K context with `cache_type_k/v = "q8_0"` when GS/STWD-sized documents are common. Measured on the RTX 5090: the chunked misses disappear (GS 8/8) and NEE/STWD read whole.
- **Do not** use where throughput matters on the Spark; gpt-oss-120b does the same job there in a fifth of the time. On a 5090-class card the throughput objection goes away (24 min for the suite).
- **Serve it with reasoning off** for chat, coding and other short-answer work (`--reasoning off`); turn thinking on only when the completion budget is well above 4,096 or a `--reasoning-budget` cap is set.
- **Try** on multi-step reasoning tasks only once the FinanceBench scorer and judge are fixed; tier 1 coding says thinking is a liability at this budget, not an asset.

## Open questions

- ~~FinanceBench and the coding tier: does the longer reasoning buy anything on tasks that need it?~~ Coding tier 1: no, it costs 68 blank answers at 4,096 (RTX 5090, 2026-09-15). FinanceBench: unanswered until the numeric scorer handles sentence answers and the judge runs; the 76 free-text answers are saved in the run folder.
- ~~Accuracy at 262K context, and the KV/prefill cost of doing so.~~ 119/121 at 262K with q8_0 KV on the RTX 5090; ~2 GB more VRAM than 131K f16, decode 55 vs 60 t/s, whole-document STWD (202K) prefills cold in 132 s.
- Same runs on the Spark at 262K (KV memory is not the constraint there) and with reasoning off, for a like-for-like row.
- Why the cover-page share count comes back `unknown` with the page in context.
- The MoE sibling (Qwen3.5-122B class) would be the fair speed comparison to gpt-oss-120b.

## Changelog

- 2026-09-15 — first Spark run; report `docs/eval-report-2026-09-spark-cuda-qwen.md`; card created.
- 2026-09-15 — RTX 5090 (Windows, CUDA b10919): 131K replication 114/121 and native 262K + q8_0 KV 119/121 (GS 8/8); report `docs/eval-report-2026-09-rtx5090-cuda-qwen.md`. Serving-config columns, two result rows, strengths/weaknesses and open questions updated.
- 2026-09-15 — RTX 5090: first FinanceBench run (numeric-only, no judge) and first coding tier-1 runs, reasoning off and on; report `docs/eval-report-2026-09-rtx5090-financebench-swe-qwen.md`. Three result rows added.
