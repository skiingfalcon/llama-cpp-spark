# Email draft: DeepSeek-V4-Flash-0731 (nothink) on the Spark (2026-09-19)

> Internal team email. Every number traces to a run directory under `state/evals/` or to
> `docs/models/deepseek-v4-flash.md`. Drafted with the `model-run-email` skill.

---

Good morning,

One more overnight run on the Spark, this time DeepSeek's V4 Flash. It is new to us, so a short
introduction first.

**What DeepSeek-V4-Flash-0731 is.** DeepSeek-AI released it on July 31, 2026 as open weights
under the MIT licence, superseding an April 2026 preview. It is a 284B-parameter
mixture-of-experts model that activates about 13B parameters per token, six of 256 routed
experts plus one shared expert. Its attention is compressed and sparse: a lightning indexer
picks each token's top 512 compressed key/value entries instead of attending to the full cache,
and YaRN scaling stretches that to a native 1M-token context. It is a reasoning model that can
think in hidden tokens, but we ran the thinking-off twin as the primary row. We tried it for one
reason: at 2 bits per weight (unsloth's UD-Q2_K_XL dynamic quant, 96.8 GB) it is the only DeepSeek
V4 checkpoint that fits the Spark's 128 GB at all, and it is worth knowing whether 2-bit quality
holds on numbers before that question is closed for good.

Same task as before: 11 financial metrics from the latest 10-K of 12 companies, 121 questions,
scored against SEC XBRL ground truth at 0.5% tolerance. Single run per row; read gaps inside
overlapping 95% intervals as noise.

**What this says**

- **Accuracy matches gpt-oss-120b exactly, 119 of 121, and the two-bit quant is not the
  problem.** Same score, same 0.96–1.00 interval, zero truncations either side. The two models
  do not miss the same questions: DeepSeek got Caterpillar's long-term debt and Goldman's
  operating cash flow right where gpt-oss-120b missed both, but gpt-oss-120b got both Palantir
  questions right where DeepSeek missed them. Net wash on this seed.
- **DeepSeek's only two misses are the same unit error, back to back.** Palantir total
  liabilities and Palantir shares outstanding both came back exactly 1,000x too small, `1412381`
  instead of `1,412,381,000` and `2391192` instead of `2,391,192,000`. Everything else on that
  filing was read correctly, so this looks like a scale-marker misread on Palantir's XBRL
  specifically, not a general numeric-extraction weakness.
- **The long-filing fallback held up better than gpt-oss-120b's.** All 17 of the section-mode
  questions (Goldman and Starwood, whose filings exceed the 131K window) came back correct,
  17 of 17, against 16 of 17 for gpt-oss-120b. The 1M native window was not used here; this is
  the ordinary 131K-budget fallback path doing its job.
- **It is not fast, and the active-parameter count explains most of it.** Decode ran at 16 t/s,
  about half gpt-oss-120b's 30 t/s, and cold time-to-first-token on the seven filings at 80K+
  tokens ran 190–306 s against 57–101 s for gpt-oss-120b, roughly 3x slower. DeepSeek activates
  13B parameters per token against gpt-oss-120b's 5.1B; even at 2 bits per weight against
  gpt-oss-120b's 4.25-bit MXFP4, that is more bytes moved per token, not fewer, and unsloth's
  dynamic quant keeps attention and the shared expert at higher precision still, which adds more.
  The lightning indexer's top-512 KV lookup is extra compute on top of that. The whole run took
  51 minutes against 26 for gpt-oss-120b.
- **The prompt-cache worry from the model card did not materialise.** 88% of prompt tokens were
  served from cache (9.3M of 10.6M), in the same range as gpt-oss-120b's 90%. Upstream PR #29008
  flagged a delimiter gap that could have broken cache reuse on this architecture; on our pin it
  did not, so that open question is closed.

**Recommendation** for document extraction is unchanged: gpt-oss-120b stays the on-prem default.
DeepSeek nothink proves the 2-bit quant is fine for numeric extraction, which was the real
question, but it buys no speed and costs roughly 2x the wall clock, with no room left in 128 GB
to co-serve a judge model for FinanceBench. Two runs would tell us more: the same suite at 262K
context, to see whether Goldman and Starwood move into full-context mode and close the last gap;
and the thinking twin at `reasoning_effort=low`, to see whether a small reasoning budget catches
the Palantir scale error without spending the whole completion budget the way Nemotron's thinking
did. If low-effort thinking fixes both Palantir misses without truncating, DeepSeek reaches 121
of 121 and is worth a second look purely on accuracy; if it truncates, we stop there.

| Stack | Accuracy, 121 Qs | When the filing fits, of 104 Qs | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 262K ctx | 98.3% | 102/104 | 55 t/s | 40–70 s | 24 min | $0 marginal |
| **DGX Spark, DeepSeek-V4-Flash-nothink (MoE, 13B active, 2-bit)** | **98.3%** | **102/104** | **16 t/s** | **190–306 s** | **51 min** | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 262K ctx | 96.7% | 100/104 | 93 t/s | 37–64 s | 20 min | $0 marginal |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 131K ctx | 94.2% | 102/104 | 60 t/s | 38–54 s | 23 min | $0 marginal |
| DGX Spark, Bonsai 2 27B (ternary), 131K ctx | 93.4% | 102/104 | 16 t/s | 120–200 s | 107 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, 131K budget | 93.4% | 99/104 | 20 t/s | 105–160 s | 148 min | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 131K ctx | 92.6% | 101/104 | 98 t/s | 37–53 s | 21 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, full 1M window | 92.6% | 97/104 | 19 t/s | 105–160 s; GS whole 6.5 min | 153 min | $0 marginal |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | 45 t/s | 40–60 s | 19 min | $0 marginal |

"When the filing fits" counts the 104 questions on the ten filings that fit a 131K window, so the
column is comparable across tokenizers. DeepSeek's cold-prefill range comes from the same
"time to first token on an 80K+ token filing" measure used for the other rows; we did not run a
filing close to exactly 100K tokens this time. This table is the canonical, cumulative version
kept at `docs/eval-comparison-table.md`; it now includes the Bonsai rows added on 2026-09-18,
which the first draft of this email missed.

Two housekeeping notes: FinanceBench is not planned for this checkpoint, since the 97 GB of
weights leaves no headroom to co-serve the gpt-oss-120b judge. The machine-generated report still
flags extract-full as runs on different configs; this run adds one more to that list, which is
expected whenever a new model or context size joins the suite. The memorisation control, same
questions with no document, and the thinking-on twin are still outstanding before any of this
goes outside the company.

**Write-ups**

- DeepSeek-V4-Flash model card, with the configuration used and the open questions: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/models/deepseek-v4-flash.md
- Bonsai 2 27B on the RTX 5090, with the ternary format explained: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-rtx5090-bonsai2.md
- Nemotron-3-Super on the Spark, for the reasoning-budget comparison: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-spark-cuda-nemotron.md
- Which report covers what: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/index.md
- Canonical cross-stack comparison table, updated with every run: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-comparison-table.md
- Machine-generated table over all runs, with intervals: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/state/evals/report-sec.md

---

## Fact trail

| Claim | Source |
| --- | --- |
| Released July 31, 2026, MIT licence, 284B total / ~13B active MoE, 256 routed + 1 shared experts, 6 active/token | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 |
| Lightning indexer (top-512 compressed KV), YaRN to 1M native context, reasoning via hidden thinking | `docs/models/deepseek-v4-flash.md` Identity |
| Checkpoint served: unsloth UD-Q2_K_XL, 96.8 GB, 3 shards | `docs/models/deepseek-v4-flash.md` Identity |
| 119/121 = 98.3%, CI 0.96–1.00, 102/104 fits, 0 truncated, 0 reasoning tokens, decode 16.0 t/s, cold prefill p50 422 t/s, cold TTFT 190–306 s (n=7), wall 51 min | `state/evals/sec/deepseek-v4-flash-nothink-spark-cuda/20260919T121223Z-extract-full/run.json` (+ `.claude/skills/model-run-email/facts.py`) |
| Misses: PLTR Liabilities and CommonStockSharesOutstanding, both answered 1,000x too small, not truncated | same run's `results.jsonl` |
| Section-mode (fallback) 17/17 correct | same `facts.py` output, `by mode` field |
| gpt-oss-120b baseline: 119/121, CI 0.96–1.00, 103/104 fits, decode 30.3 t/s, cold prefill p50 1354 t/s, cold TTFT 57–101 s, wall 26 min, 20,720 reasoning tokens, misses GS operating cash flow and CAT long-term debt | `state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full/run.json` |
| Item-level diff (PLTR flips one way, GS/CAT flip the other) | `facts.py --baseline` output comparing the two run dirs |
| Cache reuse 87.9% (9,302,499 / 10,577,083 cached prompt tokens) vs gpt-oss-120b 89.8% (9,466,001 / 10,542,592) | both `run.json` files |
| PR #29008 cache-reuse risk noted as an open question | `docs/models/deepseek-v4-flash.md` Weaknesses |
| gpt-oss-120b: 5.1B active parameters, MXFP4 (4.25 bits/weight) | https://arxiv.org/pdf/2508.10925 (gpt-oss model card) |
| 97 GB footprint rules out co-serving the FinanceBench judge | `docs/models/deepseek-v4-flash.md` Serving configuration |
| Nemotron thinking-budget precedent (six blanks from unbounded reasoning) | `docs/emails/2026-09-17-nemotron-3-super.md` |
| Prior comparison table rows, including Bonsai (both platforms) added 2026-09-18 | `docs/eval-comparison-table.md` (canonical), `docs/eval-report-2026-09-rtx5090-bonsai2.md` |
