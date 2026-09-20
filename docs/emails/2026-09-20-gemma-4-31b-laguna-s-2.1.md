# Email draft: Gemma 4 31B and Laguna S 2.1 on the Spark (2026-09-20)

> Internal team email. Every number traces to a run directory under `state/evals/` or to
> `docs/eval-report-2026-09-spark-cuda-gemma-laguna.md`. Drafted with the `model-run-email` skill.

---

Good morning,

Three more runs on the Spark since the DeepSeek note: Google's Gemma 4 31B with thinking on and
off, and a first pass of Poolside's Laguna S 2.1, which is new to us. Gemma most of you know, so
the introduction is for Laguna. Kicking off next: [next run].

**What Laguna S 2.1 is.** Poolside released it on July 21, 2026 as open weights under the
OpenMDW-1.1 licence. It is a 118B-parameter mixture-of-experts model that activates about 8B
parameters per token, ten of 256 routed experts plus one shared expert, across 48 layers of which
36 use a 512-token sliding window and 12 attend globally. It was built for agentic coding rather
than general chat, and it thinks in hidden tokens only when asked: thinking is off by default,
the opposite of every model we have run so far, so the plain entry is the primary row and the
thinking twin is the experiment. Weights are 73.4 GB at unsloth's 4-bit dynamic quant, which fits
the Spark and the Halo but not the 5090. We tried it for two reasons: its active-parameter count
is the closest to gpt-oss-120b of anything we have served, and it is the first model on our list
that was built for the coding benchmarks we also care about.

Same task as before: 11 financial metrics from the latest 10-K of 12 companies, 121 questions,
scored against SEC XBRL ground truth at 0.5% tolerance. Single run per row; read gaps inside
overlapping 95% intervals as noise.

**What this says**

- **Neither model moves the default, but each wins one column.** Gemma with thinking on is the
  most accurate dense model we have run on the Spark, 117 of 121, and read every one of the 104
  questions on filings that fit the window correctly. Laguna is the fastest local run after
  gpt-oss-120b, 29 minutes against 26, at 115 of 121. gpt-oss-120b still has both at once: 119 of
  121 in 26 minutes.
- **The tokenizer, not the model, decided where the misses landed.** Both Gemma and Laguna cost
  10 to 16 percent more tokens per filing than gpt-oss. Goldman is 271K tokens under Gemma and
  275K under Laguna against 242K under gpt-oss. That pushed NextEra out of whole-document mode
  and pushed Goldman's financial statements past the budget, so Goldman was read from eight
  retrieved chunks instead. All four of Gemma's misses and four of Laguna's six are those Goldman
  chunks, mostly answered "unknown". gpt-oss keeps Goldman on the statements section and gets
  seven of eight. Gemma's card predicted its 262K vocabulary would make filings cheaper; on 10-K
  text it did the reverse.
- **Thinking bought Gemma two answers for 69 minutes.** Thinking on scored 117, off scored 115,
  recovering Microsoft's share count and Palantir's net income and changing nothing on Goldman.
  The run took 130 minutes against 61, and the difference is exactly the 32,700 reasoning tokens
  at 7.9 tokens per second. This is the first pair where thinking on beat thinking off, after
  Qwen on coding and Nemotron went the other way, but it is two questions on one seed inside
  overlapping intervals. No truncations either way: Gemma's thinking is short, a median 202
  tokens per question, against Nemotron's five times that.
- **Laguna's two non-Goldman misses are accounting judgment, not extraction.** For Exxon revenue
  it picked the sales line, 323.9B, over total revenues and other income, 332.2B. For NextEra
  equity it picked total equity including noncontrolling interests, 66.5B, over stockholders'
  equity, 54.6B. Both are the kind of line-choice a coding-tuned model has not been trained
  toward. Once, with thinking off, it reasoned aloud in the answer field for 600 tokens and never
  gave the number.
- **Speed sits where the active parameters say it should.** Laguna reads about 5 GB of weights
  per token against gpt-oss-120b's 2.7, and decodes at 17 against 30. Gemma reads 17.6 GB per
  token, all 30.7B parameters dense, and decodes at 7.9, below the 10 its card predicted and
  below Qwen's 9.8. The sliding-window re-prefill we feared for both models did not appear: warm
  requests re-read a median 86 tokens, cache reuse 87 percent, the same as gpt-oss.

**Recommendation** for document extraction is unchanged: gpt-oss-120b stays the on-prem default.
Gemma 4 31B with thinking on takes the dense slot from Qwen3.8-27B by one question at the same
speed, and is the model to use when a filing fits the window and accuracy matters more than the
two hours. Laguna is the fast second option on the same box, four questions behind on one seed,
and the model we should judge on the coding suite it was built for rather than here. Three runs
settle the open points: Laguna with thinking on, to see whether it fixes the two line picks or
loops as its Hugging Face discussion warns; Laguna on the SWE tier 1 suite, the reason it was
registered; and both models at their native windows, where Goldman's statements section should
fit again and the chunk misses should close the way they did for Qwen at 262K on the 5090.

| Stack | Accuracy, 121 Qs | When the filing fits, of 104 Qs | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 262K ctx | 98.3% | 102/104 | 55 t/s | 40–70 s | 24 min | $0 marginal |
| DGX Spark, DeepSeek-V4-Flash-nothink (MoE, 13B active, 2-bit) | 98.3% | 102/104 | 16 t/s | 190–306 s | 51 min | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 262K ctx | 96.7% | 100/104 | 93 t/s | 37–64 s | 20 min | $0 marginal |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| **DGX Spark, Gemma 4 31B (dense), thinking on** | **96.7%** | **104/104** | **7.9 t/s** | **202–356 s** | **130 min** | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| **DGX Spark, Gemma 4 31B (dense), thinking off** | **95.0%** | **102/104** | **7.8 t/s** | **209–364 s** | **61 min** | $0 marginal |
| **DGX Spark, Laguna S 2.1 (MoE, 8B active), thinking off** | **95.0%** | **102/104** | **17 t/s** | **97–157 s** | **29 min** | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 131K ctx | 94.2% | 102/104 | 60 t/s | 38–54 s | 23 min | $0 marginal |
| DGX Spark, Bonsai 2 27B (ternary), 131K ctx | 93.4% | 102/104 | 16 t/s | 120–200 s | 107 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, 131K budget | 93.4% | 99/104 | 20 t/s | 105–160 s | 148 min | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 131K ctx | 92.6% | 101/104 | 98 t/s | 37–53 s | 21 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, full 1M window | 92.6% | 97/104 | 19 t/s | 105–160 s; GS whole 6.5 min | 153 min | $0 marginal |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | 45 t/s | 40–60 s | 19 min | $0 marginal |

"When the filing fits" counts the 104 questions on the ten filings that fit a 131K window, so the
column is comparable across tokenizers. Under Gemma's and Laguna's tokenizers NextEra no longer
fits whole, so its 11 questions were answered from the financial-statements section; they still
count in this column, and both models got them right.

Two housekeeping notes: Laguna's checkpoint was registered at about 40 GB from a secondary source
and is 73.4 GB on Hugging Face; the registry and card are corrected in this commit. Its thinking
twin is named `-thinking` rather than `-nothink`, because the vendor default is off, and the smoke
script now handles both conventions. The memorisation control, same questions with no document,
and the Laguna thinking run are the outstanding items before any of this goes outside the company.

**Write-ups**

- Gemma 4 31B and Laguna S 2.1 on the Spark, with every miss itemised and the tokenizer table: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-spark-cuda-gemma-laguna.md
- Gemma model card: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/models/gemma-4-31b.md
- Laguna model card, with the configuration used and the open questions: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/models/laguna-s-2.1.md
- DeepSeek-V4-Flash note from Friday, for the comparison: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/emails/2026-09-19-deepseek-v4-flash-nothink.md
- Which report covers what: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/index.md
- Canonical cross-stack comparison table, updated with every run: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-comparison-table.md
- Machine-generated table over all runs, with intervals: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/state/evals/report-sec.md

Thanks,
Kaushik

---

## Fact trail

| Claim | Source |
| --- | --- |
| Laguna S 2.1: 118B total / ~8B active, 48 layers, 12 global + 36 SWA (window 512), 256 routed (top-10) + 1 shared, 8 KV heads, 1,048,576 ctx, OpenMDW-1.1, `enable_thinking` kwarg | https://huggingface.co/poolside/Laguna-S-2.1 |
| Released 2026-07-21; XS.2 / M.1 2026-04-28 | https://poolside.ai/blog/introducing-laguna-s-2-1 , https://poolside.ai/blog/introducing-laguna-xs2-m1 |
| UD-Q4_K_XL is 73.4 GB in 3 shards | https://huggingface.co/unsloth/Laguna-S-2.1-GGUF/tree/main/UD-Q4_K_XL |
| llama.cpp Laguna support merged 2026-07-22, PR #25165; pin cut 2026-09-11 | https://github.com/ggml-org/llama.cpp/pull/25165 ; `memory/spark-speed-plan-parked.md` (pin date) |
| Gemma on: 117/121, CI 0.93–0.99, 104/104 fits, 0 truncated, 7.9 t/s, 367 t/s prefill, TTFT 202–356 s (n=6), 130 min, 32,696 reasoning tokens (p50 202), completion p50 216 | `state/evals/sec/gemma-4-31b-spark-cuda/20260919T174420Z-extract-full` via `facts.py` |
| Gemma off: 115/121, CI 0.91–0.98, 102/104, 0 truncated, 7.8 t/s, 355 t/s, TTFT 209–364 s, 61 min, 0 reasoning, completion p50 12 | `state/evals/sec/gemma-4-31b-nothink-spark-cuda/20260919T202304Z-extract-full` |
| Laguna: 115/121, CI 0.91–0.98, 102/104, 0 truncated, 17.2 t/s, 816 t/s, TTFT 97–157 s, 29 min, 0 reasoning, completion p50 13 | `state/evals/sec/laguna-s-2.1-spark-cuda/20260920T003059Z-extract-full` |
| Per-filing modes and doc_tokens (GS 271,429 / 274,903 / 242,026; NEE 136,728 / 138,457 / 124,130; STWD 200,441 / 203,768 / 176,354); by-mode counts 93/93, 20/20, 4/8 (Gemma on); 91/93, 20/20, 4/8 (off); 92/93, 19/20, 4/8 (Laguna) | computed from the three `results.jsonl` files and the gpt-oss-120b baseline (`ticker`, `mode`, `correct`, `doc_tokens`) |
| Miss lists and values, including the 608-token prose answer on GS net income | same `results.jsonl` files (`answer`, `expected`, `completion_tokens`) |
| Warm re-prefill gap median 86 (Gemma on, Laguna), 90 (Gemma off), 63 (gpt-oss); cache reuse 8,902,194 / 10,295,657 = 86.5%, 8,901,986 / 10,262,738 = 86.7%, 9,005,904 / 10,378,894 = 86.8% | `results.jsonl` (`prompt_tokens`, `cached_prompt_tokens`); `state/evals/report-sec.md` tokens / cached columns |
| 69 min = 32,696 reasoning tokens / 7.9 t/s | arithmetic on the two Gemma run records |
| Bytes per token: gpt-oss 5.1B × 4.25 b = 2.7 GB; Laguna 8B × (73.4 GB × 8 / 118B ≈ 5.0 b) ≈ 5.0 GB; Gemma 30.7B × (17.65 GB × 8 / 30.7B ≈ 4.6 b) ≈ 17.6 GB | arithmetic; gpt-oss model card https://arxiv.org/pdf/2508.10925 |
| Fallback order full → Item 8 → BM25 top-6 × 8,000 tokens | `src/spark_llm/evals/sec/run.py` `ContextBuilder` |
| gpt-oss-120b baseline 119/121, 30.3 t/s, 26 min, GS 7/8 on section, misses GS op. cash flow and CAT long-term debt | `state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full` |
| Qwen 9.8 t/s, 116/121, 125 min; Nemotron 131K reasoning tokens | `state/evals/report-sec.md`; `docs/eval-report-2026-09-spark-cuda-nemotron.md` |
| Gemma card predictions (~10 t/s, cheaper filings, ~2,028-token SWA re-prefill) | `docs/models/gemma-4-31b.md` as committed in bb062db |
| Thinking-loop report | https://huggingface.co/poolside/Laguna-S-2.1/discussions/23 |
| Prior comparison table rows | `docs/eval-comparison-table.md` |
