# gpt-oss-120b

**Status:** measured (Spark CUDA; Halo Vulkan and ROCm via LM Studio). **On-prem default for document extraction.**

## Identity

| | |
| --- | --- |
| Family / vendor | OpenAI gpt-oss (open weights, Aug 2025) |
| Architecture | Mixture-of-experts, ~117B total, ~5.1B active per token; reasoning model (hidden chain of thought via `reasoning_content`) |
| Checkpoint served | `ggml-org/gpt-oss-120b-GGUF`, `gpt-oss-120b-MXFP4.gguf` (~63 GB) |
| Native context | 131,072 |
| License | Apache-2.0 |
| Registry entry | `models.toml` `[models.gpt-oss-120b]`, port 8082 |

## Serving configuration used

| Knob | Spark | Halo (LM Studio 2.37, both backends) |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f`, CUDA `121a-real` | LM Studio runtime packs `…-vulkan-avx2` / `…-amd-rocm-avx2` 2.37.0 |
| Backend | cuda | vulkan / rocm |
| ctx_size / n_parallel | 131072 / default (4 slots at 131K each) | 131072 |
| Batch / ubatch | 2048 / 2048 | eval batch 512 |
| Flash attention / KV type | on / f16 | on / f16, unified KV, KV on GPU |
| Extra flags | `--jinja`, `n_gpu_layers = 70` in registry (999 works; 70 was a GB10 tuning value) | 36/36 layers offloaded, keep in memory |
| Sampling (serving only) | registry defaults | LM Studio defaults |

Eval decoding: temperature 0, seed 42, `max_tokens` 4096, `reasoning_effort` unset (model default).
The first Spark run used `max_tokens` 512 and truncated ~15% of answers mid-reasoning; that run is
kept only as the pre-fix baseline.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill (~100K filing) | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Spark, pre-fix (512 budget) | extract-full, 10-K | 90.3% raw / 89.6% paired-96 | — | 17 skipped (GS, STWD over budget) | 6 | 30 t/s | 40–60 s | 22 min | `state/evals/sec/gpt-oss-120b-spark-cuda/20260912T030429Z-extract-full` |
| **Spark, current** | extract-full, 10-K | **98.3% (119/121)** | 0.96–1.00 | 104 full / 17 Item-8 section | 0 | 30 t/s | 40–60 s (p95 63 s) | 26 min | `…/20260912T132250Z-extract-full` |
| Halo Vulkan (LM Studio) | extract-full, 10-K | 96.7% (117/121) | 0.93–0.99 | 104 full / 17 section | 0 | 29 t/s | 4–8 min (p95 300 s) | 77 min | `state/evals/sec/gpt-oss-120b-halo-vulkan/20260913T030643Z-extract-full` |
| Halo ROCm (LM Studio) | extract-full, 10-K | 95.0% (115/121) | 0.91–0.98 | 104 full / 17 section | 0 | 19 t/s | 4–8 min (p95 238 s) | 72 min | `state/evals/sec/gpt-oss-120b-halo-rocm/20260913T044559Z-extract-full` |

Prompt-cache reuse on the Spark: ~90% of prompt tokens across questions on the same filing.
Reasoning: ~140 hidden tokens per question at the median (20.7k total).

## Strengths (measured)

- On every filing that fits in 131K, ties the hosted frontier model: 103/104 on the Spark and on the Halo (Vulkan).
- Fast decode for its size: MoE keeps ~5B active parameters per token, so 30 t/s on a bandwidth-bound box, and the same 29 t/s on the AMD box.
- Zero scale errors on share counts once the completion budget was adequate; 10/10 on `CommonStockSharesOutstanding`.
- Deterministic-looking at temperature 0 with a fixed seed (not verified with a repeat run).

## Weaknesses (measured)

- 131K native context: GS (242K tokens) and STWD (176K) are read from the Item 8 excerpt, not whole. Both remaining Spark misses are which-line picks under that constraint (GS parent-only vs consolidated cash flow; CAT consolidated vs machinery-only debt).
- Hidden reasoning is charged against `max_tokens`; 512 was far too small (all six of its first-run non-shared misses were truncations). 4096 has produced zero truncations since.
- On Windows/ROCm via LM Studio decode drops to 19 t/s and accuracy to 95.0%; use Vulkan on that box.
- ~63 GB of weights plus KV leaves little room for a second large model on a 128 GB box.

## When to use / when not to

- **Use** for unattended numeric extraction from long financial documents on-prem; it is the model the recommendation rests on.
- **Use** on either box for background work; on the Halo expect a 3x wall-clock penalty from prefill.
- **Do not** assume the accuracy transfers to reasoning tasks; nothing here measures that (see the Qwen card and FinanceBench follow-up).

## Open questions

- Memorisation control (`--no-document`) not yet run: how much of the 98.3% is recall of public XBRL facts?
- Behaviour under concurrent load (4 slots are configured; nothing measured).
- Whether `n_gpu_layers = 70` vs 999 changes anything on GB10 (both fit).

## Changelog

- 2026-09-12 — first Spark run (512-token budget) and post-fix re-run.
- 2026-09-13 — Halo Vulkan and ROCm runs via LM Studio; hardware comparison published.
- 2026-09-15 — card created; CIs from `report-sec.md`.
