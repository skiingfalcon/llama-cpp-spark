# bonsai-2-27b

**Status:** measured (RTX 5090 CUDA, PrismML llama.cpp fork). **Qwen3.8-27B at 1.7 bits per weight: 2.4x less weight memory, 2–3 questions behind the 4-bit Qwen on extraction, more thinking overruns, ~1.6x faster decode.**

## Identity

| | |
| --- | --- |
| Family / vendor | PrismML Ternary Bonsai 2 27B (announced 2026-09-17), derived from Alibaba Qwen3.8-27B |
| Architecture | Dense 27B, Qwen3.8 hybrid-attention backbone (~75% linear attention), vision-language checkpoint served text-only; weights ternary (-1, 0, +1) with FP16 group-128 scales in a rotated (Hadamard) basis, 1.72 bits/weight; reasoning via hidden thinking, default effort `xhigh` |
| Checkpoint served | `prism-ml/Ternary-Bonsai-2-27B-gguf`, file `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB; the `PTQ1_0` pack is 5.95 GB) |
| Native context | 262,144 |
| License | Apache-2.0 (verify on the model card) |
| Registry entry | `models.toml` / `models.halo.toml` `[models."bonsai-2-27b"]` and `bonsai-2-27b-nothink`, port 8088 |
| Engine | **Requires PrismML's llama.cpp fork** (`github.com/PrismML-Eng/llama.cpp`, custom ternary kernels); stock llama.cpp rejects the `PQ2_0` / `PTQ1_0` tensor types |

## Serving configuration used

| Knob | RTX 5090 run 1 | RTX 5090 run 2 | Tier 1 (WSL) |
| --- | --- | --- | --- |
| llama.cpp | PrismML fork `prism-b10685-7dffb158d`, Windows CUDA 13.3 zip | same | same fork, Linux CUDA 12.8 tarball, served inside WSL2 |
| ctx_size / n_parallel | 131072 / default (4 auto slots, unified KV) | 262144 / default | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| Flash attention / KV type | on / f16 | on / **f16** (no cache quantisation needed) | on / f16 |
| Extra flags | `--jinja`, `n_gpu_layers = 999` | same | `--jinja`, `--reasoning off` for the reasoning-off row |
| VRAM in use (server, observed) | ~21 GB of 32 | ~30.5 GB of 32 | ~22 GB |

Eval decoding: temperature 0, seed 42, `max_tokens` 4096. Tokenizer identical to Qwen3.8-27B
(same token counts on every filing).

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| **RTX 5090, 131K** | extract-full, 10-K | 92.6% (112/121) | 0.88–0.97 | 90 full, 20 Item-8 section, **8 BM25 chunks** (GS 2/8) | **5** | 98 t/s | 37–53 s per ~100K filing (cold TTFT p50 28 s) | 20.6 min | `state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T152253Z-extract-full` |
| **RTX 5090, 262K, f16 KV** | extract-full, 10-K | **96.7% (117/121)** | 0.93–0.99 | 109 full, **8 Item-8 section (GS 8/8)**, no chunks | 2 | 93 t/s | 37–64 s (cold TTFT p50 41 s; STWD 202K whole: 121 s) | 19.8 min | `state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T154449Z-extract-full` |
| RTX 5090, 131K | qa-financebench, 150 Qs | numeric 59/74 (80%) unit-tolerant any-number; harness raw 8/74 (scorer defect); 76 free-text unscored (no judge) | 0.70–0.89 (numeric, lenient) | evidence excerpts as context | 5 | 141 t/s | n/a | 15.3 min | `state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T160529Z-qa-financebench` |
| RTX 5090, **reasoning off** | SWE tier 1 (evalplus, greedy) | HumanEval+ **86.6%** (142/164), MBPP+ **74.1%** (280/378); base 91.5% / 86.2% | 0.81–0.91 / 0.70–0.79 | evalplus default 768-token budget; served inside WSL2 | 0 blank answers | n/a (evalplus records no latency) | n/a | 22.2 min | `state/evals/swe/bonsai-2-27b-halo-vulkan/20260918T164442Z-tier1` |
| RTX 5090, reasoning on | SWE tier 1 | not run (stopped 2026-09-18 to free the GPU) | | 4096-token budget | | | | | |

Comparison rows for the 4-bit Qwen3.8-27B on the same card are in
[qwen3.8-27b.md](qwen3.8-27b.md): 114/121 at 131K, 119/121 at 262K (q8_0 KV), FinanceBench
numeric 60/74 lenient. Details and every miss:
[eval-report-2026-09-rtx5090-bonsai2.md](../eval-report-2026-09-rtx5090-bonsai2.md).

## Strengths (measured)

- **Same behaviour as the full-precision model where it reads the document:** section-mode answers
  perfect in both runs (20/20 and 8/8), GS recovered at 262K exactly as with Qwen, NEE and STWD
  read whole.
- **Footprint:** 7.2 GB of weights. 262K context fits in f16 KV on a 32 GB card with ~1.5 GB to
  spare; 131K uses ~21 GB, so a 24 GB card could serve it at the standard budget.
- **Faster decode than the 4-bit Qwen:** 93–98 t/s vs 55–60 t/s; whole extraction suite in ~20
  min vs 23–24.
- **No scale errors**, and it got the MSFT and HD share counts the 4-bit Qwen missed at 262K.

## Weaknesses (measured)

- **Two to three questions behind the 4-bit Qwen at each context setting** (112 vs 114 at 131K,
  117 vs 119 at 262K), one behind on FinanceBench numeric (59 vs 60), three points behind on
  coding with reasoning off (HumanEval+ 86.6 vs 90.9, MBPP+ 74.1 vs 76.5); all inside overlapping
  intervals, all the same sign; single runs.
- **Fatter thinking tail:** median hidden reasoning is shorter than Qwen's (188–202 vs 224–244
  tokens) but more questions run to the 4,096 cap: 5 at 131K and 2 at 262K (vs 1 and 0), plus 5
  of 150 on FinanceBench (vs 2). Every truncation is empty content after 4,096 reasoning tokens.
- **One genuine wrong number, stable across runs:** XOM revenues 323.9B ("sales and other
  operating revenue") for 332.2B (total revenues), 2.5% off.
- **Decode gain is well short of the 2.4x weight reduction** (1.6–1.7x). PrismML's own notes say
  batch-1 decode on Blackwell is instruction-throughput bound, not bandwidth bound.
- **Prefill is not faster** (compute-bound; 2,400–3,500 t/s cold, same as Qwen).
- **Engine lock-in:** needs the PrismML fork; a Spark run means building their fork, not the
  pinned upstream commit.

## When to use / when not to

- **Use** where memory is the constraint: a 24 GB card, or serving alongside another model, at
  a two-to-three-question cost on this suite.
- **Serve with reasoning off** (`bonsai-2-27b-nothink`) for short-answer work, or raise the
  budget: its thinking overruns 4,096 tokens more often than the base model's.
- **Do not** treat it as a free speed upgrade for the 4-bit Qwen on a 32 GB card: the wall-clock
  gain is 2–4 minutes per suite and the accuracy is slightly lower.
- **Do not** report it under the pinned upstream llama.cpp; the fork is part of the result.

## Open questions

- The `PTQ1_0` pack (5.95 GB): PrismML says it decodes faster on Ada-class cards and slower on
  Blackwell; unmeasured here.
- Reasoning effort `medium` (PrismML's suggestion for shorter answers) vs the `xhigh` default:
  would it remove the overruns without losing the section-mode accuracy?
- Same runs on the Spark (requires building the fork there), for the platform comparison.
- FinanceBench free-text judge pass, as for every other model.

## Changelog

- 2026-09-18 — first runs, RTX 5090 (PrismML fork b10685): extraction 131K 112/121 and 262K
  117/121 (GS 8/8), FinanceBench numeric-only, tier 1 reasoning off (reasoning on not run); report
  `docs/eval-report-2026-09-rtx5090-bonsai2.md`; card created.
