# gpt-oss-20b

**Status:** measured (Spark CUDA). **Not accurate enough for unattended extraction.**

## Identity

| | |
| --- | --- |
| Family / vendor | OpenAI gpt-oss (open weights, Aug 2025) |
| Architecture | Mixture-of-experts, ~21B total, ~3.6B active per token; reasoning model |
| Checkpoint served | `ggml-org/gpt-oss-20b-GGUF`, `gpt-oss-20b-MXFP4.gguf` (~12 GB) |
| Native context | 131,072 |
| License | Apache-2.0 |
| Registry entry | `models.toml` `[models.gpt-oss-20b]`, port 8080 |

## Serving configuration used

| Knob | Spark |
| --- | --- |
| llama.cpp | pinned `82d6bb284d1f`, CUDA `121a-real` |
| ctx_size / n_parallel | `ctx_size = 0` (auto-negotiates the model's 131,072) / default |
| Batch / ubatch | 2048 / 2048 |
| Flash attention / KV type | on / f16 |
| Extra flags | `--jinja`, `n_gpu_layers = 999` |

Eval decoding: temperature 0, seed 42, `max_tokens` 4096 (first run 512).

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Off-by-scale | Decode | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Spark, pre-fix (512 budget) | extract-full, 10-K | 76.7% raw / 77.1% paired-96 | — | 17 skipped | 14 | 2 | 47 t/s | 14 min | `state/evals/sec/gpt-oss-20b-spark-cuda/20260912T022410Z-extract-full` |
| **Spark, current** | extract-full, 10-K | **87.6% (106/121)** | 0.81–0.93 | 94/104 full, 12/17 section | 1 | 3 | 45 t/s | 19 min | `…/20260912T135023Z-extract-full` |

Reasoning: ~230 hidden tokens per question (28.2k total), more than the 120B model.

## Strengths (measured)

- Fastest local decode measured (45 t/s) and the smallest footprint (~12 GB); leaves room to serve other models alongside.
- Perfect on the easy tags: EPS, operating income, stockholders' equity all 12/12 or 9/9.
- Cheapest way to smoke-test the harness end to end.

## Weaknesses (measured)

- 15 misses on 121, and they are genuine model errors, not harness artefacts: three share counts reported in thousands instead of units, wrong debt lines (3 of 9), subsidiary-level totals for NextEra, an `unknown` on Home Depot net income, one truncation at 4096.
- Weak on the Item 8 fallback (12/17) and on Goldman Sachs specifically (4/8).
- Reasons longer than the 120B model and gets less for it.

## When to use / when not to

- **Use** for latency-sensitive or memory-tight serving where a human reviews the output, and for harness development.
- **Do not** use for unattended numeric extraction; the share-count scale errors alone disqualify it.

## Open questions

- Would `reasoning_effort = low` cut the wasted thinking without hurting accuracy? Untested.
- Not yet run on the Halo (LM Studio path would need the model loaded there).

## Changelog

- 2026-09-12 — first Spark run and post-fix re-run.
- 2026-09-15 — card created.
