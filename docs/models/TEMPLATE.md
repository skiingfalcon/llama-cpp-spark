# <model name as registered in models.toml>

**Status:** planned | measured | retired

## Identity

| | |
| --- | --- |
| Family / vendor | |
| Architecture | dense / MoE / hybrid; total and active parameters |
| Checkpoint served | repo, file or quant, size on disk |
| Native context | |
| License | (verify on the model card) |
| Registry entry | `models.toml` `[models.<name>]`, port |

## Serving configuration used

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned commit / release tag | |
| Backend | | |
| ctx_size / n_parallel | | |
| Batch / ubatch | | |
| Flash attention / KV type | | |
| Extra flags | | |
| Sampling (serving only) | | |

Eval decoding is fixed by `evals.toml [quality]`: temperature 0, seed 42, `max_tokens` 4096
unless a run says otherwise.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (measured)

-

## Weaknesses (measured)

-

## When to use / when not to

-

## Open questions

-

## Changelog

- YYYY-MM-DD — created.
