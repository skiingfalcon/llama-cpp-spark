# gemma-4-26b-a4b

**Status:** planned (registered 2026-09-19, no run yet). **Google's mixture-of-experts model with
3.8B active parameters: the same active size as gpt-oss-20b and glm-4.7-flash, so the fast-decode
candidate. Question to answer: does it read 10-Ks better than 20b (87.6%) and GLM (94.2%) did at
the same speed?**

## Identity

| | |
| --- | --- |
| Family / vendor | Google DeepMind Gemma 4 (open weights, March 2026; QAT checkpoints April–June 2026) |
| Architecture | Mixture-of-experts, 25.2B total / 3.8B active; 30 layers, 128 experts, 16 heads / 8 KV heads, head dim 256; 1024-token sliding-window attention interleaved 5:1 with global layers (shared K/V on global layers); optional ~550M vision encoder (not loaded); llama.cpp arch `gemma4`; reasoning via hidden thinking, toggled by `enable_thinking` |
| Checkpoint served | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, file `gemma-4-26B_q4_0-it.gguf` (14.44 GB, Google's QAT q4_0); `gemma-4-26B-it-mmproj.gguf` (1.19 GB) not loaded, text only |
| Native context | 262,144 |
| License | Apache-2.0 (Gemma 4 licence page; verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models."gemma-4-26b-a4b"]` and `[models."gemma-4-26b-a4b-nothink"]`, port 8091 |

Vendor claims (not ours): "runs almost as fast as a 4B-parameter model"; MRCR v2 8-needle 128K 44.1%.

## Serving configuration (planned)

| Knob | Spark | Halo / RTX 5090 |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (or cuda on the 5090) |
| ctx_size / n_parallel | 131072 / default (native 262144 via `--ctx-size`) | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (read "KV self size" from the server log) | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-nothink` entry adds `--reasoning off` | same |
| Sampling (serving only) | temp 1.0, top_p 0.95, top_k 64 | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: the 22-item gate in
`scripts/spark-new-models-smoke.sh`, then `extract-full` at 131K with thinking on and off, then SWE
tier 1 with thinking off.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (expected, unmeasured)

- 3.8B active parameters: the bandwidth rule predicts gpt-oss-20b-class decode (45 t/s on GB10) if
  the expert kernels behave, i.e. a suite in about 20 minutes.
- 14.44 GB of weights: serves beside gpt-oss-120b on the Spark; fits a 24 GB card.
- Same 262K vocabulary and 256K window as the 31B.

## Weaknesses (expected, unmeasured)

- Small-active-parameter models have not been accurate enough for unattended extraction here
  (gpt-oss-20b 87.6% with scale errors; glm-4.7-flash 94.2% with four truncations).
- Same sliding-window checkpoint behaviour as the 31B: expect a warm re-prefill of about one
  micro-batch per question.
- Thinking on by default with a 4096-token budget.

## When to use / when not to

- Not yet. Decide after the two 131K runs: if it beats glm-4.7-flash clearly and decodes near
  40 t/s it is the fast option for human-reviewed extraction; if it lands with 20b, the small-model
  slot stays as it is.

## Open questions

- Decode on GB10 versus the 3.8B-active prediction; expert-routing cost at 100K+ context.
- Warm cache gap; token counts under the 262K vocabulary.
- Thinking on vs off; MTP drafter (`mtp-gemma-4-26B-A4B-it.gguf`, 0.25 GB) as a follow-up.

## Changelog

- 2026-09-19 — registered on both platforms with a thinking-off twin; card created.
