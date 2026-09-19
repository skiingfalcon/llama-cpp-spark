# gemma-4-31b

**Status:** planned (registered 2026-09-19, no run yet). **Google's dense mid-size model: the
like-for-like rival to qwen3.8-27b, with a 256K window and a 262K-token vocabulary that should let
more filings fit the 131K budget whole. Question to answer: does it read 10-Ks as well as Qwen, and
what does thinking on vs off cost it?**

## Identity

| | |
| --- | --- |
| Family / vendor | Google DeepMind Gemma 4 (open weights, March 2026; QAT checkpoints April–June 2026) |
| Architecture | Dense, 30.7B parameters; 60 layers, 32 heads / 16 KV heads, head dim 256; 1024-token sliding-window attention interleaved 5:1 with global layers, final layer global; global layers share keys and values and use proportional RoPE; optional ~550M vision encoder (not loaded); llama.cpp arch `gemma4`; reasoning via hidden thinking, toggled by `enable_thinking` in the chat template (`<|think|>` at the start of the system prompt) |
| Checkpoint served | `google/gemma-4-31B-it-qat-q4_0-gguf`, file `gemma-4-31B_q4_0-it.gguf` (17.65 GB, Google's quantisation-aware-trained q4_0); `gemma-4-31B-it-mmproj.gguf` (1.2 GB) not loaded, text only |
| Native context | 262,144 |
| License | Apache-2.0 (Gemma 4 licence page; verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models."gemma-4-31b"]` and `[models."gemma-4-31b-nothink"]`, port 8089 |

Vendor claims (not ours): frontier-level performance at each size; MRCR v2 8-needle 128K 66.4%.

## Serving configuration (planned)

| Knob | Spark | Halo / RTX 5090 |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (arch `gemma4` and the gemma4 chat parser are both older) | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (or cuda on the 5090) |
| ctx_size / n_parallel | 131072 / default (native 262144 via `--ctx-size`) | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (SWA layers keep a 1024-token window; global layers share K/V; read "KV self size" from the server log) | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-nothink` entry adds `--reasoning off` | same |
| Sampling (serving only) | temp 1.0, top_p 0.95, top_k 64 (Google's profile) | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: `scripts/spark-new-models-smoke.sh`
runs a 22-item gate (AAPL, XOM) for both twins, then (1) `extract-full` with thinking on at 131K,
the row that lines up with every other local model; (2) the same with `gemma-4-31b-nothink`;
(3) SWE tier 1 with thinking off, then on; (4) the native 262K window, the run that recovered
Goldman for Qwen.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (expected, unmeasured)

- Dense 30.7B at 4-bit QAT: Google reports near-bf16 quality for the QAT checkpoints, which is the
  class in which Qwen3.8-27B read every document it saw correctly.
- 262,144-token vocabulary: filings should cost fewer tokens than under Qwen's or gpt-oss's
  tokenizer, so the 131K budget covers more filings whole and the fallback boundary moves.
- 17.65 GB of weights: serves next to gpt-oss-120b on the Spark, fits the RTX 5090 with the 131K
  KV cache.
- Multi-token-prediction drafters exist (`gemma4-assistant`; unsloth ships `mtp-gemma-4-31B-it.gguf`,
  0.28 GB, `--spec-type draft-mtp --spec-draft-n-max 4`) with identical output: the cheapest
  speed-up on the list once the parked speed plan resumes.

## Weaknesses (expected, unmeasured)

- Dense: the bandwidth rule predicts ~10 t/s on GB10, as for qwen3.8-27b (125 min per suite).
- Sliding-window attention takes llama-server's context-checkpoint path, so each warm question
  likely re-prefills about one micro-batch (qwen3.8-27b: ~2,028 tokens, ~8 s) instead of a few
  hundred tokens.
- Thinking on by default; the 4096-token budget has cost every thinking model here at least one
  answer.
- Vision tower shipped but unused; not measured on this workload.

## When to use / when not to

- Not yet. Decide after runs (1) and (2): if it lands with Qwen (116/121) or better at 131K it is
  the second dense candidate and the model to take to 262K; if it lands below glm-4.7-flash
  (114/121) at three times the decode cost, the dense slot stays with Qwen.

## Open questions

- Warm `prompt_tokens - cached_prompt_tokens` gap on the Spark versus Qwen3.8's ~2,028.
- "KV self size" at 131K with shared global K/V; whether 262K fits beside gpt-oss-120b.
- Token counts for GS, STWD and NEE under the 262K vocabulary: which filings now fit whole.
- Thinking on vs off on extraction and on tier 1; the MTP drafter's acceptance on 10-K text.

## Changelog

- 2026-09-19 — registered on both platforms with a thinking-off twin; card created.
