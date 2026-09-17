# qwen3.5-122b-a10b

**Status:** planned (registered 2026-09-17, no run yet). **The like-for-like challenger to
gpt-oss-120b: Qwen's reading quality in a mixture-of-experts body that should decode 2x faster
than the dense 27B on the Spark.**

## Identity

| | |
| --- | --- |
| Family / vendor | Alibaba Qwen 3.5 (open weights, early 2026; verify licence and date on the model card) |
| Architecture | Mixture-of-experts, 122B total / ~10B active: 256 routed experts, 8 per token, plus 1 shared; 48 layers, 12 full attention (32 heads, 2 KV heads, head dim 256) and 36 linear gated-DeltaNet layers; vision-language checkpoint served text-only; llama.cpp arch `qwen35moe`; reasoning via hidden thinking, on by default |
| Checkpoint served | `unsloth/Qwen3.5-122B-A10B-GGUF`, quant `UD-Q4_K_XL` (~70 GB, split shards); `mmproj-F16.gguf` not loaded (text only) |
| Native context | 262,144 (1M with YaRN, untested) |
| License | Apache-2.0 (verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models."qwen3.5-122b-a10b"]` and `[models."qwen3.5-122b-a10b-nothink"]`, port 8087 |

## Serving configuration (planned)

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan |
| ctx_size / n_parallel | **262144 / 1** (whole native window on one slot) | 262144 / 1 |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (24 KiB per token: ~6 GiB at 262K) | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-nothink` entry adds `--reasoning off` | same |
| Sampling (serving only) | thinking: temp 0.6, top_p 0.95, top_k 20; non-thinking: temp 0.7, top_p 0.8, top_k 20 | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: (1) `extract-full` at the
native 262K with thinking on, the configuration that recovered Goldman for the dense 27B on the
RTX 5090; (2) the same with `qwen3.5-122b-a10b-nothink`; (3) SWE tier 1, thinking off then on.
Memory: ~70 GB weights + ~6 GiB KV; stop gpt-oss-120b before serving.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (expected, unmeasured)

- The dense sibling missed nothing it could see (113/113 on whole filings and sections on the Spark; 119/121 at 262K on the 5090). If that survives the MoE, this is a second on-prem default candidate.
- ~10B active parameters: the bandwidth rule predicts ~20 t/s on GB10, between Nemotron (12.7B, 20 t/s) and gpt-oss-120b (5.1B, 30 t/s), versus 10 t/s for the dense 27B.
- Only 12 of 48 layers hold a KV cache, so the 262K window costs ~6 GiB.

## Weaknesses (expected, unmeasured)

- Qwen's tokenizer is 10–15% fatter than gpt-oss's; GS (~273K tokens) still exceeds 262K and falls to its Item 8 section.
- 70 GB of weights: not servable beside gpt-oss-120b; on the Halo it sits just under the 96 GB graphics-memory cap; does not fit the RTX 5090.
- Thinking on by default; the 27B spent all 4096 tokens thinking on one share-count question per run.

## When to use / when not to

- Not yet. Decide after run (1): if accuracy matches gpt-oss-120b and decode lands near 20 t/s, it becomes the second default and the model for documents up to 262K; if it decodes below 15 t/s or misses on whole documents, the Qwen family question is closed.

## Open questions

- Does the reading quality of the dense 27B survive the MoE routing?
- Decode and cold prefill on GB10 versus the bandwidth prediction.
- Thinking on vs off on extraction and on tier 1.

## Changelog

- 2026-09-17 — registered on both platforms with a thinking-off twin; card created.
