# glm-4.7-flash

**Status:** planned (registered 2026-09-17, no run yet). **The small-and-fast candidate: same
active size as gpt-oss-20b, 200K window, MIT licence. Question to answer: does it read 10-Ks
better than 20b's 87.6%, and does thinking help or hurt it?**

## Identity

| | |
| --- | --- |
| Family / vendor | Z.ai (Zhipu) GLM-4.7 family, "Flash" size; open weights released January 2026 (verify on the model card) |
| Architecture | Mixture-of-experts, 30B total, ~3.6B active per token: 47 layers, 64 routed experts + 1 shared, 4 routed per token, MLA attention (KV LoRA rank 512, 20 heads), 154,880-token vocabulary; llama.cpp arch `glm4_moe_lite`; reasoning via hidden thinking, on by default |
| Checkpoint served | `unsloth/GLM-4.7-Flash-GGUF`, quant `UD-Q4_K_XL` (17.5 GB); no vision projector |
| Native context | 202,752 |
| License | MIT |
| Registry entry | `models.toml` and `models.halo.toml` `[models."glm-4.7-flash"]` and `[models."glm-4.7-flash-nothink"]`, port 8086 |

Vendor claims (not ours): strongest model in the 30B class, SWE-bench Verified 59.2, AIME 91.6.

## Serving configuration (planned)

| Knob | Spark | Halo / RTX 5090 |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (glm4_moe_lite support and the Jan 21 2026 router fix are both older) | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (or cuda on the 5090) |
| ctx_size / n_parallel | 131072 / default (native 202752 via `--ctx-size`) | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-nothink` entry adds `--reasoning off` | same |
| Sampling (serving only) | temp 1.0, top_p 0.95 (Z.ai general profile) | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: (1) `extract-full` with
thinking on at 131K, the row that lines up with every other local model; (2) the same with
`glm-4.7-flash-nothink`; (3) SWE tier 1 with thinking off, then on with the 4096 budget, the
pair the RTX 5090 Qwen runs established. If thinking-on truncates, cap it with
`--reasoning-budget 2048` before raising `max_tokens`.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (expected, unmeasured)

- ~3.6B active parameters: gpt-oss-20b-class decode (45 t/s on GB10) if the MoE kernels behave.
- 17.5 GB of weights: serves next to gpt-oss-120b on the Spark, or alone on a 24 GB card.
- MLA attention keeps the KV cache small, so the 200K window is cheap.

## Weaknesses (expected, unmeasured)

- 30B-class models have not been accurate enough for unattended extraction here (20b: 87.6%, three share-count scale errors).
- Thinking on by default; the 5090 runs showed thinking can cost more than it earns at a 4096 budget.
- Its tokenizer (154,880 vocab) has not been measured on our corpus; the fallback boundary will move.

## When to use / when not to

- Not yet. Decide after run (1) and (2): if it beats 20b clearly and lands near 120b, it is the fast option for human-reviewed extraction and the coding-tier candidate; if it lands with 20b, the small-model slot stays with gpt-oss-20b.

## Open questions

- Accuracy on `extract-full` with thinking on vs off, and the truncation count at 4096.
- Token count of the 12 filings under the GLM tokenizer (drives fallback).
- Coding tier 1 vs Qwen3.8-27B's reasoning-off 90.9% / 76.5% on the 5090.

## Changelog

- 2026-09-17 — registered on both platforms with a thinking-off twin; card created.
