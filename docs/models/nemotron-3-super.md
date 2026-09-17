# nemotron-3-super

**Status:** measured twice on Spark (131K input budget and full 1M window). **Reads every
filing whole at 1M, but the bigger window did not raise the headline score; over-thinking /
truncation at 4096 still dominates. A thinking cap or higher `max_tokens` is the next lever.**

Long-form write-up:
[eval-report-2026-09-spark-cuda-nemotron.md](../eval-report-2026-09-spark-cuda-nemotron.md).

## Identity

| | |
| --- | --- |
| Family / vendor | NVIDIA Nemotron 3 Super (open weights, spring 2026) |
| Architecture | Hybrid: 40 Mamba-2 layers, 40 latent-MoE layers, 8 grouped-query attention layers (2 KV heads, head dim 128); 120.6B total, ~12.7B active per token; NoPE positions (no RoPE scaling needed); reasoning via `<think>` tokens exposed as `reasoning_content` |
| Checkpoint served | `ggml-org/nemotron-3-super-120b-GGUF`, `Nemotron-3-Super-120B-Q4_K.gguf` (69.9 GB). Do **not** use Ollama's GGUF blobs (expert tensors packed differently; `ffn_down_exps … wrong shape` in upstream llama.cpp) |
| Native context | 1,048,576 |
| License | NVIDIA Open Model License (verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models.nemotron-3-super]`, port 8085 |

## Serving configuration (used)

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (Nemotron-3 support landed March 2026) | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (rocm second) |
| ctx_size / n_parallel | **524288 / 1** default; unconstrained run used **1048576 / 1** | 524288 / 1 |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (KV is only 8 KiB per token: 4 GiB at 524K, 8 GiB at 1M) | on / f16, `--no-mmap` |
| Extra flags | `--jinja` | `--jinja --no-mmap` |
| Sampling (serving only) | temp 0.6, top_p 0.95 (NVIDIA's tool-calling profile) | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096.

Spark tip before the first load: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
(70 GB mmap can OOM against a warm page cache). Load takes ~6 minutes.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Spark, **capped** `--max-input-tokens 131072`, ctx 524288 / 1 slot | extract-full, 10-K | 93.4% (113/121) | 0.88–0.98 | 93 full (88 ✓), 20 Item-8 (20 ✓), 8 BM25 GS (5 ✓) | **6** | 19.7 t/s | 148 min | `…/20260915T122649Z-extract-full` |
| Spark, **unconstrained**, ctx **1048576** / 1 slot | extract-full, 10-K | 92.6% (112/121) | 0.87–0.97 | **121 full (112 ✓), fallback 0** | **6** | 19.4 t/s | 153 min | `…/20260916T103526Z-extract-full` |

Under the 131K cap the tokenizer is ~13% fatter than gpt-oss (GS 274K vs 242K; NEE 139K vs
124K), so NEE fell to Item 8 and GS to chunks — same shape as Qwen. The unconstrained run
removed that variable: every filing was read whole. Net vs the capped run: GS Assets /
Liabilities recovered; GS NetIncomeLoss and two NEE lines (cash, shares) were newly missed
(truncations / wrong number) → **−1 overall**.

Both runs: ~131–134k reasoning tokens (~6.5× gpt-oss-120b). Six of the misses each time are
`finish_reason=length` at 4096 with an empty answer. Persistent wrong-number misses: XOM
Revenues (323.9B vs 332.2B) and PLTR shares (2.291B vs 2.391B).

## Strengths

- Full 1M window is real on the Spark: `fallback=0` on this suite.
- When it finished an answer under the capped run it was almost always right (113 of 115
  non-truncated); Item 8 sections 20/20; no scale errors.
- Decode ~19–20 t/s on GB10, between Qwen (10) and gpt-oss-120b (30), as active-parameter
  count predicts.
- KV cache is small enough that 524K–1M costs little next to 70 GB weights.

## Weaknesses

- **Over-thinks.** Six truncations at 4096 in *both* runs; raising input context did not
  reduce them. That is most of the gap to gpt-oss-120b (98.3%).
- ~150 min wall clock, ~6× gpt-oss-120b, driven by reasoning length more than decode speed.
- Full window did **not** improve the headline vs the 131K-budget run (92.6% vs 93.4%).
- 69.9 GB of weights: fits, but not alongside gpt-oss-120b.

## When to use / when not to

- **Not as the Spark default for this suite.** gpt-oss-120b stays ahead on accuracy and
  latency.
- Reach for Nemotron when you specifically need whole-document reads beyond 131K and can
  afford ~2.5 h per suite — after first testing a thinking cap (`--reasoning-budget`) or a
  higher `max_tokens`, since truncation is the binding failure mode.

## Open questions

- Does `--reasoning-budget 2048` (or `max_tokens` 8192) remove the six truncations without
  hurting the non-truncated items?
- Whether the ggml-org Q4_K quant costs accuracy versus Unsloth's UD-Q4_K_XL (83.8 GB).
- Reasoning toggle: does the chat template honour a no-think mode, and does extraction need
  thinking at all?

## Changelog

- 2026-09-15 — registered with a 524K default window and the `--max-input-tokens` comparison plan; card created.
- 2026-09-15 — first Spark run, capped at 131K: 93.4%, 6 truncations at 4096.
- 2026-09-16 — unconstrained Spark run at 1M ctx: 92.6%, fallback 0, still 6 truncations; comparison report added.
