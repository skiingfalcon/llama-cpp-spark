# nemotron-3-super

**Status:** measured once (Spark, capped at the 131K input budget). **Reads well when it answers; over-thinks and truncates on hard items at the 4096 budget. Unconstrained-context run and a thinking cap are next.**

## Identity

| | |
| --- | --- |
| Family / vendor | NVIDIA Nemotron 3 Super (open weights, spring 2026) |
| Architecture | Hybrid: 40 Mamba-2 layers, 40 latent-MoE layers, 8 grouped-query attention layers (2 KV heads, head dim 128); 120.6B total, ~12.7B active per token; NoPE positions (no RoPE scaling needed); reasoning via `<think>` tokens exposed as `reasoning_content` |
| Checkpoint served | `ggml-org/nemotron-3-super-120b-GGUF`, `Nemotron-3-Super-120B-Q4_K.gguf` (69.9 GB). Do **not** use Ollama's GGUF blobs (expert tensors packed differently; `ffn_down_exps … wrong shape` in upstream llama.cpp) |
| Native context | 1,048,576 |
| License | NVIDIA Open Model License (verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models.nemotron-3-super]`, port 8085 |

## Serving configuration (planned)

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (Nemotron-3 support landed March 2026) | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (rocm second) |
| ctx_size / n_parallel | **524288 / 1** (whole window on one slot); 1M via `--ctx-size 1048576` | 524288 / 1 |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (KV is only 8 KiB per token: 4 GiB at 524K, 8 GiB at 1M) | on / f16, `--no-mmap` |
| Extra flags | `--jinja` | `--jinja --no-mmap` |
| Sampling (serving only) | temp 0.6, top_p 0.95 (NVIDIA's tool-calling profile) | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: run **capped**
(`--max-input-tokens 131072`) to compare with gpt-oss under identical budget, then
**unconstrained** to measure what the window buys. If `truncated` climbs, cap thinking with
llama-server `--reasoning-budget 2048` before raising `max_tokens`.

Spark tip before the first load: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
(70 GB mmap can OOM against a warm page cache). Load takes ~6 minutes.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Spark, **capped** `--max-input-tokens 131072`, ctx 524288 / 1 slot | extract-full, 10-K | 93.4% (113/121) | 0.88–0.98 | 93 full (88 ✓), 20 Item-8 section (20 ✓), 8 BM25 chunks GS (5 ✓) | **6** | 19.7 t/s | cold TTFT 59–160 s, ~800 t/s prefill | 148 min | `state/evals/sec/nemotron-3-super-spark-cuda/20260915T122649Z-extract-full` |

Tokenizer is ~13% fatter than gpt-oss's (GS 274K vs 242K tokens; NEE 139K vs 124K), so under the
131K cap NEE fell to the Item 8 section and GS to chunks, as Qwen did. Reasoning: 133.6k hidden
tokens, 6.5x gpt-oss-120b; completion p50 729 tokens, p90 2,634. Six of the eight misses are
`finish_reason=length` at 4096 with an empty answer (GS cash flow / assets / liabilities from
chunks; XOM debt, HD debt, HD share count from the full document). The two real misses: XOM
revenue (the net-sales line, same ambiguity as every other model's first run) and PLTR share
count (2.291B vs 2.391B, a different date's figure).

## Strengths (measured, one capped run)

- When it finished an answer it was almost always right: 113 of 115 non-truncated items, 20/20 on Item 8 sections, no scale errors.
- Decode 19.7 t/s on GB10, between Qwen (10) and gpt-oss-120b (30), as the active-parameter count predicts.
- Cold prefill ~800 t/s, faster than Qwen (610–740) though below gpt-oss-120b (1,240–1,830).
- KV cache is small enough that the 524K window cost nothing measurable in memory.

## Still to verify

- Every 10-K read whole at 524K–1M: the unconstrained run has not been done.

## Weaknesses (measured)

- **Over-thinks.** 6.5x the hidden reasoning of gpt-oss-120b on the same questions; six truncations at 4096, all with an empty answer. That is the whole gap to gpt-oss on this run.
- 148 minutes wall clock, 5.7x gpt-oss-120b, driven by reasoning length more than decode speed.
- Tokenizer ~13% fatter than gpt-oss: under an equal token budget it hits the fallback ladder sooner.
- 69.9 GB of weights: fits, but not alongside gpt-oss-120b.

## When to use / when not to

- Not yet. Two runs decide it: (1) serve with `-- --reasoning-budget 2048` (or raise `--max-tokens` to 8192) and repeat the capped run to remove truncation as a variable; (2) the unconstrained run at 524K so GS and STWD are read whole. If (2) is clean and wall clock is tolerable, this is the model for documents beyond 131K; if the thinking cannot be tamed, gpt-oss-120b stays the default.

## Open questions

- Tokenizer size on this corpus relative to gpt-oss (drives the capped-run fallback count).
- Whether the ggml-org Q4_K quant costs accuracy versus Unsloth's UD-Q4_K_XL (83.8 GB).
- Reasoning toggle: does the chat template honour a no-think mode, and does extraction need thinking at all?

## Changelog

- 2026-09-15 — registered with a 524K default window and the `--max-input-tokens` comparison plan; card created.
- 2026-09-15 — first Spark run, capped at 131K: 93.4%, 6 truncations at 4096; card updated with results and the two follow-up runs.
