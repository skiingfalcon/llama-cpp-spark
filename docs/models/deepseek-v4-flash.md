# deepseek-v4-flash

**Status:** planned (registered 2026-09-19, Spark only, no run yet). **The only DeepSeek V4 that
fits 128 GB, and only at 2 bits per weight: a 1M-context, ~13B-active mixture-of-experts model
served from a 96.8 GB UD-Q2_K_XL. Two questions: does 2-bit quality hold on extraction, and does
the pinned llama.cpp reuse the filing prefix between questions?**

## Identity

| | |
| --- | --- |
| Family / vendor | DeepSeek-AI, DeepSeek-V4-Flash-0731 (open weights, 2026-07-31; supersedes the April 2026 V4-Flash preview) |
| Architecture | Mixture-of-experts, 43 layers, hidden 4096, 256 routed experts + 1 shared, 6 active per token, ~13B active parameters; 64 attention heads with head dim 512, per-layer compressed sparse attention (`compress_ratios` 4x / 128x, `sliding_window` 128) with a lightning indexer (`index_topk` 512, 64 index heads); YaRN x16 from 64K to 1M; llama.cpp arch `deepseek4`; reasoning via hidden thinking with `reasoning_effort` low / high / max |
| Checkpoint served | `unsloth/DeepSeek-V4-Flash-0731-GGUF`, quant `UD-Q2_K_XL` (3 shards in `UD-Q2_K_XL/`, 96.8 GB). `UD-IQ3_XXS` (104.2 GB) is the upgrade if memory allows; `UD-Q4_K_XL` (155 GB) does not fit. `ggml-org/DeepSeek-V4-Flash-0731-GGUF` has `Q2_K_S` (98.6 GB, no imatrix) as an alternative source |
| Native context | 1,048,576 |
| License | MIT (verify on the model card) |
| Registry entry | `models.toml` `[models."deepseek-v4-flash"]` and `[models."deepseek-v4-flash-nothink"]`, port 8092; not in `models.halo.toml` (over the 96 GB VGM cap and the 5090) |

Not fitting: DeepSeek-V4.1-Flash (552B backbone, 2026-09-10) and DeepSeek-V4-Pro (0813) at any
published quant.

## Serving configuration (planned)

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (arch `deepseek4` merged 2026-06-29; V4 Flash 0731 template 2026-08-03; both in the pin) | not served |
| Backend | cuda | |
| ctx_size / n_parallel | 131072 / 1 (the 1M window is not attempted beside 97 GB of weights) | |
| Batch / ubatch | 2048 / 2048 | |
| Flash attention / KV type | on / f16 (the pin forces matching K/V types and FA when V is quantised, PR #25871); vendor quotes ~3.5 KB/token in its own engine; read llama.cpp's "KV self size" from the log | |
| Extra flags | `--jinja`; `-nothink` entry adds `--reasoning off`; `LOCAL_LLM_HEALTH_TIMEOUT_S=1500` for the load; drop the page cache first | |
| Sampling (serving only) | model defaults (vendor: temp 1.0, top_p 0.95 agentic / 1.0 otherwise, thinking mode) | |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. The vendor recommends a maximum
output length of 384K tokens at high and max reasoning effort, so **the `-nothink` twin is the
primary row**; the thinking twin is run second and expected to truncate.

Plan, all through `scripts/spark-new-models-smoke.sh`: (1) 22-item gate (AAPL, XOM) on both twins
checking the served chat template, `reasoning_content` separation and the warm-row prompt-cache
ratio; (2) if the gate passes, `extract-full` at 131K, nothink first; (3) `reasoning_effort = low`
with the thinking twin as a later experiment. FinanceBench is not planned: the judge
(`gpt-oss-120b`, 63 GB) cannot be co-served beside 97 GB of weights, so free-text items would be
unscored.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Spark CUDA, 131K, thinking off (`-nothink`) | extract-full | 98.3% (119/121) | 0.96–1.00 | 104 full (102/104) / 17 section (17/17) | 0 | 16.0 t/s | 422 t/s; 190–306 s TTFT on 80K+ filings | 51 min | [`20260919T121223Z-extract-full`](../../state/evals/sec/deepseek-v4-flash-nothink-spark-cuda/20260919T121223Z-extract-full/) |

Misses: PLTR Liabilities and PLTR CommonStockSharesOutstanding, both exactly 1,000x too small.
Prompt-cache reuse 87.9% (9,302,499 of 10,577,083 prompt tokens), so the PR #29008 risk below did
not materialise on this pin. Email: [`emails/2026-09-19-deepseek-v4-flash-nothink.md`](../emails/2026-09-19-deepseek-v4-flash-nothink.md).
The thinking twin has not been run.

## Strengths (expected, unmeasured)

- ~13B active parameters: the bandwidth rule predicts roughly 20 t/s on GB10, in the Nemotron /
  Qwen3.5-122B band, from a model the vendor places with frontier systems on agentic benchmarks.
- 1M native context and a compressed KV cache: the whole 10-K set would fit in one window if
  memory allowed a larger `ctx_size`; on the Spark 262K is the practical ceiling to test.
- MIT licence.

## Weaknesses (expected, unmeasured)

- 2 bits per weight is a quantisation no other row here uses; unsloth's dynamic quants keep
  attention and shared experts higher, but the effect on numeric extraction is unknown.
- 96.8 GB of weights: nothing else can be served alongside it, including the FinanceBench judge.
- Thinking mode is sized for a 384K completion budget; ours is 4,096.
- Pin risks: DeepSeek publishes no Jinja template (the pinned llama.cpp ships its own for 0731),
  and upstream PR #29008 (2026-09-17, after our pin) reports that the V4 parser did not expose
  user-message delimiters, so the prompt cache was barely reused. If that applies here every
  question re-prefills the filing and the suite takes many hours; the gate stops that.

## When to use / when not to

- Not yet. If the gate passes and the nothink row lands at or above the dense Qwen (95.9%), it is
  the long-document candidate on the Spark and the case for a Linux box with more memory. If
  2-bit quality fails on numbers, or the cache does not reuse, the DeepSeek question on 128 GB is
  closed until a smaller V4 or a newer pin.

## Open questions

- Which chat template serves (GGUF-embedded or the pinned `deepseek-ai-DeepSeek-V4-Flash-0731.jinja`) and whether `--reasoning off` maps to `enable_thinking=false` as the pinned template suggests.
- Warm-row `cached_prompt_tokens / prompt_tokens` on the pin; whether a pin bump past PR #29008 is needed.
- "KV self size" at 131K; whether 262K fits.
- `UD-IQ3_XXS` (104.2 GB) headroom on the box once the KV size is known.
- `reasoning_effort = low` with the thinking twin: does a small thinking budget beat thinking off?

## Changelog

- 2026-09-20 — nothink Spark run recorded (119/121, 16 t/s, 51 min, cache reuse 88%).
- 2026-09-19 — registered on the Spark only, with a thinking-off twin as the primary row; card created.
