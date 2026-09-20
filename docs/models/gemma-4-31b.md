# gemma-4-31b

**Status:** measured (Spark, both twins, 2026-09-19). **Google's dense mid-size model, the
like-for-like rival to qwen3.8-27b. Thinking on: 117/121 (96.7%), perfect on every filing that fits
the window, losing only Goldman's BM25 chunks. Thinking off: 115/121 (95.0%) in half the wall
clock. Slowest dense model on the Spark at 7.9 t/s, and the 262K vocabulary made filings cost
more tokens, not fewer.**

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

## Serving configuration used

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
| Spark CUDA, 131K, thinking on | extract-full | 96.7% (117/121) | 0.93–0.99 | 93 full / 28 fallback (NEE + STWD Item 8: 31/31; GS BM25 chunks: 4/8) | 0 | 7.9 t/s | 367 t/s; 202–356 s TTFT on 80K+ filings | 130 min | [`20260919T174420Z-extract-full`](../../state/evals/sec/gemma-4-31b-spark-cuda/20260919T174420Z-extract-full/) |
| Spark CUDA, 131K, thinking off (`-nothink`) | extract-full | 95.0% (115/121) | 0.91–0.98 | 93 full / 28 fallback (Item 8: 31/31; GS chunks: 4/8) | 0 | 7.8 t/s | 355 t/s; 209–364 s | 61 min | [`20260919T202304Z-extract-full`](../../state/evals/sec/gemma-4-31b-nothink-spark-cuda/20260919T202304Z-extract-full/) |

Both runs on the pin `82d6bb284d1f`, `n_ctx_per_slot` 131,072, temperature 0, seed 42,
`max_tokens` 4096. Report: [`eval-report-2026-09-spark-cuda-gemma-laguna.md`](../eval-report-2026-09-spark-cuda-gemma-laguna.md).

## Strengths (measured)

- 104 of 104 on the ten filings that fit the window with thinking on, the first local model to do
  that alongside the Spark Qwen run; 102 of 104 with thinking off (MSFT shares came back "unknown",
  PLTR net income landed 0.6% high, just outside tolerance).
- Perfect on the Item 8 section fallback: 31 of 31 across NEE and STWD in both runs.
- Zero truncations in either run. Thinking is short and disciplined: 32.7K reasoning tokens over the
  suite, median 202 per question, against Nemotron's 131K and Qwen3.5-122B's 219K.
- Prompt-cache reuse is normal: 86.5% of prompt tokens served from cache, warm re-prefill gap a
  median 86 tokens. The predicted one-ubatch (~2,028-token) SWA re-prefill did not happen on this pin.
- 17.65 GB of weights; serves beside gpt-oss-120b on the Spark and fits the RTX 5090.

## Weaknesses (measured)

- 7.9 t/s decode, below the ~10 t/s the bandwidth rule predicted and below qwen3.8-27b's 9.8; the
  slowest dense model on the Spark. Cold prefill 355–367 t/s is a quarter of gpt-oss-120b's 1,354.
- The 262,144-token vocabulary did not shrink the filings. GS is 271,429 tokens here against 242,026
  under gpt-oss (+12%), NEE 136,728 against 124,130 (+10%). NEE therefore left full-document mode,
  and GS's Item 8 no longer fit the budget, so Goldman dropped to BM25 chunks: 4 of 8 in both runs,
  three "unknown" and one equity figure 1% off.
- Thinking on cost 69 minutes (130 vs 61) for two more correct answers; the difference is the
  32.7K reasoning tokens at 7.9 t/s almost exactly. One seed; the intervals overlap.
- Vision tower shipped but unused; the MTP drafter is untested.

## When to use / when not to

- The second dense candidate after qwen3.8-27b, and the more accurate of the two on filings that
  fit (104/104 vs Qwen's 104/104 at 131K on the Spark; both read every fitting filing right).
  Not a speed option: 130 minutes per suite with thinking, 61 without, against 26 for gpt-oss-120b.
- Reach for it when a document fits 131K and accuracy on that document matters more than latency;
  do not reach for it on Goldman-sized filings at 131K, where the fatter tokenizer forces chunks.

## Open questions

- Native 262K window on the Spark, the run that recovered Goldman for Qwen: does GS's Item 8 fit
  and do the four chunk misses close?
- Thinking on vs off on the SWE tier 1 suite; whether the two-question thinking gain repeats on a
  second seed.
- The MTP drafter's acceptance rate on 10-K text (`mtp-gemma-4-31B-it.gguf`, 0.28 GB), once the
  parked speed plan resumes.
- "KV self size" at 131K was not captured in the run record; read it from the server log.

## Changelog

- 2026-09-20 — both Spark runs recorded (thinking on 117/121, off 115/121); strengths and
  weaknesses rewritten from measurements; status measured.
- 2026-09-19 — registered on both platforms with a thinking-off twin; card created.
