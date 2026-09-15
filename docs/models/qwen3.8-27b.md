# qwen3.8-27b

**Status:** measured (Spark CUDA). **Careful reader, slow on this hardware; candidate for reasoning tasks, untested there.**

## Identity

| | |
| --- | --- |
| Family / vendor | Alibaba Qwen 3.8 |
| Architecture | Dense 27B, vision-language checkpoint served text-only; llama.cpp arch `qwen35`; reasoning via hidden thinking |
| Checkpoint served | `unsloth/Qwen3.8-27B-GGUF`, quant `UD-Q4_K_XL` (17.6 GB) |
| Native context | 262,144 (served at 131,072 to match the gpt-oss runs) |
| License | Apache-2.0 (verify on the model card) |
| Registry entry | `models.toml` `[models."qwen3.8-27b"]`, port 8084 |

## Serving configuration used

| Knob | Spark |
| --- | --- |
| llama.cpp | pinned `82d6bb284d1f`, CUDA `121a-real` |
| ctx_size / n_parallel | 131072 / default |
| Batch / ubatch | 2048 / 2048 |
| Flash attention / KV type | on / f16 |
| Extra flags | `--jinja`, `n_gpu_layers = 999` |

Eval decoding: temperature 0, seed 42, `max_tokens` 4096. Tokenizer is 10–15% fatter than
gpt-oss's on the same filings (AAPL 51.4K vs 45.6K tokens; XOM 122K vs 106K).

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| **Spark** | extract-full, 10-K | **95.9% (116/121)** | 0.92–0.99 | 93 full, 20 Item-8 section, **8 BM25 chunks** (GS) | 1 | 9.8 t/s | 70–200 s per filing (cold TTFT p50 121 s) | 125 min | `state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full` |

By mode: full 93/93, section 20/20, chunked 3/8. Reasoning ~230 hidden tokens per question
(43.6k total, about 2x gpt-oss-120b). Prompt-cache reuse 85%.

## Strengths (measured)

- Perfect wherever it saw the document: 113/113 across whole filings and Item 8 sections, including the CAT debt line gpt-oss-120b missed and the HD share count Terra missed.
- No scale errors on share counts.
- Small footprint (17.6 GB): can be served next to another model, or given a much larger context window.

## Weaknesses (measured)

- Dense architecture on a bandwidth-bound GPU: 9.8 t/s decode, about a third of gpt-oss-120b, and it reasons twice as long. Net 5x wall clock (125 vs 26 min) on the same suite.
- Fatter tokenizer plus the 131K window pushed NEE to the section fallback and GS all the way to keyword chunks, where it scored 3/8. All five misses are that one filing.
- One truncation at 4096 (GS share count, all budget spent on reasoning).
- Cold prefill 2.5x slower than gpt-oss-120b (610–740 vs 1,240–1,830 t/s).

## When to use / when not to

- **Use** when reading quality matters more than wall clock and the document fits: it did not miss a question it could see.
- **Use** at its native 262K context (`--ctx-size 262144`) if GS/STWD-sized documents are common; the chunked misses would very likely disappear.
- **Do not** use where throughput matters on this suite; gpt-oss-120b does the same job in a fifth of the time.
- **Try** on reasoning-heavy tasks (FinanceBench, coding tier) before drawing conclusions about the family; extraction cannot rank it.

## Open questions

- FinanceBench and the coding tier: does the longer reasoning buy anything on tasks that need it?
- Accuracy at 262K context, and the KV/prefill cost of doing so.
- The MoE sibling (Qwen3.5-122B class) would be the fair speed comparison to gpt-oss-120b.

## Changelog

- 2026-09-15 — first Spark run; report `docs/eval-report-2026-09-spark-cuda-qwen.md`; card created.
