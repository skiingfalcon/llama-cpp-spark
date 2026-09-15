# SEC 10-K extraction — Qwen3.8-27B on an RTX 5090 (September 2026)

Same 121-question `extract-full` suite, same 12 filings, same scoring (`2026-09-13.1`, 0.5%
relative tolerance) as the [Spark Qwen report](eval-report-2026-09-spark-cuda-qwen.md). This
report adds two rows for **`qwen3.8-27b`** (Unsloth `UD-Q4_K_XL`) on a single **NVIDIA GeForce
RTX 5090 (32 GB)** under Windows 11, CUDA llama.cpp release `b10919`:

- **Run 1** replicates the Spark configuration at 131,072 context.
- **Run 2** serves the model's native 262,144 context with an 8-bit KV cache, the experiment the
  Spark report proposed to recover the Goldman Sachs questions.

| Stack | Accuracy (121 Qs) | When the filing fits (of 104 Qs) | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| **RTX 5090, Qwen3.8-27B (dense), 262K ctx + q8_0 KV** | **98.3%** | **102/104** | **55 t/s** | **40–70 s** | **24 min** | **$0 marginal** |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| **RTX 5090, Qwen3.8-27B (dense), 131K ctx** | **94.2%** | **102/104** | **60 t/s** | **38–54 s** | **23 min** | **$0 marginal** |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | — | 40–60 s | 19 min | $0 marginal |

Rows other than the two RTX 5090 rows are copied from the September 15 summary and the reports
it links. Single run per row; the bootstrap 95% intervals in
[`state/evals/report-sec.md`](../state/evals/report-sec.md) overlap for the top four local rows
(RTX 5090 262K: 0.96–1.00; Spark 120b: 0.96–1.00; Spark Qwen: 0.92–0.99), so read gaps of one
or two questions as noise.

"When the filing fits" is scored on the 10 filings gpt-oss read whole (104 questions), so the
column is comparable across tokenizers. "Cold prefill" is time to first token on the first
question of each 90–140K-token filing (INTU, PLTR, CAT, XOM, plus GS Item 8 and NEE in run 2);
the largest whole document in run 2 (STWD, 202K tokens) took 132 s cold.

## Setup

- **Task / corpus / tags / scoring:** identical to the Spark Qwen report. Corpus fetched from
  EDGAR on 2026-09-15; every filing, accession and Qwen token count matches the Spark run
  (e.g. AAPL 51,383, XOM 122,370, GS 273,285 tokens).
- **Decoding:** `temperature=0`, `seed=42`, `max_tokens=4096`, `--parallel 1` in the harness
  (llama-server auto-selected 4 slots with a unified KV cache, same as the Spark run).
- **Hardware / build:** NVIDIA GeForce RTX 5090, 32 GB, driver 616.92, Windows 11. llama.cpp
  release **`b10919`** (`d3146f2b5`), official `llama-b10919-bin-win-cuda-13.3-x64.zip` plus
  its `cudart` zip (the release has no CUDA 13.4 x64 zip). Weights: the same 17.6 GB
  `Qwen3.8-27B-UD-Q4_K_XL.gguf`. `--n-gpu-layers 999 --flash-attn on --batch-size 2048
  --ubatch-size 2048 --jinja`.
  - Run 1: `--ctx-size 131072`, f16 KV. ~26 GB VRAM in use by the server.
  - Run 2: `--ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0`. ~28 GB VRAM.
- **CLI note:** this box runs the `halo` (Windows) platform code with `LOCAL_LLM_LLAMA_BIN_DIR`
  pointing at the CUDA zip, so both runs are labelled `halo-vulkan` in `run.json` and the GPU
  field shows the machine's AMD iGPU. `server.build_info` (`b10919-d3146f2b5`) and
  `server.n_ctx_per_slot` are the authoritative fields. In run 2, `runtime.ctx_size` echoes the
  registry's 131072; `server.n_ctx_per_slot` = 262144 is what was served.
- **Artifacts:**
  - Run 1: [`state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T145801Z-extract-full/`](../state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T145801Z-extract-full/)
  - Run 2: [`state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T152312Z-extract-full/`](../state/evals/sec/qwen3.8-27b-halo-vulkan/20260915T152312Z-extract-full/)
  - Spark Qwen (unchanged): [`state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full/`](../state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full/)

## Results

### Accuracy by context mode

| Slice | Spark Qwen (131K) | **RTX 5090 run 1 (131K)** | **RTX 5090 run 2 (262K)** |
| --- | ---: | ---: | ---: |
| Full document | 93/93 (100%) | **91/93 (97.8%)** | **111/113 (98.2%)** |
| Item 8 section fallback | 20/20 (100%) | **20/20 (100%)** | **8/8 (100%) — GS only** |
| BM25 chunk fallback | 3/8 (37.5%) | **3/8 (37.5%)** | **— (no filing fell to chunks)** |
| All questions | 116/121 (95.9%) | **114/121 (94.2%)** | **119/121 (98.3%)** |

Three readings:

- **The Goldman misses were a context cap, as the Spark report inferred.** At 262K the harness
  kept GS on the Item 8 path: the section is **131,589 Qwen tokens**, 517 over the 131,072
  window, which is exactly why it fell to chunks at 131K. In section mode GS scored **8/8**,
  including the four values the chunked path got wrong and the operating-cash-flow sign error
  shared by all three local models.
- **NEE and STWD moved from Item 8 to whole-document** at 262K (137.5K and 202.4K tokens) and
  stayed perfect, so the larger window cost nothing on the filings that already fit.
- **Run 1 reproduced the Spark run on GS exactly**: same chunked mode, same 3/8, same wrong
  values (17.007B operating cash flow, 425.7B liabilities, `unknown` assets/equity).

### Per-tag accuracy

| Tag | Spark Qwen | **RTX 5090 run 1** | **RTX 5090 run 2** |
| --- | ---: | ---: | ---: |
| Assets | 11/12 | **11/12** | **12/12** |
| CashAndCashEquivalentsAtCarryingValue | 11/11 | **11/11** | **11/11** |
| CommonStockSharesOutstanding | 9/10 | **7/10** | **8/10** |
| EarningsPerShareDiluted | 12/12 | **12/12** | **12/12** |
| Liabilities | 10/11 | **10/11** | **11/11** |
| LongTermDebtNoncurrent | 9/9 | **9/9** | **9/9** |
| NetCashProvidedByUsedInOperatingActivities | 11/12 | **11/12** | **12/12** |
| NetIncomeLoss | 12/12 | **12/12** | **12/12** |
| OperatingIncomeLoss | 9/9 | **9/9** | **9/9** |
| Revenues | 11/11 | **11/11** | **11/11** |
| StockholdersEquity | 11/12 | **11/12** | **12/12** |

Every tag except common shares outstanding is perfect in run 2. No `off_by_scale` errors in any
run.

### Per-company accuracy

| Ticker | Spark Qwen | **RTX 5090 run 1** | **RTX 5090 run 2** | Mode (Spark / run 1 / run 2) |
| --- | ---: | ---: | ---: | --- |
| AAPL | 11/11 | **11/11** | **11/11** | full / full / full |
| CAT | 11/11 | **11/11** | **11/11** | full / full / full |
| GS | 3/8 | **3/8** | **8/8** | chunked / chunked / **section** |
| HD | 11/11 | **10/11** | **10/11** | full / full / full |
| INTU | 11/11 | **11/11** | **11/11** | full / full / full |
| MSFT | 11/11 | **11/11** | **10/11** | full / full / full |
| NEE | 11/11 | **11/11** | **11/11** | section / section / **full** |
| PG | 9/9 | **9/9** | **9/9** | full / full / full |
| PLTR | 10/10 | **10/10** | **10/10** | full / full / full |
| STWD | 9/9 | **9/9** | **9/9** | section / section / **full** |
| WMT | 9/9 | **9/9** | **9/9** | full / full / full |
| XOM | 10/10 | **9/10** | **10/10** | full / full / full |

### Every miss, itemized

**Run 1 (131K), 7 misses.** The five GS chunked misses are the Spark report's five, with the same
answers, except that the share count came back `unknown` instead of truncating. Two new:

- **XOM common shares outstanding:** `unknown` vs 4.179B (whole document in context).
- **HD common shares outstanding:** empty answer, `finish_reason=length` — the run's one
  truncation, all 4,096 tokens spent on reasoning.

**Run 2 (262K), 2 misses.** Both are the cover-page share count, both with the whole filing in
context, both `unknown`:

- **MSFT common shares outstanding:** `unknown` vs 7.427B.
- **HD common shares outstanding:** `unknown` vs 996M.

Across the three runs the model answered `CommonStockSharesOutstanding` wrong for GS, XOM, HD and
MSFT in different combinations and never with a wrong number, only `unknown` or a blown budget.
Everything else the model has seen in context it has answered correctly, three runs running.

## Why the 5090 is ~5× faster than the Spark on the same model

| Phase | Spark Qwen | **RTX 5090 run 1** | **RTX 5090 run 2** | Spark / 5090 |
| --- | ---: | ---: | ---: | ---: |
| Prefill (time to first token, summed) | 47.7 min | **10.6 min** | **12.7 min** | ~4× |
| Decode (completion) | 77.5 min | **11.9 min** | **11.6 min** | ~6.5× |
| **Wall clock** | **125.2 min** | **22.5 min** | **24.3 min** | **~5.3×** |

Wall clock equals summed request time to within 0.1 min in all three runs (serial harness, GPU at
96–100% throughout).

1. **Bandwidth.** The Spark report attributed Qwen's 10 t/s to a dense model reading all ~17 GB
   of weights per token on a bandwidth-bound GPU. The RTX 5090 has roughly 6× the GB10's memory
   bandwidth and decodes the same checkpoint at **55–60 t/s**, i.e. about twice gpt-oss-120b's
   30 t/s on the Spark. Same architectural floor, different ceiling.
2. **Prefill.** Cold prompt throughput is 2,100–3,100 t/s on the 5090 vs 610–740 t/s on the
   Spark (~4×). A ~120K filing prefills in 54–58 s instead of 203 s.
3. **Reasoning is unchanged.** p50 hidden reasoning per question is 224–244 tokens across all
   three runs (Spark 227); totals 35.5k–43.6k. The model is not thinking less on the 5090, it is
   just emitting faster.
4. **Run 2 is ~8% slower than run 1** despite fewer reasoning tokens: prefill grows because NEE
   and STWD are now read whole (12.77M vs 10.37M prompt tokens) and the q8_0 cache costs a little
   decode throughput (55 vs 60 t/s).

### Latency and tokens

| Row | Wall clock | Request time | TTFT p50 | Cold TTFT p50 | Total p50 / p95 | Prompt tok | Cached | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spark Qwen (131K) | 125 min | 125 min | 8.6 s | 121 s | 36 s / 196 s | 10.37 M | 84% | 43.6 k |
| **RTX 5090 run 1 (131K)** | **22.5 min** | **22.4 min** | **1.27 s** | **30 s** | **5.6 s / 41 s** | **10.37 M** | **84%** | **41.9 k** |
| **RTX 5090 run 2 (262K)** | **24.3 min** | **24.2 min** | **1.44 s** | **45 s** | **6.2 s / 43 s** | **12.77 M** | **88%** | **35.5 k** |

## Pros and cons

### `qwen3.8-27b` on the RTX 5090 at 262K

**Pros**

- 98.3%, tying gpt-oss-120b on the Spark and one question behind hosted Terra, in 24 minutes.
- Goldman Sachs 8/8 on the Item 8 path; nothing fell to BM25 chunks.
- Fits a single 32 GB consumer card with ~3 GB to spare (17.6 GB weights + q8_0 KV for 262K).
- No scale errors, no truncations.

**Cons**

- The cover-page share count is the recurring miss (2 of 10 here, 3 of 10 in run 1) and is not
  stable run to run at temperature 0.
- 262K still does not hold GS whole (273K tokens); it only recovers the Item 8 path.
- KV at q8_0 is a quantisation the Spark runs did not use; its effect on the numbers is not
  isolated here (run 1 vs run 2 differ in context too).
- Runs are mislabelled `halo-vulkan` until the CLI grows a Windows/CUDA platform.

## Caveats

> **Review notes (2026-09-15).** Single run per configuration. Run 1 and run 2 differ in two
> variables (context window and KV cache type), so the GS recovery is attributable to context
> (the section is 131,589 tokens) but the two run-2 share-count misses cannot be attributed to
> either. The first run-1 attempt aborted after 11 questions on a CUDA illegal-memory-access
> during prefill that coincided with a GPU driver reset; the GPU was undervolted at the time, the
> undervolt was disabled, and the complete run reported here was a clean rerun. No
> `--no-document` memorisation control, same as every other row. Wall clocks exclude model load
> (~45 s) and the aborted attempt.

1. **Not a like-for-like hardware comparison with the Spark.** Different GPU class, driver, OS
   and llama.cpp build (`b10919` prebuilt CUDA 13.3 vs `82d6bb284d1f` source build); both are
   September 2026 builds of the same code base.
2. **`CommonStockSharesOutstanding` variance.** Four different filings have missed it across
   three runs of the same model at temperature 0. That is a prompt or document-layout effect
   worth a harness follow-up before it is read as model quality.
3. **Provenance fields.** Platform, backend and GPU name in `run.json` are wrong for this box
   (see Setup); build id and served context are right.

## Reproduce

```powershell
# Windows: CUDA zip for b10919 in C:\llama\b10919-cuda; .env holds
#   LOCAL_LLM_LLAMA_BIN_DIR=C:\llama\b10919-cuda
#   LOCAL_LLM_EDGAR_USER_AGENT="local-llm you@example.com"
# qwen3.8-27b registered in models.halo.toml (ctx 131072, port 8084, ubatch 2048, --jinja).
$env:PYTHONUTF8 = "1"                       # fetch writes UTF-8 filings; cp1252 crashes
uv run local-llm download qwen3.8-27b
uv run local-llm eval sec fetch

# Run 1
uv run local-llm serve qwen3.8-27b
uv run local-llm eval sec run qwen3.8-27b --task extract-full --forms 10-K
uv run local-llm stop

# Run 2: `serve ... -- --cache-type-k q8_0` does not reach llama-server on this CLI version
# (the flags are parsed as model names), so set the registry fields instead:
#   cache_type_k = "q8_0"
#   cache_type_v = "q8_0"
uv run local-llm serve qwen3.8-27b --ctx-size 262144
uv run local-llm eval sec run qwen3.8-27b --task extract-full --forms 10-K
uv run local-llm stop
uv run local-llm eval report --suite sec
```

## Bottom line for the team

- **The Spark report's hypothesis is confirmed:** the five Goldman misses were the 131K cap, not
  the model. At 262K Qwen scores **119/121 (98.3%)**, level with gpt-oss-120b.
- **Dense-model slowness was a Spark property, not a Qwen property.** On an RTX 5090 the same
  checkpoint runs the suite in **23–24 minutes**, in line with 120b on the Spark (26 min), with
  decode at 55–60 t/s.
- **Extraction still cannot separate the top stacks.** Terra, Spark 120b and 5090 Qwen are within
  two questions of each other on 121; the remaining local misses are one cover-page fact. This
  supports the proposal to move model selection to FinanceBench and the coding tier.
- **Harness follow-ups:** (a) a Windows/CUDA platform label so 5090 runs are not filed under
  `halo-vulkan`; (b) log the rejected Item 8 token count (GS is 131,589); (c) look at why
  `CommonStockSharesOutstanding` returns `unknown` with the cover page in context.
