# SEC 10-K extraction — AMD Strix Halo vs NVIDIA DGX Spark (September 2026)

Same model (`gpt-oss-120b` MXFP4), same 121-question `extract-full` suite as the
[Spark re-run](eval-report-2026-09-rerun.md). The AMD Strix Halo box was locked down
(no Python / `uv` / compiler), so the Halo numbers come from
[`scripts/lmstudio.mjs`](../scripts/lmstudio.mjs) against LM Studio's OpenAI-compatible
server instead of `local-llm serve`.

| Stack | Engine | Score | Wall clock |
| --- | --- | ---: | ---: |
| **NVIDIA DGX Spark (CUDA)** | llama-server, pinned llama.cpp | **119/121 (98.3%)** | **26 min** |
| **AMD Strix Halo (Vulkan)** | LM Studio bundled llama.cpp 2.37 | **117/121 (96.7%)** | **77 min** |
| **AMD Strix Halo (ROCm)** | LM Studio bundled llama.cpp 2.37 | **115/121 (95.0%)** | **72 min** |

Accuracy is within ~2–3 points across all three stacks. Wall clock on the AMD box is
~3× the NVIDIA Spark — driven almost entirely by cold prefill of ~100K-token filings
(4–8 min each), not by decode.

![Inference stacks: NVIDIA DGX Spark vs AMD Strix Halo (Vulkan / ROCm)](inference-stack-spark-vs-halo.png)

## Setup

| | NVIDIA DGX Spark (CUDA) | AMD Strix Halo (Vulkan) | AMD Strix Halo (ROCm) |
| --- | --- | --- | --- |
| Hardware | GB10 Blackwell, 128 GB | Radeon 8060S (RDNA 3.5), 128 GB UMA | same |
| OS | DGX OS (Ubuntu, ARM64) | Windows 11 x86-64 | same |
| Server | `llama-server` via `local-llm` | LM Studio local server | same |
| Backend | ggml-cuda / CUDA + cuBLAS | ggml-vulkan / Vulkan | ggml-hip / ROCm HIP |
| Runtime | pinned `LLAMA_CPP_RELEASE` | LM Studio `llama.cpp-win-x86_64-vulkan-avx2` **2.37.0** | LM Studio `llama.cpp-win-x86_64-amd-rocm-avx2` **2.37.0** |
| Context | 131,072 | 131,072 | 131,072 |
| Driver | CUDA | Adrenalin (Vulkan) | Adrenalin + Windows ROCm |

**AMD LM Studio load settings** (both backends): GPU offload 36/36 layers (~65 GB),
evaluation/physical batch 512, unified KV cache on, KV cache offloaded to GPU, keep model
in memory. Auto-update of runtime packs was off for these runs.

**Task / decoding** (all three): `extract-full`, 12 companies × latest 10-K, 11 XBRL tags,
temperature 0, seed 42, `max_tokens` 4096, 0.5% relative tolerance. Halo runs used
`node scripts/lmstudio.mjs run --forms 10-K` (same prompts, ground truth, and
full → section → BM25 degradation as the Python harness).

**Artifacts**

- NVIDIA DGX Spark (CUDA): [`state/evals/sec/gpt-oss-120b/20260912T132250Z-extract-full/`](../state/evals/sec/gpt-oss-120b/20260912T132250Z-extract-full/)
- AMD Strix Halo (Vulkan): [`state/evals/gpt-oss-120b/20260913T030643Z-extract-full/`](../state/evals/gpt-oss-120b/20260913T030643Z-extract-full/)
- AMD Strix Halo (ROCm): [`state/evals/gpt-oss-120b/20260913T044559Z-extract-full/`](../state/evals/gpt-oss-120b/20260913T044559Z-extract-full/)

## Results

### Accuracy by context mode

| Slice | n | NVIDIA DGX Spark (CUDA) | AMD Strix Halo (Vulkan) | AMD Strix Halo (ROCm) |
| --- | ---: | ---: | ---: | ---: |
| Full document | 104 | 103/104 (99.0%) | 103/104 (99.0%) | 101/104 (97.1%) |
| GS/STWD via Item 8 fallback | 17 | 16/17 (94.1%) | 14/17 (82.4%) | 14/17 (82.4%) |
| **All questions** | **121** | **119/121 (98.3%)** | **117/121 (96.7%)** | **115/121 (95.0%)** |

On full-document items, NVIDIA Spark and AMD Vulkan tie at 103/104. The AMD score gap is
almost entirely in the Item 8 fallback slice (GS/STWD) plus a few which-line picks that
flip between builds.

### Per-tag accuracy

| Tag | NVIDIA DGX Spark (CUDA) | AMD Strix Halo (Vulkan) | AMD Strix Halo (ROCm) |
| --- | ---: | ---: | ---: |
| Assets | 12/12 | 12/12 | 12/12 |
| CashAndCashEquivalentsAtCarryingValue | 11/11 | 11/11 | 11/11 |
| CommonStockSharesOutstanding | 10/10 | **9/10** | **9/10** |
| EarningsPerShareDiluted | 12/12 | **11/12** | **11/12** |
| Liabilities | 11/11 | 11/11 | 11/11 |
| LongTermDebtNoncurrent | **8/9** | 9/9 | **8/9** |
| NetCashProvidedByUsedInOperatingActivities | **11/12** | 12/12 | 12/12 |
| NetIncomeLoss | 12/12 | **11/12** | **10/12** |
| OperatingIncomeLoss | 9/9 | 9/9 | 9/9 |
| Revenues | 11/11 | 11/11 | 11/11 |
| StockholdersEquity | 12/12 | **11/12** | **11/12** |

### Per-company accuracy

| Ticker | NVIDIA DGX Spark (CUDA) | AMD Strix Halo (Vulkan) | AMD Strix Halo (ROCm) |
| --- | ---: | ---: | ---: |
| AAPL | 11/11 | 11/11 | 11/11 |
| CAT | **10/11** | 11/11 | 11/11 |
| GS | **7/8** | **6/8** | **6/8** |
| HD | 11/11 | 11/11 | 11/11 |
| INTU | 11/11 | 11/11 | 11/11 |
| MSFT | 11/11 | 11/11 | 11/11 |
| NEE | 11/11 | 11/11 | **10/11** |
| PG | 9/9 | 9/9 | **8/9** |
| PLTR | 10/10 | **9/10** | **9/10** |
| STWD | 9/9 | **8/9** | **8/9** |
| WMT | 9/9 | 9/9 | 9/9 |
| XOM | 10/10 | 10/10 | 10/10 |

### Every miss, itemized

**Shared by both AMD backends (3):**

- **GS `NetIncomeLoss`** (Item 8): answered 16.3B vs expected 17.176B. Close but outside 0.5%.
- **GS `EarningsPerShareDiluted`** (Item 8): AMD Vulkan replied `unknown`; AMD ROCm answered
  51.95 vs expected 51.32. NVIDIA Spark got both.
- **PLTR `CommonStockSharesOutstanding`** (full): answered 2,391,192 vs expected
  2,391,192,000 — classic `off_by_scale` (thousands in the table, not multiplied out). Both
  AMD backends; NVIDIA Spark got it.

**AMD Strix Halo (Vulkan) only (1):**

- **STWD `StockholdersEquity`** (Item 8): 7.125B vs expected 6.796B. AMD ROCm got this one.

**AMD Strix Halo (ROCm) only (3):**

- **PG `LongTermDebtNoncurrent`**: 29.3B vs expected 22.8B (which-line / total-debt pick).
- **NEE `StockholdersEquity`**: 66.5B vs expected 54.6B (likely including NCI or a broader
  equity total).
- **STWD `NetIncomeLoss`** (Item 8): 655M vs expected 412M.

**NVIDIA DGX Spark (CUDA) misses that both AMD backends got right:**

- **GS operating cash flow**: Spark answered +17.0B (parent-only condensed statement in the
  notes); both AMD backends answered −45.154B (consolidated). Documented in the
  [re-run report](eval-report-2026-09-rerun.md#every-remaining-miss-itemized).
- **CAT `LongTermDebtNoncurrent`**: Spark answered 50.7B (incl. Financial Products);
  both AMD backends answered 30.7B (machinery-only, matching XBRL).

Same weights, same prompts, different llama.cpp build/backend — the borderline which-line
items flip. Treat the ~2-point score spread as run-to-run / build-to-build variance at the
accuracy ceiling, **not** evidence that one GPU vendor reads 10-Ks better than the other.

### Latency and throughput

| Metric | NVIDIA DGX Spark (CUDA) | AMD Strix Halo (Vulkan) | AMD Strix Halo (ROCm) |
| --- | ---: | ---: | ---: |
| Wall clock | 26 min | 77 min | 72 min |
| Warm TTFT p50 | 0.57 s | 1.38 s | 1.19 s |
| Total p50 / p95 | 5.8 s / 62.9 s | 5.1 s / 300 s | 7.4 s / 238 s |
| Cold prefill (12 filings) TTFT p50 | ~40–60 s (suite p95) | **271 s** | **257 s** |
| Cold prefill tok/s p50 | — | ~297 | ~315 |
| Decode tok/s p50 | 30.3 | **28.8** | 19.4 |
| Prompt tokens (suite) | 10.52 M | 9.98 M | 9.98 M |

- **Cold prefill dominates AMD wall clock.** The 12 first-of-filing requests take 2–8 minutes
  each (~90–115K tokens at ~300 tok/s). Warm follow-ups on the same filing are ~1.2–1.4 s
  TTFT — usable, but the Spark stays ~2× snappier on warm TTFT and much faster on cold.
- **AMD Vulkan decodes ~50% faster than AMD ROCm** (28.8 vs 19.4 tok/s); ROCm edges Vulkan
  on cold prefill. Net wall clock favors ROCm slightly (72 vs 77 min) because prefill is the
  long pole; **accuracy and decode favor Vulkan**.
- Ignore the `run.json` `prompt_tps_p50` figures (58K / 69K). Those are inflated by
  cache-warm requests that still report full `prompt_tokens` against a ~1 s TTFT. Use the
  cold-prefill tok/s above.

## Pros and cons

### NVIDIA DGX Spark (CUDA) + `local-llm`

**Pros**

- Highest score (98.3%); pinned, recorded llama.cpp build; full harness (`/tokenize`,
  `/props`, cache accounting).
- ~3× faster wall clock on this suite; cold prefill measured in tens of seconds, not minutes.
- Reproducible serving flags (`models.toml` / `models.halo.toml` parity path).

**Cons**

- Requires the full Python stack and a CUDA build — not an option on locked-down Windows
  hosts.

### AMD Strix Halo via LM Studio (`scripts/lmstudio.mjs`)

**Pros**

- **96–97% accuracy without Python** — good enough as an on-box accuracy signal when only
  LM Studio is installable.
- Same model weights and essentially the same task definition as the Spark re-run.
- AMD Vulkan slightly ahead of AMD ROCm on this suite (score + decode).

**Cons**

- ~3× wall clock vs the Spark; cold 100K-token prefills are multi-minute.
- Unpinned, auto-updatable LM Studio runtime — not a hardware bake-off against the Spark's
  recorded build.
- No `/tokenize` (chars/4.6 estimates), no cached-token column, TTFT includes full prefill
  wait because LM Studio withholds response headers until the first token.

## Caveats

1. **Vendor labels matter more than backend names.** "Vulkan" and "ROCm" here always mean
   **AMD Strix Halo** under LM Studio; "CUDA" means **NVIDIA DGX Spark** under `llama-server`.
2. **Not an apples-to-apples engine comparison.** Spark uses a pinned llama.cpp; Halo uses
   LM Studio 2.37.0's bundled builds. Prefer `local-llm serve --backend vulkan|hip` on an
   unlocked Halo box for a true backend bake-off.
3. **Token accounting differs.** Spark sums tokenizer-exact prompt tokens (10.52 M) with 90%
   cache hits; Halo estimates / reports ~9.98 M with `cached_prompt_tokens: null`.
4. **Single run per AMD backend**; local decoding is deterministic per engine, but the two
   engines disagree on four items.
5. Config hash matches between the two AMD runs (`58b2946660ca`); Spark runs use the Python
   harness hash and are expected to differ.

## Reproduce

```powershell
# On the AMD Strix Halo box (locked-down path)
$env:EDGAR_UA = "local-llm you@example.com"
# Load gpt-oss-120b in LM Studio: ctx 131072, GPU offload max, batch 512, KV on GPU
# Runtime → pick Vulkan or ROCm llama.cpp (Windows) v2.37.0; start the local server

node scripts/lmstudio.mjs fetch
node scripts/lmstudio.mjs run --forms 10-K
node scripts/lmstudio.mjs report
```

On an unlocked Halo box prefer the native path in
[Windows / AMD Strix Halo](../README.md#windows--amd-strix-halo). For the Spark numbers,
see [eval-report-2026-09-rerun.md](eval-report-2026-09-rerun.md).

## Bottom line for the team

- **`gpt-oss-120b` on AMD Strix Halo via LM Studio is a credible on-prem extraction path
  when the box is locked down:** 96.7% (Vulkan) / 95.0% (ROCm) vs 98.3% on the NVIDIA Spark,
  same 121-question suite.
- **Prefer AMD Vulkan over AMD ROCm for this workload** — higher score and ~50% faster
  decode; ROCm only wins slightly on cold prefill / total wall clock.
- **The NVIDIA Spark remains the benchmark reference** for speed, pinned builds, and full
  harness telemetry. Use Halo + LM Studio for an accuracy signal you can collect without
  Python; do not treat the multi-minute cold prefills as competitive with the Spark.
- Borderline which-line misses flip across builds (Spark's GS cash-flow and CAT debt misses
  are correct on both AMD runs; AMD introduces different misses). Do not over-interpret a
  2-point gap as a GPU-vendor quality difference.
