# SEC 10-K extraction — Qwen3.8-27B on the Spark (September 2026)

Same 121-question `extract-full` suite as the
[Spark re-run](eval-report-2026-09-spark-cuda-rerun.md) (gpt-oss-20b / 120b / Terra). This
report adds **`qwen3.8-27b`** (Unsloth `UD-Q4_K_XL`, dense 27B, llama.cpp arch `qwen35`) on
the same NVIDIA DGX Spark / CUDA stack and the same 131,072 context. Nemotron-3-Super on this
suite is in [eval-report-2026-09-spark-cuda-nemotron.md](eval-report-2026-09-spark-cuda-nemotron.md).

| Model | Where it ran | Params (active / total) | Score | Wall clock |
| --- | --- | --- | ---: | ---: |
| `gpt-5.6-terra` | OpenAI API | hosted | **99.2% (120/121)** | 20 min† |
| `gpt-oss-120b` | Local Spark | ~5.1B / 117B MoE | **98.3% (119/121)** | **26 min** |
| **`qwen3.8-27b`** | **Local Spark** | **27B / 27B dense** | **95.9% (116/121)** | **125 min** |
| `gpt-oss-20b` | Local Spark | ~3.6B / 21B MoE | 87.6% (106/121) | 19 min |

† Terra wall clock includes HTTP 429 backoff; request time was 3.5 min. Local wall clock
equals request time (`--parallel 1`, no queueing).

Qwen sits between 120b and 20b on accuracy, and is **~5× slower than 120b** on the same GPU.
On every filing that actually fit in context it was perfect (93/93). The five misses are all
Goldman Sachs, after the Qwen tokenizer pushed Item 8 itself past 131K and the harness fell
through to BM25 chunks.

## Setup

- **Task / corpus / tags / scoring:** identical to the re-run (`scoring_version` `2026-09-13.1`,
  0.5% relative tolerance, 12 companies, latest 10-K, 11 XBRL tags).
- **Decoding:** `temperature=0`, `seed=42`, `max_tokens=4096`, `--parallel 1`.
- **Hardware / build:** NVIDIA GB10, llama.cpp `82d6bb284d1f` (`121a-real`), `ctx_size=131072`,
  `--jinja`, `--n-gpu-layers 999`. Weights: 17.6 GB GGUF at
  `/opt/models/unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- **Artifacts:**
  - Qwen: [`state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full/`](../state/evals/sec/qwen3.8-27b-spark-cuda/20260915T024745Z-extract-full/)
  - Spark 20b / 120b / Terra (unchanged): linked from the [re-run report](eval-report-2026-09-spark-cuda-rerun.md#setup)

The machine-generated table is [`state/evals/report-sec.md`](../state/evals/report-sec.md).
Its “gpt-oss vs Terra” comparison block still keys on those three models; the numbers below
were sliced from `results.jsonl` so Qwen is in every table.

## Results

### Accuracy by context mode

| Slice | n (Qwen) | 20b | 120b | **Qwen** | Terra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full document | 93 / 104‡ | 94/104 (90.4%) | 103/104 (99.0%) | **93/93 (100%)** | 103/104 (99.0%) |
| Item 8 section fallback | 20 / 17‡ | 12/17 (70.6%) | 16/17 (94.1%) | **20/20 (100%)** | 17/17 (100%)§ |
| BM25 chunk fallback | 8 / 0 | — | — | **3/8 (37.5%)** | — |
| **All questions** | **121** | **106/121 (87.6%)** | **119/121 (98.3%)** | **116/121 (95.9%)** | **120/121 (99.2%)** |

‡ Mode counts differ because tokenizers differ. gpt-oss treats NEE as full (124K tokens) and
GS/STWD as Item 8 (17 questions). Qwen’s tokenizer is ~10–15% fatter on the same text, so
NEE (138K) also falls to Item 8, and **GS Item 8 itself no longer fits** (~112K gpt-oss tokens
× ~1.20 from the STWD section ratio ≈ 134K > 131K) → BM25 chunks.

§ Terra has no fallback; it reads GS/STWD whole at 1.05M context.

Two readings:

- **When Qwen sees the document (full or Item 8), it does not miss.** 113/113. That includes
  CAT `LongTermDebtNoncurrent` (30.7B machinery-only), which 120b got wrong, and every NEE
  line after the section fallback.
- **The 95.9% headline is a retrieval-path penalty, not a dense-27B quality gap vs 120b.**
  All five misses are GS in chunked mode. 120b, reading the same filing from Item 8, scored
  7/8 there.

### Per-tag accuracy

| Tag | 20b | 120b | **Qwen** | Terra |
| --- | ---: | ---: | ---: | ---: |
| Assets | 11/12 | 12/12 | **11/12** | 12/12 |
| CashAndCashEquivalentsAtCarryingValue | 10/11 | 11/11 | 11/11 | 11/11 |
| CommonStockSharesOutstanding | 5/10 | 10/10 | **9/10** | 9/10 |
| EarningsPerShareDiluted | 12/12 | 12/12 | 12/12 | 12/12 |
| Liabilities | 10/11 | 11/11 | **10/11** | 11/11 |
| LongTermDebtNoncurrent | 6/9 | 8/9 | **9/9** | 9/9 |
| NetCashProvidedByUsedInOperatingActivities | 11/12 | 11/12 | 11/12 | 12/12 |
| NetIncomeLoss | 10/12 | 12/12 | 12/12 | 12/12 |
| OperatingIncomeLoss | 9/9 | 9/9 | 9/9 | 9/9 |
| Revenues | 10/11 | 11/11 | 11/11 | 11/11 |
| StockholdersEquity | 12/12 | 12/12 | **11/12** | 12/12 |

Qwen has none of 20b’s `off_by_scale` share-count errors. The four tags that are not 12/12
(or 11/11 / 10/10 / 9/9) are exactly the GS chunked misses.

### Per-company accuracy

| Ticker | 20b | 120b | **Qwen** | Terra | Qwen mode |
| --- | ---: | ---: | ---: | ---: | --- |
| AAPL | 10/11 | 11/11 | 11/11 | 11/11 | full |
| CAT | 11/11 | 10/11 | **11/11** | 11/11 | full |
| GS | **4/8** | 7/8 | **3/8** | 8/8 | **chunked** |
| HD | 10/11 | 11/11 | 11/11 | 10/11 | full |
| INTU | 10/11 | 11/11 | 11/11 | 11/11 | full |
| MSFT | 11/11 | 11/11 | 11/11 | 11/11 | full |
| NEE | 9/11 | 11/11 | 11/11 | 11/11 | section (120b: full) |
| PG | 8/9 | 9/9 | 9/9 | 9/9 | full |
| PLTR | 9/10 | 10/10 | 10/10 | 10/10 | full |
| STWD | 8/9 | 9/9 | 9/9 | 9/9 | section |
| WMT | 8/9 | 9/9 | 9/9 | 9/9 | full |
| XOM | 8/10 | 10/10 | 10/10 | 10/10 | full |

### Every Qwen miss, itemized

All five are `GS:10-K:2025-12-31`, mode `chunked`:

- **Operating cash flow:** +17.007B vs expected −45.154B. Identical wrong number to both
  gpt-oss models on this item. Consolidated cash flow says “used for”; the parent-only note
  says “provided by”. Chunked retrieval can easily surface the note and miss the
  consolidated statement.
- **Assets:** `unknown` vs 1.809T.
- **Liabilities:** 425.7B vs 1.684T (a real GS subtotal, not a scale error).
- **Stockholders’ equity:** `unknown` vs 125.0B.
- **Common shares outstanding:** empty answer, `finish_reason=length` — the one truncated
  item (hit the 4096 budget entirely on reasoning). Cover-page share count is the kind of
  fact BM25 chunks of Item 8 are bad at.

Qwen **got right** on GS: net income, diluted EPS, cash. 120b’s other miss (CAT long-term
debt, consolidated-vs-machinery) Qwen did not repeat.

## Why Qwen took ~5× longer than gpt-oss-120b

Wall clock is 125 min vs 26 min on the same Spark, same ctx, same serial harness. Request
time decomposes as:

| Phase | Qwen | gpt-oss-120b | gpt-oss-20b | Qwen / 120b |
| --- | ---: | ---: | ---: | ---: |
| Prefill (new prompt tokens) | **47 min** | 13 min | 7 min | **~3.7×** |
| Decode (completion) | **78 min** | 12 min | 11 min | **~6.3×** |
| **Wall clock** | **125 min** | **26 min** | **19 min** | **~4.8×** |

Cold first-question prefill on a ~50–120K filing is 70–200 s for Qwen vs 26–101 s for 120b.
Warm follow-ups (prompt cache hit) are 34 s p50 vs 5 s p50.

Approximate causes, in order:

1. **Dense vs mixture-of-experts, on a bandwidth-bound GPU.** Decode on GB10 is limited by
   how many weight bytes move per token. Qwen3.8-27B is dense: every token reads the whole
   ~17 GB Q4_K checkpoint. `gpt-oss-120b` is MXFP4 MoE (~117B total, **~5.1B active** per
   token); `gpt-oss-20b` activates ~3.6B. Observed decode: **9.8 t/s vs 30 t/s vs 45 t/s**.
   That ~3× gap vs 120b is the architectural floor — you would see it on a single-token
   chat, before any eval.
2. **Qwen thinks about twice as much.** 43.6k reasoning tokens vs 20.7k (120b) vs 28.2k
   (20b); p50 227 vs 140 tokens of hidden reasoning per question. Combined with (1):
   3× slower tokens × 2× more tokens ≈ the **6× decode-time** line in the table. This is
   also why warm p50 is 34 s vs 5 s even when the filing is already in cache.
3. **Prefill kernels are slower on the dense graph, and the tokenizer is fatter.** Cold
   prompt throughput is ~610–740 t/s for Qwen vs ~1,240–1,830 t/s for 120b (~2.5×). On top
   of that the Qwen tokenizer emits **~10–15% more tokens** for the same 10-K (AAPL 51.4K vs
   45.6K; XOM 122K vs 106K). More tokens at fewer tokens/s is the ~3.7× prefill time.
   (The eval-wide `prompt_tps` p50 of 280 vs 250 is *not* the fair prefill number — it is
   pulled down by cached warm requests with tiny new-token counts.)
4. **Cache reuse is a bit worse, not the main story.** Qwen reused 85% of prompt tokens vs
   90% for gpt-oss. The extra fallback path (NEE section + GS chunks) means more distinct
   prefixes, so slightly less prefix-cache hit rate. That is a few minutes, not an hour.

What this is **not**: SSH/tmux overhead, a sick GPU, or the overnight attach. GPU was at
~96% during the run; wall clock matches the sum of per-request times to <1%. Terra’s 20 min
wall vs 3.5 min compute is the rate-limit story from the re-run report and does not apply
here.

If the goal is “same Spark, faster 10-K extraction,” `gpt-oss-120b` remains the throughput
choice. Qwen would need a larger ctx (so GS stays on Item 8 / full doc) *and* would still
decode at ~10 t/s.

### Latency and tokens

| Model | Wall clock | Request time | TTFT p50 | Total p50 / p95 | Prompt tok | Cached | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-oss-20b` | 19 min | 19 min | 0.39 s | 3.6 s / 38 s | 10.52 M | 90% | 28.2 k |
| `gpt-oss-120b` | 26 min | 26 min | 0.57 s | 5.8 s / 63 s | 10.52 M | 90% | 20.7 k |
| **`qwen3.8-27b`** | **125 min** | **125 min** | **8.6 s** | **36 s / 196 s** | 10.33 M | **85%** | **43.6 k** |
| `gpt-5.6-terra` | 20 min | 3.5 min | 1.44 s | 1.6 s / 2.9 s | 12.39 M | 0% | 1.3 k |

Qwen TTFT p50 (8.6 s) is dominated by warm requests that still prefill a small uncached tail
and then stream reasoning at 10 t/s. Cold TTFT p50 is **121 s**.

## Pros and cons

### `qwen3.8-27b` on the Spark

**Pros**

- Perfect on every question where the filing (or Item 8) fit: 113/113.
- No scale errors; beat 120b on CAT long-term debt; beat Terra on HD share count.
- Fits easily (~18 GB weights vs ~80 GB for 120b MXFP4) — room to serve something else
  alongside, or to raise ctx if you rebuild KV budget.
- Same air-gap / no-egress properties as the other local models.

**Cons**

- **~5× wall clock vs 120b** on this workload (dense decode + longer reasoning).
- Fatter tokenizer + 131K cap sent GS to BM25; that path scored 3/8 and is why it trails
  120b overall.
- One truncation at 4096 reasoning tokens (GS share count).
- Native ctx is advertised at 262K; we served 131K to match the gpt-oss runs. Raising it
  would likely recover the GS Item 8 path (and eat more KV memory / even slower long prefills).

### Still true from the re-run

`gpt-oss-120b` remains the on-prem recommendation for unattended extraction (98.3%, 26 min).
Terra still buys the last point and whole-document GS/STWD for ~$25/run. `gpt-oss-20b` is
still the latency/memory option with human review (87.6%, scale errors).

## Caveats

> **Review notes (2026-09-15).** Single run; bootstrap 95% CI for Qwen is 0.92–0.99 in
> [`report-sec.md`](../state/evals/report-sec.md) and overlaps 120b (0.96–1.00). The
> five-miss gap is concentrated on one filing and one retrieval mode, so it is more
> informative as a fallback-path result than as a general “27B vs 120b quality” claim.
> No `--no-document` memorisation control. Item 8 overflow for Qwen/GS is inferred from
> tokenizer ratios plus the observed `chunked` mode — the harness does not currently log
> the rejected section size.

1. **Not a 27B-vs-117B bake-off.** Active parameter count and quant (Q4_K_XL vs MXFP4) both
   differ; decode t/s is the fair hardware comparison.
2. **Tokenizers differ.** Same HTML 10-K is 10–15% more Qwen tokens than gpt-oss, which
   changes which filings fit. Terra uses `o200k_base` (12.39 M prompt tokens).
3. **Config hashes differ** across local vs API vs this Qwen registry entry; the report
   drift warning is expected.
4. Qwen3.8-27B is a VL checkpoint served text-only (`--jinja`). Vision was not exercised.

## Reproduce

```bash
cd ~/projects/llama-cpp-spark
export LOCAL_LLM_EDGAR_USER_AGENT="local-llm you@example.com"

# Model is in models.toml as qwen3.8-27b; GGUF already downloaded on this box.
uv run local-llm serve qwen3.8-27b
# optional: tmux new -s qwen
uv run local-llm eval sec run qwen3.8-27b --task extract-full --forms 10-K
uv run local-llm eval report --suite sec
uv run local-llm stop qwen3.8-27b
```

gpt-oss / Terra reproduction: [eval-report-2026-09-spark-cuda-rerun.md](eval-report-2026-09-spark-cuda-rerun.md#reproduce).

To try whether GS stays on Item 8, serve with a larger window (KV memory will grow):

```bash
uv run local-llm serve qwen3.8-27b -- --ctx-size 262144
```

## Bottom line for the team

- **`qwen3.8-27b` is a strong reader when the document fits (100% on 113 questions) and a
  weak one when the harness is down to BM25 chunks of GS (3/8).** Headline 95.9% is that mix.
- **Do not use it if wall clock matters on this suite.** 125 min vs 26 min for 120b is mostly
  dense-vs-MoE decode (10 vs 30 t/s) plus ~2× reasoning tokens, not a flaky session.
- **Keep `gpt-oss-120b` as the Spark default for 10-K extraction.** Reach for Qwen when you
  want a denser non-oss checkpoint, have memory headroom, and can give it enough context that
  oversized filings stay on Item 8 (or whole-doc).
- **Harness follow-up:** log the token count of the Item 8 candidate before rejecting it, and
  consider a slightly higher local ctx for fat tokenizers so GS does not skip from section to
  chunks.
