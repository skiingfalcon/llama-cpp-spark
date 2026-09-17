# SEC 10-K extraction — Nemotron-3-Super on the Spark (September 2026)

Same 121-question `extract-full` suite as the
[Spark re-run](eval-report-2026-09-spark-cuda-rerun.md) (gpt-oss-20b / 120b / Terra) and the
[Qwen add-on](eval-report-2026-09-spark-cuda-qwen.md). This report adds
**`nemotron-3-super`** (NVIDIA Nemotron-3-Super-120B-A12B, ggml-org Q4_K, hybrid
Mamba-2 / latent-MoE / attention, ~12.7B active, native 1M context) on the same
NVIDIA DGX Spark / CUDA stack.

Two runs on purpose:

| Run | Served ctx | Eval input budget | Score | Fallback | Truncated | Wall clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fair vs gpt-oss | 524,288 | **131,072** (`--max-input-tokens`) | **93.4% (113/121)** | 28 | 6 | 148 min |
| Unconstrained | **1,048,576** | none (full window) | **92.6% (112/121)** | **0** | 6 | 153 min |

| Model | Where it ran | Params (active / total) | Score | Wall clock |
| --- | --- | --- | ---: | ---: |
| `gpt-5.6-terra` | OpenAI API | hosted | **99.2% (120/121)** | 20 min† |
| `gpt-oss-120b` | Local Spark | ~5.1B / 117B MoE | **98.3% (119/121)** | **26 min** |
| `qwen3.8-27b` | Local Spark | 27B / 27B dense | 95.9% (116/121) | 125 min |
| **`nemotron-3-super` (131K budget)** | **Local Spark** | **~12.7B / 120B hybrid** | **93.4% (113/121)** | **148 min** |
| **`nemotron-3-super` (full window)** | **Local Spark** | same | **92.6% (112/121)** | **153 min** |
| `gpt-oss-20b` | Local Spark | ~3.6B / 21B MoE | 87.6% (106/121) | 19 min |

† Terra wall clock includes HTTP 429 backoff.

**Headline:** giving Nemotron the whole filing (1M ctx, zero fallback) did **not** beat the
131K-budget run, and neither run approaches `gpt-oss-120b`. Six truncations at the 4096
completion budget dominate the miss list in both runs; raising input context does not fix
that. The full window *did* eliminate section/chunk fallback and recovered GS Assets /
Liabilities, but new full-doc misses and truncations offset the gain.

## Setup

- **Task / corpus / tags / scoring:** identical to the re-run (`scoring_version`
  `2026-09-13.1`, 0.5% relative tolerance, 12 companies, latest 10-K, 11 XBRL tags).
- **Decoding:** `temperature=0`, `seed=42`, `max_tokens=4096`, `--parallel 1`.
- **Hardware / build:** NVIDIA GB10, llama.cpp `82d6bb284d1f` (`121a-real`),
  `--jinja`, `--n-gpu-layers 999`, `n_parallel=1`. Weights: ~70 GB GGUF
  `Nemotron-3-Super-120B-Q4_K.gguf` from `ggml-org/nemotron-3-super-120b-GGUF`.
- **Run A (fair):** served at 524K; eval capped with `--max-input-tokens 131072` so GS/STWD
  (and NEE under this tokenizer) still fall back like the other local models.
- **Run B (unconstrained):** served at **1M** (`--ctx-size 1048576`); eval with no input
  cap → every filing read whole (`fallback=0`).
- **Artifacts:**
  - 131K budget: [`state/evals/sec/nemotron-3-super-spark-cuda/20260915T122649Z-extract-full/`](../state/evals/sec/nemotron-3-super-spark-cuda/20260915T122649Z-extract-full/)
  - Full window: [`state/evals/sec/nemotron-3-super-spark-cuda/20260916T103526Z-extract-full/`](../state/evals/sec/nemotron-3-super-spark-cuda/20260916T103526Z-extract-full/)
  - Spark 20b / 120b / Terra / Qwen: linked from the [re-run](eval-report-2026-09-spark-cuda-rerun.md#setup) and [Qwen](eval-report-2026-09-spark-cuda-qwen.md#setup) reports

The machine-generated table is [`state/evals/report-sec.md`](../state/evals/report-sec.md);
it currently shows the **latest** Nemotron run (full window). Both runs are on disk; the
tables below include both.

## Results

### Accuracy by context mode

| Slice | 20b | 120b | Qwen | **Nemo 131K** | **Nemo full** | Terra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full document | 94/104 | 103/104 | 93/93 | **88/93** | **112/121** | 120/121 |
| Item 8 section | 12/17 | 16/17 | 20/20 | **20/20** | — | — |
| BM25 chunks | — | — | 3/8 | **5/8** | — | — |
| **All questions** | **106/121** | **119/121** | **116/121** | **113/121** | **112/121** | **120/121** |

Under the fair 131K budget Nemotron falls back on the same 28 questions as Qwen (NEE +
GS/STWD shape). On Item 8 it is perfect (20/20); on GS chunks 5/8 (better than Qwen’s 3/8).
The unconstrained run removes that variable entirely — and still trails Qwen and 120b.

### What the bigger window bought (and what it did not)

Diff of the two Nemotron runs (same model, same decoding, same suite):

| Item | 131K budget | Full window |
| --- | --- | --- |
| GS Assets | miss (chunked, truncated) | **ok** (full) |
| GS Liabilities | miss (chunked, truncated) | **ok** (full) |
| GS NetIncomeLoss | **ok** (chunked) | miss (16.3B vs 17.2B) |
| NEE Cash | **ok** (section) | miss (truncated) |
| NEE Shares | **ok** (section) | miss (truncated) |

Net: **+2 GS balance-sheet lines, −3 elsewhere → −1 overall** (113 → 112). Whole-document
context is necessary for fair comparison with Terra on GS/STWD, but on this suite it is not
sufficient to close the gap to gpt-oss-120b. The binding constraint is **output truncation**,
not input window size.

### Per-tag accuracy

| Tag | 20b | 120b | Qwen | **Nemo 131K** | **Nemo full** | Terra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assets | 11/12 | 12/12 | 11/12 | 11/12 | **12/12** | 12/12 |
| CashAndCashEquivalentsAtCarryingValue | 10/11 | 11/11 | 11/11 | 11/11 | **10/11** | 11/11 |
| CommonStockSharesOutstanding | 5/10 | 10/10 | 9/10 | 8/10 | **7/10** | 9/10 |
| EarningsPerShareDiluted | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| Liabilities | 10/11 | 11/11 | 10/11 | 10/11 | **11/11** | 11/11 |
| LongTermDebtNoncurrent | 6/9 | 8/9 | 9/9 | 7/9 | 7/9 | 9/9 |
| NetCashProvidedByUsedInOperatingActivities | 11/12 | 11/12 | 11/12 | 11/12 | 11/12 | 12/12 |
| NetIncomeLoss | 10/12 | 12/12 | 12/12 | 12/12 | **11/12** | 12/12 |
| OperatingIncomeLoss | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| Revenues | 10/11 | 11/11 | 11/11 | 10/11 | 10/11 | 11/11 |
| StockholdersEquity | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 12/12 |

### Per-company accuracy

| Ticker | 20b | 120b | Qwen | **Nemo 131K** | **Nemo full** | Terra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | 10/11 | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| CAT | 11/11 | 10/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| GS | 4/8 | 7/8 | 3/8 | 5/8 | **6/8** | 8/8 |
| HD | 10/11 | 11/11 | 11/11 | 9/11 | 9/11 | 10/11 |
| INTU | 10/11 | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| MSFT | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| NEE | 9/11 | 11/11 | 11/11 | 11/11 | **9/11** | 11/11 |
| PG | 8/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| PLTR | 9/10 | 10/10 | 10/10 | 9/10 | 9/10 | 10/10 |
| STWD | 8/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| WMT | 8/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| XOM | 8/10 | 10/10 | 10/10 | 8/10 | 8/10 | 10/10 |

### Every Nemotron miss (full-window run)

Six of nine are empty answers with `finish_reason=length` (hit the 4096 budget on
`<think>` / reasoning):

- **GS operating cash flow** — truncated (same item the gpt-oss models also struggle with
  when wording matches the parent-only note).
- **GS NetIncomeLoss** — answered 16.3B vs 17.2B (real wrong number with the whole filing).
- **XOM Revenues** — 323.9B vs 332.2B (same wrong answer as the 131K run).
- **XOM LongTermDebtNoncurrent** — truncated.
- **PLTR CommonStockSharesOutstanding** — 2.291B vs 2.391B (same as 131K run).
- **NEE Cash / NEE Shares** — truncated (both were correct under Item 8 in the 131K run).
- **HD LongTermDebt / HD Shares** — truncated (same as 131K run).

### Latency and tokens

| Model | Wall clock | TTFT p50 | Total p50 / p95 | Decode t/s | Reasoning tok | Cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-oss-20b` | 19 min | 0.39 s | 3.6 s / 38 s | 45 | 28 k | 90% |
| `gpt-oss-120b` | 26 min | 0.57 s | 5.8 s / 63 s | 30 | 21 k | 90% |
| `qwen3.8-27b` | 125 min | 8.6 s | 36 s / 196 s | 10 | 44 k | 85% |
| **Nemo 131K** | **148 min** | **3.0 s** | **44 s / 211 s** | **20** | **134 k** | **84%** |
| **Nemo full** | **153 min** | **3.2 s** | **47 s / 218 s** | **19** | **131 k** | **87%** |
| `gpt-5.6-terra` | 20 min | 1.44 s | 1.6 s / 2.9 s | — | 1.3 k | 0% |

Nemotron decode (~20 t/s) sits between dense Qwen (10) and gpt-oss-120b (30), as expected
for ~12.7B active parameters. Wall clock is still ~6× 120b because it emits **~6× more
reasoning tokens** (131–134k vs 21k). The full-window run spent more total prompt tokens
(14.1M vs 10.5M) reading GS/STWD/NEE whole, but median latency barely moved — decode of
long `<think>` traces dominates.

## Pros and cons

### `nemotron-3-super` on the Spark

**Pros**

- Cheap KV for long context (hybrid: only 8 attention layers keep KV; 1M fits next to ~70 GB
  weights on 128 GB unified memory).
- Full-window run really does read every 10-K whole (`fallback=0`) — the capability gpt-oss
  and Qwen lack at 131K.
- Better than 20b overall; better than Qwen on GS chunks under the fair budget (5/8 vs 3/8).
- Same air-gap / no-egress properties as the other local models.

**Cons**

- **Not competitive with gpt-oss-120b on this suite** (92.6–93.4% vs 98.3%).
- **Truncation-heavy:** 6 empty answers per run at `max_tokens=4096`. Raising input ctx did
  not reduce truncations.
- **~6× slower than 120b** (≈150 min vs 26 min) from long reasoning, not from the window.
- Full window did not improve the headline score vs the 131K-budget run.
- Ollama GGUFs are incompatible (`ffn_down_exps` shape); must use the ggml-org file.

## Caveats

> **Review notes (2026-09-17).** Single run per budget. Bootstrap 95% CI for the full-window
> run is 0.87–0.97 in [`report-sec.md`](../state/evals/report-sec.md) and overlaps Qwen and
> the Halo stacks; it does **not** overlap gpt-oss-120b Spark (0.96–1.00) at the point
> estimates, but the sample is still 121 items / one seed. No `--no-document` memorisation
> control. The report CLI currently surfaces the latest Nemotron config only; keep both
> artifact dirs when comparing budgets.

1. **Fair vs unconstrained are different `config_hash` values** by design (`max_input_tokens`
   is in `task_config`).
2. Served ctx for the unconstrained run is 1M; registry default remains 524K.
3. Tokenizers differ across gpt-oss / Qwen / Nemotron / Terra; fallback boundaries move.
4. Truncation count is sensitive to `max_tokens`; a follow-up with a larger completion
   budget (or reasoning budget) would be a better test of “is Nemotron accurate when it
   finishes” than another context-window bump.

## Reproduce

```bash
cd ~/projects/llama-cpp-spark
export LOCAL_LLM_EDGAR_USER_AGENT="local-llm you@example.com"

uv run local-llm download nemotron-3-super
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Fair vs gpt-oss (131K input budget, 524K serve)
uv run local-llm serve nemotron-3-super
uv run local-llm eval sec run nemotron-3-super --task extract-full --forms 10-K --max-input-tokens 131072
uv run local-llm stop nemotron-3-super

# Unconstrained (1M serve, no input cap)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
uv run local-llm serve nemotron-3-super --ctx-size 1048576
uv run local-llm eval sec run nemotron-3-super --task extract-full --forms 10-K

uv run local-llm eval report --suite sec
uv run local-llm stop nemotron-3-super
```

Prefer `tmux` around the eval commands so SSH drops do not kill the overnight run.

## Bottom line for the team

- **`gpt-oss-120b` remains the Spark default for unattended 10-K extraction** — higher
  accuracy, far lower wall clock, fewer truncations.
- **Nemotron’s 1M window is real** (zero fallback on this suite) but **did not buy accuracy
  here**. The miss pattern is dominated by reasoning that hits the 4096 completion budget.
- Use Nemotron when you specifically need >131K whole-document reads and can afford ~2.5 h
  per suite; otherwise keep 120b for quality/latency and Terra for the last percentage point.
- **Next experiment worth running:** same full-window serve with a higher `max_tokens` (or
  a reasoning-effort cap) before another context-window comparison. The window hypothesis is
  already tested; the truncation hypothesis is not.
