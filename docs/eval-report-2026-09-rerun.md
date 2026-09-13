# SEC 10-K extraction eval — September 2026 re-run (post-fix)

Re-run of the three-model comparison after the harness fixes documented in
[the first report](eval-report-2026-09.md#what-changed-since-this-run). Keep both documents:
the first report holds the pre-fix raw numbers (96-question paired set, 512-token budget,
skips), this one holds the post-fix numbers (121 questions, 4096-token budget, per-filing
fallback). Cross-checking the two shows exactly what each fix bought.

| Model | Where it ran | Context | First run (paired 96) | **Re-run (all 121)** |
| --- | --- | ---: | ---: | ---: |
| `gpt-oss-20b` | Local Spark | 131,072 | 77.1% (74/96) | **87.6% (106/121)** |
| `gpt-oss-120b` | Local Spark | 131,072 | 89.6% (86/96) | **98.3% (119/121)** |
| `gpt-5.6-terra` | OpenAI API | 1,050,000 | 95.8% (92/96) | **99.2% (120/121)** |

All three models answered **every** question this time — no context skips, no rate-limit
losses — so raw score and paired score are the same 121-question set. Oversized filings
(GS 242K tokens, STWD 176K) were answered by the local models from their Item 8
financial-statements section via the new per-filing fallback; Terra read them whole.

## What changed between the runs

1. **`max_tokens` 512 → 4096.** gpt-oss spends hidden reasoning from the completion budget;
   at 512 it truncated mid-thought on ~15% of items. This run: 1 truncated item (20b), 0 (120b).
2. **Ground-truth alias fixes.** The four misses shared by all three models in run 1
   (WMT/NEE/XOM `Revenues`, HD `LongTermDebt`) were expected-value bugs; all green now.
3. **Per-filing context fallback** (full → Item 8 → BM25 chunks) instead of skipping
   GS/STWD: coverage went from 103–110 answered to 121 for every model.
4. **OpenAI client honours `Retry-After`** (8 retries, 60 s cap): run 1 lost 10 items to
   HTTP 429; this run lost none.

## Setup

- **Task:** `extract-full`, 12 companies (AAPL, MSFT, GS, XOM, PLTR, WMT, PG, CAT, NEE, HD,
  INTU, STWD), latest 10-K each, 11 XBRL tags, 0.5% relative tolerance, ground truth from
  EDGAR company facts with the fixed alias rules.
- **Decoding:** local `temperature=0`, `seed=42`, `max_tokens=4096`; Terra provider defaults
  with the same 4096 completion budget, `--parallel 1`.
- **Hardware / build:** NVIDIA GB10, llama.cpp `82d6bb284d1f`, local models at 131,072
  tokens per slot.
- **Artifacts (this run):**
  - [`state/evals/sec/gpt-oss-20b/20260912T135023Z-extract-full/`](../state/evals/sec/gpt-oss-20b/20260912T135023Z-extract-full/)
  - [`state/evals/sec/gpt-oss-120b/20260912T132250Z-extract-full/`](../state/evals/sec/gpt-oss-120b/20260912T132250Z-extract-full/)
  - [`state/evals/sec/openai_gpt-5.6-terra/20260912T142344Z-extract-full/`](../state/evals/sec/openai_gpt-5.6-terra/20260912T142344Z-extract-full/)
- **First-run artifacts (for cross-checking):** `20260912T022410Z` (20b), `20260912T030429Z`
  (120b), `20260912T033135Z` (Terra), linked from the first report.

## Results

### Accuracy by context mode

| Slice | n | 20b | 120b | Terra |
| --- | ---: | ---: | ---: | ---: |
| Full document (no OSS fallback) | 104 | 94/104 (90.4%) | 103/104 (99.0%) | 103/104 (99.0%) |
| GS/STWD via Item 8 fallback | 17 | 12/17 (70.6%) | 16/17 (94.1%) | 17/17 (100%) |
| **All questions** | **121** | **106/121 (87.6%)** | **119/121 (98.3%)** | **120/121 (99.2%)** |

Two readings:

- **On equal footing (full document in context), 120b ties Terra: 103/104 each.** The
  frontier model's remaining edge on this suite comes almost entirely from its 1.05M window
  reading GS and STWD whole while the local models work from an excerpt.
- The Item 8 fallback works well for 120b (16/17) and poorly for 20b (12/17). The excerpt
  itself is complete: parsing the GS 10-K with the harness shows Item 8 spans 460K characters
  and contains the consolidated balance sheet, cash-flow statement and notes. The one 120b
  miss inside it is a wrong-statement pick, not a missing statement (see below).

### Per-tag accuracy

| Tag | 20b | 120b | Terra | First run (20b/120b/Terra of paired 96) |
| --- | ---: | ---: | ---: | --- |
| Assets | 11/12 | 12/12 | 12/12 | 8/9, 9/9, 9/9 |
| CashAndCashEquivalentsAtCarryingValue | 10/11 | 11/11 | 11/11 | 9/9, 9/9, 9/9 |
| CommonStockSharesOutstanding | **5/10** | **10/10** | 9/10 | 2/8, 4/8, 8/8 |
| EarningsPerShareDiluted | 12/12 | 12/12 | 12/12 | 9/9, 9/9, 9/9 |
| Liabilities | 10/11 | 11/11 | 11/11 | 6/8, 8/8, 8/8 |
| LongTermDebtNoncurrent | 6/9 | 8/9 | 9/9 | 3/8, 6/8, 7/8 |
| NetCashProvidedByUsedInOperatingActivities | 11/12 | 11/12 | 12/12 | 7/8, 7/8, 8/8 |
| NetIncomeLoss | 10/12 | 12/12 | 12/12 | 9/10, 10/10, 10/10 |
| OperatingIncomeLoss | 9/9 | 9/9 | 9/9 | 8/8, 8/8, 8/8 |
| Revenues | 10/11 | 11/11 | 11/11 | **7/10, 7/10, 7/10** |
| StockholdersEquity | 12/12 | 12/12 | 12/12 | 6/9, 9/9, 9/9 |

The alias fixes are visible in the cross-check: `Revenues` went from 7/10 *for all three
models* to near-perfect, confirming harness ambiguity rather than model weakness.
`CommonStockSharesOutstanding` remains 20b's clearest capability gap (5/10, including 3
`off_by_scale` answers); 120b improved from 4/8 to 10/10 once its answers stopped truncating.

### Per-company accuracy

| Ticker | 20b | 120b | Terra |
| --- | ---: | ---: | ---: |
| AAPL | 10/11 | 11/11 | 11/11 |
| CAT | 11/11 | 10/11 | 11/11 |
| GS | **4/8** | 7/8 | 8/8 |
| HD | 10/11 | 11/11 | 10/11 |
| INTU | 10/11 | 11/11 | 11/11 |
| MSFT | 11/11 | 11/11 | 11/11 |
| NEE | 9/11 | 11/11 | 11/11 |
| PG | 8/9 | 9/9 | 9/9 |
| PLTR | 9/10 | 10/10 | 10/10 |
| STWD | 8/9 | 9/9 | 9/9 |
| WMT | 8/9 | 9/9 | 9/9 |
| XOM | 8/10 | 10/10 | 10/10 |

### Every remaining miss, itemized

**gpt-oss-120b (2):**

- GS operating cash flow: answered +17.0B vs expected −45.2B. The 20b gave the *identical*
  wrong number. Both values are inside the Item 8 excerpt: −45,154 is on the consolidated
  statement of cash flows (line reads "Net cash **used for** operating activities"), while
  17,007 is the parent-company-only condensed cash-flow statement in the notes (line reads
  "Net cash **provided by** operating activities"). The question asks for "net cash provided
  by operating activities", so both local models matched the wording literally and took the
  parent-only line. Terra, reading the whole filing, did not. This is a model miss with a
  label-wording contributor, not a section-boundary problem; the harness follow-up is to word
  the question "consolidated net cash provided by (used for) operating activities".
- CAT `LongTermDebtNoncurrent`: answered 50.7B (consolidated, incl. Financial Products
  captive-finance debt) vs expected 30.7B. Genuine which-line ambiguity for industrials with
  finance arms.

**gpt-5.6-terra (1):**

- HD `CommonStockSharesOutstanding`: replied `unknown` despite the count being on the 10-K
  cover page. Both local models got it.

**gpt-oss-20b (15):** 5 share counts (3 scale errors), 3 long-term debt lines, 4 of the 8
GS Item-8 questions, NEE Assets/Liabilities (subsidiary-level totals), HD NetIncomeLoss
(`unknown`), XOM Revenues, 1 truncated answer (GS cash, hit the 4096 budget mid-reasoning).

### Latency, tokens, cost

| Model | Wall clock | Request time (sum) | TTFT p50 | Total p50 / p95 | Prompt tokens | Cached | Output + reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-oss-20b` | 19 min | 19 min | 0.39 s | 3.6 s / 37.8 s | 10.52 M | 9.47 M (90%) | 29.8 k |
| `gpt-oss-120b` | 26 min | 26 min | 0.57 s | 5.8 s / 62.9 s | 10.52 M | 9.47 M (90%) | 22.4 k |
| `gpt-5.6-terra` | 20 min | **3.5 min** | 1.44 s | 1.6 s / 2.9 s | 12.39 M | **0** | 2.3 k |

- Terra's wall clock is ~6× its request time: `--parallel 1` plus rate-limit backoff. The
  client now honours `Retry-After`, so every item eventually succeeded (run 1 dropped 10).
- Estimated Terra cost at list prices ($2/M fresh input, $0.20/M cached, $12/M output):
  **~$25 per run**, with **zero** cache-discounted input this time. Repeated questions on
  the same 240K-token filing re-paid full input price every time.
- Local prefix caching reused 90% of prompt tokens. That asymmetry is the core
  unit-economics argument for local serving on many-questions-per-document workloads.
- **Local TTFT is not comparable to the first run.** The client now stamps time-to-first-token
  on the first streamed *reasoning* delta; the first run stamped it on the first visible
  answer token, after reasoning had finished (every first-run 120b item had TTFT within 1 s of
  total time; no re-run item does). The re-run's 0.4–0.6 s p50 is a true first-token latency,
  the first run's 3.7–6.1 s was effectively total latency. Compare **total p50** across runs
  instead: 3.6 s / 5.8 s now vs 4.3 s / 6.5 s before. The p95 (~40–60 s) is each new filing's
  cold prefill.

## Pros and cons

### Local Spark (`gpt-oss-20b` / `gpt-oss-120b`)

**Pros**

- **120b is at practical parity with the frontier model on this workload**: 98.3% vs 99.2%,
  dead even (103/104) when both see the whole document.
- No per-token cost; ~$25/run of API spend avoided every run.
- Data never leaves the building — matters for confidential filings / MNPI.
- 90% prompt-cache reuse makes follow-up questions on the same filing nearly free.
- Deterministic decoding (temp 0 / fixed seed) — reruns are reproducible.

**Cons**

- Hard 131K native context: GS/STWD answered from Item 8 rather than the whole filing.
  Anything outside Item 8 (cover page, MD&A tables) is invisible to the local models on
  those filings.
- 20b is not accurate enough for unattended numeric extraction (87.6%, scale errors).
- Slower per-answer latency than the API (3.6–5.8 s p50 vs 1.6 s), long cold prefills.
- You own ops: serving, CUDA/llama.cpp upgrades, memory budgeting, auth in front of `0.0.0.0`.

### Hosted frontier (`gpt-5.6-terra`)

**Pros**

- Best score (99.2%) and the only model to read the 242K-token GS filing whole.
- Fastest, most consistent per-answer latency (1.6 s p50, 2.9 s p95).
- No local GPU / serving ops.

**Cons**

- ~$25 per suite run with no cache discount observed — cost scales with questions × document size.
- Rate limits stretched 3.5 min of work into 20 min of wall clock even at `--parallel 1`.
- Data egress; provider retention/residency terms apply.
- No fixed temperature/seed — reruns aren't bit-identical.

## Caveats

1. **Latency is not a hardware bake-off.** Terra numbers include network and provider queueing.
2. **Tokenizers differ.** GGUF tokenizer vs `o200k_base`: the same corpus sums to 10.52M vs
   12.39M prompt tokens.
3. **The GS cash-flow miss is shared by both local models with the identical wrong value.**
   Verified against the parsed filing: the excerpt contains both the consolidated (−45,154)
   and parent-only (17,007) statements; the models chose the one whose line wording matched
   the question. Count it as a model miss; tighten the question wording.
4. **Config hashes differ by design** between local and API runs; the report's drift warning
   is expected.
5. Single run per model; no variance estimate. Local runs are deterministic, Terra is not.

## Reproduce

```bash
cd ~/projects/llama-cpp-spark
export LOCAL_LLM_EDGAR_USER_AGENT="local-llm you@example.com"

uv run local-llm eval sec fetch                                   # corpus (once)

uv run local-llm serve gpt-oss-20b
uv run local-llm eval sec run gpt-oss-20b --task extract-full --forms 10-K
uv run local-llm stop

uv run local-llm serve gpt-oss-120b                               # models.toml pins 131072
uv run local-llm eval sec run gpt-oss-120b --task extract-full --forms 10-K
uv run local-llm stop

# Frontier (OPENAI_API_KEY from .env; serial to respect rate limits)
uv run local-llm eval sec run gpt-5.6-terra \
  --provider openai --task extract-full --forms 10-K --parallel 1

# Cross-model table + gpt-oss-vs-Terra comparison (slices, by-tag, by-company,
# per-item disagreements), written to state/evals/report-sec.md
uv run local-llm eval report --suite sec
```

To repeat this on the AMD Strix Halo box (Windows), follow the README section
[Windows / AMD Strix Halo](../README.md#windows--amd-strix-halo); the eval commands are the
same and the report adds a Hardware block comparing the two machines. Locked-down hosts that
can only run LM Studio should use [`scripts/lmstudio.mjs`](../scripts/lmstudio.mjs) and see
[eval-report-2026-09-halo.md](eval-report-2026-09-halo.md).

## Bottom line for the team

- **`gpt-oss-120b` on the Spark is the recommendation for on-prem SEC extraction.** 98.3%
  overall, identical to Terra (103/104) whenever the document fits in 131K, zero API spend,
  no data egress, deterministic decoding. Its two remaining misses are both which-line picks:
  GS parent-only vs consolidated cash flow, and CAT consolidated vs machinery-only debt.
- **`gpt-5.6-terra` buys the last percentage point and >131K documents** for ~$25/run,
  slower wall clock under rate limits, no cache economics, and data egress. Use it when the
  filing genuinely exceeds 131K or the workload is one-shot rather than
  many-questions-per-document.
- **`gpt-oss-20b` is not accurate enough for unattended extraction** (87.6%, scale errors on
  share counts). Keep it for latency-sensitive or memory-tight serving with human review.
- **Next harness fix:** word the cash-flow question as "consolidated net cash provided by
  (used for) operating activities" so parent-only condensed statements in the notes are not
  a literal match. The Item 8 excerpt does not need widening.
