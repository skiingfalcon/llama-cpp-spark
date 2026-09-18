| model | task | n | score | 95% CI | skipped | truncated | ttft p50 s | total p50 s | prompt t/s | decode t/s | tokens | cached | reasoning | ctx | build | platform | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai:gpt-5.6-terra | extract-full | 121 | 0.992 | 0.98–1.00 | 0 | 0 | 1.44 | 1.55 | - | - | 12394653 | 0 | 1260 | 1050000 | openai/api | api | - |
| gpt-oss-120b | extract-full | 121 | 0.983 | 0.96–1.00 | 0 | 0 | 0.565 | 5.79 | 250 | 30 | 10542592 | 9466001 | 20720 | 131072 | 82d6bb284d1f/? | spark/cuda | - |
| qwen3.8-27b | extract-full | 121 | 0.983 | 0.96–1.00 | 0 | 0 | 1.44 | 6.21 | 1,668 | 55 | 12770130 | 11231740 | 35530 | 262144 | 82d6bb284d1f/? | halo/vulkan | - |
| gpt-oss-120b | extract-full | 121 | 0.967 | 0.93–0.99 | 0 | 0 | 1.38 | 5.15 | - | 29 | 9976288 | - | - | 131072 | 2.37.0/vulkan:lmstudio-2.37.0 | halo/vulkan | LM Studio stats; prompt t/s not measurable |
| qwen3.8-27b | extract-full | 121 | 0.959 | 0.92–0.99 | 0 | 1 | 8.6 | 36 | 280 | 9.76 | 10371887 | 8762395 | 43613 | 131072 | 82d6bb284d1f/121a-real | spark/cuda | - |
| gpt-oss-120b | extract-full | 121 | 0.95 | 0.91–0.98 | 0 | 0 | 1.19 | 7.43 | - | 19 | 9976288 | - | - | 131072 | 2.37.0/rocm:lmstudio-2.37.0 | halo/rocm | LM Studio stats; prompt t/s not measurable |
| glm-4.7-flash | extract-full | 121 | 0.942 | 0.90–0.98 | 0 | 4 | 0.602 | 24 | 146 | 28 | 10900593 | 9713858 | 103079 | 131072 | 82d6bb284d1f/121a-real | spark/cuda | - |
| nemotron-3-super | extract-full | 121 | 0.926 | 0.87–0.97 | 0 | 6 | 3.15 | 47 | 676 | 19 | 14103962 | 12318170 | 131326 | 1048576 | 82d6bb284d1f/121a-real | spark/cuda | - |
| qwen3.5-122b-a10b | extract-full | 121 | 0.909 | 0.85–0.96 | 0 | 10 | 3.87 | 101 | 578 | 17 | 12949305 | 11228093 | 219423 | 262144 | 82d6bb284d1f/121a-real | spark/cuda | - |
| gpt-oss-20b | extract-full | 121 | 0.876 | 0.81–0.93 | 0 | 1 | 0.39 | 3.59 | 564 | 45 | 10550003 | 9466000 | 28150 | 131072 | 82d6bb284d1f/? | spark/cuda | - |
| qwen3.8-27b | qa-financebench | 150 | 0.0946 | 0.04–0.16 | 0 | 2 | 0.347 | 4.69 | 2,569 | 76 | 216047 | 8715 | 81450 | 131072 | 82d6bb284d1f/? | halo/vulkan | - |

> warning: task extract-full: runs use different configs (455128a4cdd6, 58b2946660ca, 607363d040e8, 9466ddd63b34, 99d094a0fde9, 9cc51d84df5a, afc7e6a05fdb, cb268609a9e5, ea71250ebf92)

### Paired (items answered by every run of the task)

| task | model | paired n | correct | paired score | 95% CI |
|---|---|---|---|---|---|
| extract-full | openai:gpt-5.6-terra | 121 | 120 | 0.992 | 0.98–1.00 |
| extract-full | gpt-oss-120b | 121 | 119 | 0.983 | 0.96–1.00 |
| extract-full | qwen3.8-27b | 121 | 119 | 0.983 | 0.96–1.00 |
| extract-full | gpt-oss-120b | 121 | 117 | 0.967 | 0.93–0.99 |
| extract-full | qwen3.8-27b | 121 | 116 | 0.959 | 0.92–0.99 |
| extract-full | gpt-oss-120b | 121 | 115 | 0.95 | 0.91–0.98 |
| extract-full | glm-4.7-flash | 121 | 114 | 0.942 | 0.90–0.98 |
| extract-full | nemotron-3-super | 121 | 112 | 0.926 | 0.87–0.97 |
| extract-full | qwen3.5-122b-a10b | 121 | 110 | 0.909 | 0.85–0.96 |
| extract-full | gpt-oss-20b | 121 | 106 | 0.876 | 0.81–0.93 |

### extract-full: gpt-oss vs Terra

Latest finished open-weights runs and the latest hosted-API run. `full` = whole filing in context; `section`/`chunked` = oversized-filing fallback (local models only; the hosted model reads every filing whole).

Compared: gpt-oss-120b (`20260912T132250Z-extract-full`), gpt-oss-20b (`20260912T135023Z-extract-full`), openai:gpt-5.6-terra (`20260912T142344Z-extract-full`)


### Accuracy slices

| slice | n | gpt-oss-120b | gpt-oss-20b | openai:gpt-5.6-terra |
|---|---|---|---|---|
| own scored (raw) | - | 119/121 (0.983) | 106/121 (0.876) | 120/121 (0.992) |
| paired (all three answered) | 121 | 119/121 (0.983) | 106/121 (0.876) | 120/121 (0.992) |
| full document (no OSS fallback) | 104 | 103/104 (0.990) | 94/104 (0.904) | 103/104 (0.990) |
| OSS fallback filings | 17 | 16/17 (0.941) | 12/17 (0.706) | 17/17 (1.000) |

### Context mode

| model | fallback | by_mode |
|---|---|---|
| gpt-oss-120b | 17 | full 103/104, section 16/17 |
| gpt-oss-20b | 17 | full 94/104, section 12/17 |
| openai:gpt-5.6-terra | 0 | full 120/121 |

### By tag

| tag | n | gpt-oss-120b | gpt-oss-20b | openai:gpt-5.6-terra |
|---|---|---|---|---|
| Assets | 12 | 12/12 (1.000) | 11/12 (0.917) | 12/12 (1.000) |
| CashAndCashEquivalentsAtCarryingValue | 11 | 11/11 (1.000) | 10/11 (0.909) | 11/11 (1.000) |
| CommonStockSharesOutstanding | 10 | 10/10 (1.000) | 5/10 (0.500) | 9/10 (0.900) |
| EarningsPerShareDiluted | 12 | 12/12 (1.000) | 12/12 (1.000) | 12/12 (1.000) |
| Liabilities | 11 | 11/11 (1.000) | 10/11 (0.909) | 11/11 (1.000) |
| LongTermDebtNoncurrent | 9 | 8/9 (0.889) | 6/9 (0.667) | 9/9 (1.000) |
| NetCashProvidedByUsedInOperatingActivities | 12 | 11/12 (0.917) | 11/12 (0.917) | 12/12 (1.000) |
| NetIncomeLoss | 12 | 12/12 (1.000) | 10/12 (0.833) | 12/12 (1.000) |
| OperatingIncomeLoss | 9 | 9/9 (1.000) | 9/9 (1.000) | 9/9 (1.000) |
| Revenues | 11 | 11/11 (1.000) | 10/11 (0.909) | 11/11 (1.000) |
| StockholdersEquity | 12 | 12/12 (1.000) | 12/12 (1.000) | 12/12 (1.000) |

### By company

| ticker | n | gpt-oss-120b | gpt-oss-20b | openai:gpt-5.6-terra |
|---|---|---|---|---|
| AAPL | 11 | 11/11 (1.000) | 10/11 (0.909) | 11/11 (1.000) |
| CAT | 11 | 10/11 (0.909) | 11/11 (1.000) | 11/11 (1.000) |
| GS | 8 | 7/8 (0.875) | 4/8 (0.500) | 8/8 (1.000) |
| HD | 11 | 11/11 (1.000) | 10/11 (0.909) | 10/11 (0.909) |
| INTU | 11 | 11/11 (1.000) | 10/11 (0.909) | 11/11 (1.000) |
| MSFT | 11 | 11/11 (1.000) | 11/11 (1.000) | 11/11 (1.000) |
| NEE | 11 | 11/11 (1.000) | 9/11 (0.818) | 11/11 (1.000) |
| PG | 9 | 9/9 (1.000) | 8/9 (0.889) | 9/9 (1.000) |
| PLTR | 10 | 10/10 (1.000) | 9/10 (0.900) | 10/10 (1.000) |
| STWD | 9 | 9/9 (1.000) | 8/9 (0.889) | 9/9 (1.000) |
| WMT | 9 | 9/9 (1.000) | 8/9 (0.889) | 9/9 (1.000) |
| XOM | 10 | 10/10 (1.000) | 8/10 (0.800) | 10/10 (1.000) |

### Disagreements

| id | ticker | gpt-oss-120b | gpt-oss-20b | openai:gpt-5.6-terra |
|---|---|---|---|---|
| AAPL:10-K:2025-09-27:CommonStockSharesOutstanding | AAPL | ok | miss | ok |
| CAT:10-K:2025-12-31:LongTermDebtNoncurrent | CAT | miss | ok | ok |
| GS:10-K:2025-12-31:CashAndCashEquivalentsAtCarryingValue | GS | ok/section | miss/section | ok |
| GS:10-K:2025-12-31:CommonStockSharesOutstanding | GS | ok/section | miss/section | ok |
| GS:10-K:2025-12-31:NetCashProvidedByUsedInOperatingActivities | GS | miss/section | miss/section | ok |
| GS:10-K:2025-12-31:NetIncomeLoss | GS | ok/section | miss/section | ok |
| HD:10-K:2026-02-01:CommonStockSharesOutstanding | HD | ok | ok | miss |
| HD:10-K:2026-02-01:NetIncomeLoss | HD | ok | miss | ok |
| INTU:10-K:2026-07-31:CommonStockSharesOutstanding | INTU | ok | miss | ok |
| NEE:10-K:2025-12-31:Assets | NEE | ok | miss | ok |
| NEE:10-K:2025-12-31:Liabilities | NEE | ok | miss | ok |
| PG:10-K:2026-06-30:LongTermDebtNoncurrent | PG | ok | miss | ok |
| PLTR:10-K:2025-12-31:CommonStockSharesOutstanding | PLTR | ok | miss | ok |
| STWD:10-K:2025-12-31:CommonStockSharesOutstanding | STWD | ok/section | miss/section | ok |
| WMT:10-K:2026-01-31:LongTermDebtNoncurrent | WMT | ok | miss | ok |
| XOM:10-K:2025-12-31:LongTermDebtNoncurrent | XOM | ok | miss | ok |
| XOM:10-K:2025-12-31:Revenues | XOM | ok | miss | ok |

## Hardware: same model, different box

Rows are (platform/backend); `paired` scores every run on the items all of them answered; `95% CI` is a percentile bootstrap over per-item verdicts — overlapping intervals mean the runs are not distinguishable at this sample size. Latency includes each box's own prefill, so the gap is the hardware gap. Prompt t/s is blank for LM Studio-served runs (not measurable from its stats).


### gpt-oss-120b extract-full

| platform | score | 95% CI | paired | truncated | ttft p50 s | total p50 s | total p95 s | prompt t/s | decode t/s | ctx | gpu | build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo/rocm | 0.95 | 0.91–0.98 | 115/121 (0.950) | 0 | 1.19 | 7.43 | 238 | - | 19 | 131072 | - | 2.37.0/rocm:lmstudio-2.37.0 |
| halo/vulkan | 0.967 | 0.93–0.99 | 117/121 (0.967) | 0 | 1.38 | 5.15 | 300 | - | 29 | 131072 | - | 2.37.0/vulkan:lmstudio-2.37.0 |
| spark/cuda | 0.983 | 0.96–1.00 | 119/121 (0.983) | 0 | 0.565 | 5.79 | 63 | 250 | 30 | 131072 | NVIDIA GB10 | 82d6bb284d1f/? |

### qwen3.8-27b extract-full

| platform | score | 95% CI | paired | truncated | ttft p50 s | total p50 s | total p95 s | prompt t/s | decode t/s | ctx | gpu | build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halo/vulkan | 0.983 | 0.96–1.00 | 119/121 (0.983) | 0 | 1.44 | 6.21 | 43 | 1,668 | 55 | 262144 | AMD Radeon(TM) Graphics | 82d6bb284d1f/? |
| spark/cuda | 0.959 | 0.92–0.99 | 116/121 (0.959) | 1 | 8.6 | 36 | 196 | 280 | 9.76 | 131072 | NVIDIA GB10 | 82d6bb284d1f/121a-real |
