| model | task | n | score | skipped | truncated | ttft p50 s | total p50 s | prompt t/s | decode t/s | tokens | cached | reasoning | ctx | build | platform | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai:gpt-5.6-terra | extract-full | 121 | 0.992 | 0 | 0 | 1.44 | 1.55 | - | - | 12394653 | 0 | 1260 | 1050000 | openai/api | api | - |
| gpt-oss-120b | extract-full | 121 | 0.983 | 0 | 0 | 0.565 | 5.79 | 250 | 30 | 10542592 | 9466001 | 20720 | 131072 | 82d6bb284d1f/? | spark/cuda | - |
| gpt-oss-20b | extract-full | 121 | 0.876 | 0 | 1 | 0.39 | 3.59 | 564 | 45 | 10550003 | 9466000 | 28150 | 131072 | 82d6bb284d1f/? | spark/cuda | - |

> warning: task extract-full: runs use different configs (9466ddd63b34, 9cc51d84df5a, ea71250ebf92)

### Paired (items answered by every run of the task)

| task | model | paired n | correct | paired score |
|---|---|---|---|---|
| extract-full | openai:gpt-5.6-terra | 121 | 120 | 0.992 |
| extract-full | gpt-oss-120b | 121 | 119 | 0.983 |
| extract-full | gpt-oss-20b | 121 | 106 | 0.876 |

### extract-full: gpt-oss vs Terra

Latest finished gpt-oss-20b, gpt-oss-120b, and OpenAI/Terra runs. `full` = whole filing in context; `section`/`chunked` = oversized-filing fallback. Terra's 1.05M window still sees GS/STWD in full.

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
