# Ternary Bonsai 2 27B on an RTX 5090 — SEC 10-K extraction, FinanceBench, coding tier 1 (September 2026)

PrismML's **Ternary Bonsai 2 27B** is Qwen3.8-27B re-expressed in ternary weights (-1, 0, +1 with
FP16 group scales), 1.72 bits per weight, shipped as a 7.2 GB GGUF against 17.6 GB for the
Unsloth UD-Q4_K_XL Qwen we measured on 2026-09-15. Same architecture, same tokenizer, same 262K
hybrid-attention backbone, thinking on by default. PrismML claims 98.2% of the FP16 model's
aggregate benchmark score; this report asks the narrower question we can answer: **on our
tasks, on the same card, what does 2.4x less weight memory cost?**

Same suites, corpus, scoring and hardware as the two RTX 5090 Qwen reports
([extraction](eval-report-2026-09-rtx5090-cuda-qwen.md),
[FinanceBench / tier 1](eval-report-2026-09-rtx5090-financebench-swe-qwen.md)). One difference in
the stack: the Bonsai GGUF packs (`PQ2_0`, `PTQ1_0`) need PrismML's llama.cpp fork; stock llama.cpp
rejects the tensor types. We ran their prebuilt Windows CUDA 13.3 release `prism-b10685-7dffb158d`
(upstream base b10685, five builds behind our b10919 pin) selected per run through
`LOCAL_LLM_LLAMA_BIN_DIR`.

| Stack | Accuracy (121 Qs) | When the filing fits (of 104 Qs) | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| RTX 5090, Qwen3.8-27B (dense), 262K ctx + q8_0 KV | 98.3% | 102/104 | 55 t/s | 40–70 s | 24 min | $0 marginal |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| **RTX 5090, Bonsai 2 27B (ternary), 262K ctx, f16 KV** | **96.7%** | **100/104** | **93 t/s** | **37–64 s** | **20 min** | **$0 marginal** |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 131K ctx | 94.2% | 102/104 | 60 t/s | 38–54 s | 23 min | $0 marginal |
| **RTX 5090, Bonsai 2 27B (ternary), 131K ctx** | **92.6%** | **101/104** | **98 t/s** | **37–53 s** | **21 min** | **$0 marginal** |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | — | 40–60 s | 19 min | $0 marginal |

Single run per row. Percentile-bootstrap 95% intervals (the harness's `bootstrap_ci` over
`results.jsonl`): Bonsai 262K 0.93–0.99, Bonsai 131K 0.88–0.97, Qwen 262K 0.96–1.00, Qwen 131K
0.90–0.98. Every local interval overlaps its neighbours; read gaps of one to three questions as
noise. "When the filing fits" is scored on the 10 filings gpt-oss read whole (104 questions).
"Cold prefill" is time to first token on the first question of each 90–140K-token filing.

## Setup

- **Task / corpus / tags / scoring:** identical to the Qwen reports (`scoring_version 2026-09-13.1`,
  0.5% relative tolerance, 12 companies, latest 10-K, 11 XBRL tags). Same corpus files as
  2026-09-15; Bonsai's token counts match Qwen's exactly (same tokenizer: AAPL 51,383, GS 273,285,
  GS Item 8 131,589).
- **Decoding:** `temperature=0`, `seed=42`, `max_tokens=4096`, serial requests. Reasoning left at
  the model default (PrismML documents `xhigh` effort as the default).
- **Model:** `prism-ml/Ternary-Bonsai-2-27B-gguf`, file `Ternary-Bonsai-2-27B-PQ2_0.gguf`
  (7.21 GB, 2.13 bits/weight as packed; PrismML's faster pack on Blackwell and for prompt
  processing). Registered as `bonsai-2-27b` / `bonsai-2-27b-nothink` on port 8088 in both
  registries. Vision tower (`mmproj`) not loaded.
- **Server:** PrismML fork `prism-b10685-7dffb158d`, Windows CUDA 13.3 zip + cudart. Flags from
  the registry: `--n-gpu-layers 999 --flash-attn on --batch-size 2048 --ubatch-size 2048 --jinja`,
  4 auto slots with unified KV.
  - Run 1: `--ctx-size 131072`, f16 KV. ~21 GB VRAM in use (observed).
  - Run 2: `--ctx-size 262144`, **f16 KV** (no cache quantisation needed at 7 GB of weights).
    ~30.5 GB VRAM in use (observed). Qwen's 262K row needed q8_0 K/V to fit; this one does not.
- **CLI note:** as with the Qwen runs, this Windows box runs the `halo` platform code, so runs
  are filed under `bonsai-2-27b-halo-vulkan`, `run.json` says platform halo / backend vulkan / an
  AMD iGPU, and `runtime.ctx_size` echoes the registry's 131072 in run 2. Authoritative fields:
  `server.build_info` = `b10685-7dffb158d` (the fork), `server.n_ctx_per_slot`.
- **Artifacts:**
  - Run 1 (131K): [`state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T152253Z-extract-full/`](../state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T152253Z-extract-full/)
  - Run 2 (262K): [`state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T154449Z-extract-full/`](../state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T154449Z-extract-full/)
  - FinanceBench: [`state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T160529Z-qa-financebench/`](../state/evals/sec/bonsai-2-27b-halo-vulkan/20260918T160529Z-qa-financebench/)
  - Tier 1: see the [coding section](#coding-tier-1).

## Extraction results

### Accuracy by context mode

| Slice | Qwen 131K | **Bonsai 131K** | Qwen 262K | **Bonsai 262K** |
| --- | ---: | ---: | ---: | ---: |
| Full document | 91/93 | **90/93** | 111/113 | **109/113** |
| Item 8 section fallback | 20/20 | **20/20** | 8/8 (GS) | **8/8 (GS)** |
| BM25 chunk fallback (GS) | 3/8 | **2/8** | — | — |
| **All questions** | 114/121 | **112/121 (92.6%)** | 119/121 | **117/121 (96.7%)** |
| Truncated at 4,096 tokens | 1 | **5** | 0 | **2** |

- **Same shape as the full-precision model.** GS falls to chunks at 131K and is recovered at 262K
  (8/8 in section mode, Item 8 = 131,589 tokens); NEE and STWD read whole at 262K. Bonsai's
  section-mode answers are perfect in both runs, like Qwen's.
- **Two to three questions behind Qwen at each setting**, inside overlapping intervals. Where the
  two differ item by item, Bonsai gets the MSFT and HD share counts Qwen missed; Qwen gets the
  XOM revenue line, the NEE share count, and the two questions Bonsai spent its whole budget
  thinking about.
- **Truncations are the ternary model's distinctive failure.** Median hidden reasoning is
  *shorter* than Qwen's (202 vs 244 tokens at 131K; 188 vs 224 at 262K) but the tail is fatter:
  5 questions ran to the 4,096 cap at 131K (3 of them GS chunks) and 2 at 262K (PLTR share
  count, PG equity), against 1 and 0 for Qwen. Every truncation is `reasoning_tokens = 4096`,
  content empty.
- **One genuine wrong number, repeated in both runs:** XOM revenues 323.9B ("sales and other
  operating revenue") instead of 332.2B (total revenues), 2.5% off, outside the 0.5% tolerance.
  Qwen returned the total both times. No `off_by_scale` errors in either run.

### Per-tag accuracy

| Tag | Qwen 131K | **Bonsai 131K** | Qwen 262K | **Bonsai 262K** |
| --- | ---: | ---: | ---: | ---: |
| Assets | 11/12 | **11/12** | 12/12 | **12/12** |
| CashAndCashEquivalentsAtCarryingValue | 11/11 | **11/11** | 11/11 | **11/11** |
| CommonStockSharesOutstanding | 7/10 | **8/10** | 8/10 | **8/10** |
| EarningsPerShareDiluted | 12/12 | **12/12** | 12/12 | **12/12** |
| Liabilities | 10/11 | **10/11** | 11/11 | **11/11** |
| LongTermDebtNoncurrent | 9/9 | **9/9** | 9/9 | **9/9** |
| NetCashProvidedByUsedInOperatingActivities | 11/12 | **11/12** | 12/12 | **12/12** |
| NetIncomeLoss | 12/12 | **11/12** | 12/12 | **12/12** |
| OperatingIncomeLoss | 9/9 | **9/9** | 9/9 | **9/9** |
| Revenues | 11/11 | **10/11** | 11/11 | **10/11** |
| StockholdersEquity | 11/12 | **10/12** | 12/12 | **11/12** |

### Per-company accuracy

| Ticker | **Bonsai 131K** | **Bonsai 262K** | Mode (131K / 262K) |
| --- | ---: | ---: | --- |
| AAPL | 11/11 | 11/11 | full / full |
| CAT | 11/11 | 11/11 | full / full |
| GS | **2/8** | **8/8** | chunked / **section** |
| HD | 11/11 | 11/11 | full / full |
| INTU | 11/11 | 11/11 | full / full |
| MSFT | 11/11 | 11/11 | full / full |
| NEE | 11/11 | **10/11** | section / **full** |
| PG | 8/9 | 8/9 | full / full |
| PLTR | 9/10 | 9/10 | full / full |
| STWD | 9/9 | 9/9 | section / **full** |
| WMT | 9/9 | 9/9 | full / full |
| XOM | 9/10 | 9/10 | full / full |

### Every miss, itemized

**Run 1 (131K), 9 misses.** GS in chunked mode 6/8 wrong: net income, operating cash flow and
assets truncated (4,096 tokens of reasoning, no answer); liabilities 425.7B (the same GS subtotal
Qwen and both gpt-oss models return from chunks); equity `unknown`; shares 307.1M vs 296.5M
(3.6% off). Plus XOM revenues 323.9B vs 332.2B, and two thinking overruns on whole documents:
PLTR share count and PG stockholders' equity.

**Run 2 (262K), 4 misses.** XOM revenues (same wrong line), PLTR share count and PG equity (same
two overruns, reproduced exactly under greedy decoding), and NEE share count `unknown` with the
whole filing in context. GS is 8/8.

## Speed

| Phase | Qwen 131K | **Bonsai 131K** | Qwen 262K | **Bonsai 262K** |
| --- | ---: | ---: | ---: | ---: |
| Prefill (TTFT summed) | 10.6 min | **10.1 min** | 12.7 min | **11.6 min** |
| Decode (completion) | 11.9 min | **10.5 min** | 11.6 min | **8.1 min** |
| **Wall clock** | 22.5 min | **20.6 min** | 24.3 min | **19.8 min** |
| Decode t/s (p50) | 60 | **98** | 55 | **93** |
| Cold prompt t/s | 2,300–3,200 | **2,400–3,500** | 1,500–3,100 | **1,700–3,500** |

- **Decode is 1.6–1.7x faster, not 2.4x.** Weight traffic per token drops 2.4x (17.6 → 7.2 GB)
  but decode only goes from 55–60 to 93–98 t/s. PrismML's own notes say batch-1 decode on
  Blackwell is bound by instruction throughput and launch overhead rather than memory, and the
  gap between "bytes moved" and "tokens out" here is consistent with that.
- **Prefill is unchanged.** Prompt processing is compute-bound, so a smaller weight footprint
  buys little; cold throughput is within a few percent of Qwen's on every filing.
- **Wall clock gain is modest, 2–4 minutes per suite**, because the suite is half prefill and
  because Bonsai spends more of its decode budget on reasoning tails (58k reasoning tokens at
  131K vs Qwen's 42k).
- **The footprint is the real win:** ~21 GB VRAM for 131K with f16 KV, ~30.5 GB for 262K with f16
  KV, on a card where the 4-bit Qwen needed q8_0 KV to fit 262K at all. A 24 GB card could serve
  Bonsai at 131K; it cannot serve the Q4 Qwen at anything like that window.

## FinanceBench

Same task and scorer caveats as the [Qwen FinanceBench report](eval-report-2026-09-rtx5090-financebench-swe-qwen.md#financebench):
150 analyst questions over human-selected excerpts, 74 with a numeric reference, 76 free-text
that need the judge model (`gpt-oss-120b`), which this card cannot hold. The harness's raw
numeric score takes the first number in a sentence answer and is not meaningful; the
unit-tolerant any-number rescoring is reported alongside it.

| Slice | n | Qwen3.8-27B (Q4) | **Bonsai 2** |
| --- | ---: | ---: | ---: |
| Numeric, harness raw | 74 | 7 (9.5%) | **8 (10.8%)** |
| Numeric, unit-tolerant any-number | 74 | 60 (81.1%), CI 0.72–0.89 | **59 (79.7%), CI 0.70–0.89** |
| of which `metrics-generated` (pure calculation) | 50 | 46 (92%) | **45 (90%)** |
| Free-text (need the judge) | 76 | unscored | unscored |
| Abstained (`unknown` or "does not state/contain/provide…") | 150 | 20 | **20** |
| Truncated at 4,096 tokens | 150 | 2 | **5** |
| Hidden reasoning, total / p50 / p90 tokens | | 81k / 278 / 1,543 | **113k / 352 / 2,100** |
| Decode (p50) | | 76 t/s | **141 t/s** |
| Wall clock | | 20.6 min | **15.3 min** |

One question apart on the numeric half, well inside the interval. The lenient misses are the
same population as Qwen's: about eight free-text questions bucketed as numeric because the
reference contains one number, and about six genuine errors (the dividends-total-for-per-share
misread, two ROA denominators, the EBITDA margin, the Pfizer separation cost, and an operating
margin Qwen got). Bonsai thinks ~40% more on these questions than the 4-bit Qwen and hits the
4,096 cap on five of them (Qwen: two), the same tail behaviour as on extraction.

## Coding tier 1

Reasoning off (`--reasoning off`), evalplus 0.3.1 greedy pass@1 at its default 768-token
budget: the same configuration as the Qwen reasoning-off row. Server and harness both ran
inside WSL2 for this suite (see the harness notes below).

| Dataset | Qwen3.8-27B (Q4), reasoning off | **Bonsai 2, reasoning off** |
| --- | ---: | ---: |
| HumanEval base pass@1 | 93.3% (153/164) | **91.5%** (150/164) |
| HumanEval+ pass@1 | 90.9% (149/164) | **86.6%** (142/164), CI 0.81–0.91 |
| MBPP base pass@1 | 89.2% (337/378) | **86.2%** (326/378) |
| MBPP+ pass@1 | 76.5% (289/378) | **74.1%** (280/378), CI 0.70–0.79 |
| Empty solutions | 0 / 542 | **0 / 542** |
| Wall clock | ~26 min | **22.2 min** |

Pass counts follow evalplus's printed pass@1 (plus requires base); the harness's plus-only
`correct` field gives 282/378 on MBPP. Item by item against the Qwen run, Bonsai solved 14
problems Qwen missed and missed 30 Qwen solved: a net 16 problems (3 points) on the plus tests.
No truncations in either model with reasoning off.

**Reasoning on was not run for Bonsai** (stopped at the user's request after the reasoning-off
pass; GPU time). Given the extraction and FinanceBench truncation counts, the expectation is
the same blank-answer failure Qwen showed at 4,096, probably more of it; that is a prediction,
not a measurement. Artifacts:
[`state/evals/swe/bonsai-2-27b-halo-vulkan/20260918T164442Z-tier1/`](../state/evals/swe/bonsai-2-27b-halo-vulkan/20260918T164442Z-tier1/),
with `evalplus/` holding the per-problem result files and `empty_solutions.json` (both empty lists).

## What this says

- **The ternary model keeps the base model's behaviour, at a small and consistent cost.** Two
  questions behind Qwen on extraction at both context settings (112 vs 114, 117 vs 119), one
  behind on FinanceBench's numeric half (59 vs 60), three points behind on coding (HumanEval+
  86.6 vs 90.9, MBPP+ 74.1 vs 76.5). Every gap is inside overlapping intervals on its own, but
  the sign is the same on every suite. Where the model can see the document it behaves like
  Qwen: GS recovered at 262K, section-mode answers perfect.
- **Its distinctive weakness is the thinking tail.** Median reasoning is shorter than Qwen's, yet
  more questions run to the 4,096 cap: 5 vs 1 at 131K, 2 vs 0 at 262K, 5 vs 2 on FinanceBench.
  Serve it with reasoning off or a larger budget for short-answer work.
- **Speed: 1.6–1.7x decode, unchanged prefill, 2–4 minutes off a 20-minute suite.** Not the 2.4x
  the weight reduction suggests; PrismML's own notes attribute Blackwell batch-1 decode to
  instruction throughput rather than bandwidth.
- **Memory is the real gain.** 7.2 GB of weights: 131K context in ~21 GB, 262K in ~30.5 GB with
  plain f16 KV. This is the first 27B-class model in these reports that a 24 GB card could serve
  at the standard 131K budget, and it does so at 92.6% on extraction.
- **Engine caveat.** Every Bonsai number here comes from PrismML's fork, not the pinned upstream
  build. Reproducing on the Spark means building that fork.

## Harness and ops notes

- **Fork as a second binary dir.** `LOCAL_LLM_LLAMA_BIN_DIR` pointed at the fork for Bonsai runs
  and at b10919 for everything else. `run.json` records the fork through `server.build_info`
  (`b10685-7dffb158d`); the registry comment says which engine each model needs.
- **Serving from inside WSL.** The WSL-to-Windows loopback used on 2026-09-15 stopped working
  after the host's network adapters changed, in both mirrored and NAT modes. Tier 1 was run with
  the fork's Linux CUDA build inside WSL instead (its `libgomp`, `libcudart` and `libcublas`
  staged next to the binary from a Ubuntu package and the NVIDIA PyPI wheels; no root needed).
  Two harness gaps surfaced: the Halo platform hard-codes `llama-server.exe` (a symlink of that
  name to the Linux binary satisfies it), and it passes no `setsid` on Linux, so a server started
  from an interactive WSL session dies with the session (start `local-llm serve` under `setsid
  nohup`, or add `start_new_session=True` to `popen_kwargs` off Windows).
- **Windows Smart App Control** (enforce mode on this box since 2026-09-18) blocks the unsigned
  `_ssl` module of uv's managed Python, which breaks `local-llm download` and `eval`. The runs
  used the signed system Python 3.14 via `UV_PYTHON`; `.python-version` was left at 3.12.
- The `%LOCALAPPDATA%` seen inside the Claude desktop app's sandbox resolves to the app's package
  cache (`AppData\Local\Packages\Claude_…\LocalCache\Local`), so models downloaded from a session
  live there, not in the real `%LOCALAPPDATA%\local-llm\models`. Set `LOCAL_LLM_MODELS_DIR` in
  `.env` or move the files if the harness is also driven from a normal shell.

## Caveats

> **Review notes (2026-09-18).** Single run per row. Bonsai runs used PrismML's llama.cpp fork at
> upstream b10685, Qwen runs stock b10919; both are September 2026 builds of the same code base,
> but kernels differ by construction (custom ternary GEMM/GEMV) so speed comparisons are
> engine-plus-format, not format alone. Run 2 used f16 KV where Qwen's 262K row used q8_0, a
> second variable in that comparison, chosen because Bonsai does not need the quantised cache to
> fit. The XOM revenue error and the two whole-document truncations reproduce across both runs
> under greedy decoding, so they are properties of this model at this budget, not noise. No
> `--no-document` control.

## Reproduce

```powershell
# Windows, PowerShell, repo root. PrismML fork zip + cudart extracted to C:\llama\prism-b10685-cuda.
$env:PYTHONUTF8 = "1"
$env:LOCAL_LLM_LLAMA_BIN_DIR = "C:\llama\prism-b10685-cuda"     # overrides the .env b10919 path for this model
uv run local-llm download bonsai-2-27b
uv run local-llm serve bonsai-2-27b                              # 131K
uv run local-llm eval sec run bonsai-2-27b --task extract-full --forms 10-K
uv run local-llm stop
uv run local-llm serve bonsai-2-27b --ctx-size 262144            # f16 KV fits: 7 GB weights
uv run local-llm eval sec run bonsai-2-27b --task extract-full --forms 10-K
uv run local-llm stop
```
