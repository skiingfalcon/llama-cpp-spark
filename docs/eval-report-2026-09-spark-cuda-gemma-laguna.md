# SEC 10-K extraction — Gemma 4 31B and Laguna S 2.1 on the Spark (September 2026)

Same 121-question `extract-full` suite as the
[Spark re-run](eval-report-2026-09-spark-cuda-rerun.md) (gpt-oss-20b / 120b / Terra), the
[Qwen](eval-report-2026-09-spark-cuda-qwen.md) and [Nemotron](eval-report-2026-09-spark-cuda-nemotron.md)
add-ons. This report adds three runs on the same NVIDIA DGX Spark / CUDA stack:

- **`gemma-4-31b`** (Google Gemma 4 31B-it, dense 30.7B, Google QAT q4_0, 17.65 GB), thinking on
- **`gemma-4-31b-nothink`**, the same weights with `--reasoning off`
- **`laguna-s-2.1`** (Poolside Laguna S 2.1, MoE 118B / ~8B active, unsloth UD-Q4_K_XL, 73.4 GB),
  thinking off, which is the vendor default

| Model | Params (active / total) | Score | 95% CI | Fits, of 104 | Truncated | Decode | Wall clock |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.6-terra` (OpenAI API) | hosted | 99.2% (120/121) | 0.98–1.00 | 103/104 | 0 | — | 20 min |
| `gpt-oss-120b` | ~5.1B / 117B MoE | **98.3% (119/121)** | 0.96–1.00 | 103/104 | 0 | **30 t/s** | **26 min** |
| `deepseek-v4-flash-nothink` | ~13B / 284B MoE, 2-bit | 98.3% (119/121) | 0.96–1.00 | 102/104 | 0 | 16 t/s | 51 min |
| **`gemma-4-31b`, thinking on** | **30.7B dense** | **96.7% (117/121)** | **0.93–0.99** | **104/104** | **0** | **7.9 t/s** | **130 min** |
| `qwen3.8-27b` | 27B dense | 95.9% (116/121) | 0.92–0.99 | 104/104 | 1 | 9.8 t/s | 125 min |
| **`gemma-4-31b-nothink`** | **30.7B dense** | **95.0% (115/121)** | **0.91–0.98** | **102/104** | **0** | **7.8 t/s** | **61 min** |
| **`laguna-s-2.1`, thinking off** | **~8B / 118B MoE** | **95.0% (115/121)** | **0.91–0.98** | **102/104** | **0** | **17.2 t/s** | **29 min** |
| `gpt-oss-20b` | ~3.6B / 21B MoE | 87.6% (106/121) | 0.81–0.93 | 94/104 | 0 | 45 t/s | 19 min |

**Headline:** neither model moves the on-prem default. Gemma with thinking on is the most accurate
dense model on the Spark and reads every filing that fits the window correctly, but at 7.9 t/s it is
the slowest model here and its own tokenizer pushed Goldman into BM25 chunks, where all four of its
misses live. Laguna is the story on speed: 29 minutes for the suite, three behind gpt-oss-120b,
at four questions' cost. Both models have fatter tokenizers than gpt-oss, and that, not model
quality, decides what falls back at 131K.

## Setup

- **Task / corpus / tags / scoring:** identical to the re-run (`scoring_version` `2026-09-13.1`,
  0.5% relative tolerance, 12 companies, latest 10-K, 11 XBRL tags).
- **Decoding:** `temperature=0`, `seed=42`, `max_tokens=4096`.
- **Hardware / build:** NVIDIA GB10, llama.cpp `82d6bb284d1f` (`b869`, `121a-real`), `--jinja`,
  `--n-gpu-layers 999`, `n_ctx_per_slot` 131,072, batch/ubatch 2048.
- **Gemma:** `google/gemma-4-31B-it-qat-q4_0-gguf`, file `gemma-4-31B_q4_0-it.gguf`; the `-nothink`
  twin adds `--reasoning off`. Served template sha `6a1015c47ccf`.
- **Laguna:** `unsloth/Laguna-S-2.1-GGUF` `UD-Q4_K_XL` (3 shards, 73.4 GB). Thinking is off by the
  vendor's default, so the bare entry is the primary row; `laguna-s-2.1-thinking` (not run yet)
  adds `--chat-template-kwargs {"enable_thinking": true}`. Served template sha `444819b8ad46`.
- **Artifacts:**
  - Gemma, thinking on: [`state/evals/sec/gemma-4-31b-spark-cuda/20260919T174420Z-extract-full/`](../state/evals/sec/gemma-4-31b-spark-cuda/20260919T174420Z-extract-full/)
  - Gemma, thinking off: [`state/evals/sec/gemma-4-31b-nothink-spark-cuda/20260919T202304Z-extract-full/`](../state/evals/sec/gemma-4-31b-nothink-spark-cuda/20260919T202304Z-extract-full/)
  - Laguna: [`state/evals/sec/laguna-s-2.1-spark-cuda/20260920T003059Z-extract-full/`](../state/evals/sec/laguna-s-2.1-spark-cuda/20260920T003059Z-extract-full/)
  - Baseline `gpt-oss-120b`: [`state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full/`](../state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full/)

The machine-generated table is [`state/evals/report-sec.md`](../state/evals/report-sec.md).

## Results

### Accuracy by context mode

The harness shows the model the whole filing if it fits the budget, else the financial-statements
Item 8, else the BM25 top-6 chunks of 8,000 tokens (`ContextBuilder` in `evals/sec/run.py`).

| Slice | 120b | Qwen | **Gemma on** | **Gemma off** | **Laguna** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full document | 103/104 | 93/93 | **93/93** | **91/93** | **92/93** |
| Item 8 section | 16/17 | 20/20 | **20/20** | **20/20** | **19/20** |
| BM25 chunks | — | 3/8 | **4/8** | **4/8** | **4/8** |
| **All questions** | **119/121** | **116/121** | **117/121** | **115/121** | **115/121** |

Gemma and Laguna fall back on the same 28 questions Qwen and Nemotron did: NEE (11) and STWD (9)
on the Item 8 path, GS (8) on chunks. gpt-oss-120b falls back on only 17, because its tokenizer
keeps NEE whole and GS's Item 8 inside the budget.

### Tokenizer size decides the fallback shape

| Filing | gpt-oss tokens | Gemma tokens | Laguna tokens | Budget at 131K |
| --- | ---: | ---: | ---: | --- |
| GS | 242,026 | 271,429 (+12%) | 274,903 (+14%) | Item 8 fits under gpt-oss only |
| STWD | 176,354 | 200,441 (+14%) | 203,768 (+16%) | Item 8 under all three |
| NEE | 124,130 | 136,728 (+10%) | 138,457 (+12%) | Whole under gpt-oss; Item 8 under Gemma/Laguna |

Gemma's card predicted the opposite: a 262,144-token vocabulary was expected to make filings
cheaper. On 10-K text it did not. Both Gemma and Laguna cost 10–16% more tokens per filing than
gpt-oss, the same effect that pushed Qwen and Nemotron onto the fallback path.

### The misses

**Gemma, thinking on (4):** all Goldman, all on BM25 chunks. Operating cash flow, total liabilities
and shares outstanding came back `unknown`; stockholders' equity came back 123.733B against
124.972B, 1% off.

**Gemma, thinking off (6):** the same four Goldman questions, now with wrong numbers instead of
`unknown` on two (operating cash flow 17.007B against −45.154B, the same wrong figure gpt-oss-120b
gave on the section path; liabilities 512.3B against 1,684.3B), plus MSFT shares outstanding
(`unknown`) and PLTR net income (1,634.6M against 1,625.0M, 0.6% high, just outside tolerance).

**Laguna, thinking off (6):** four Goldman questions on chunks (net income, assets, liabilities,
equity; three `unknown`, and on net income the model wrote a 608-token prose search, "I need to
find the net income...", and never produced a number), XOM revenues (323.9B, the sales line, against
332.2B total revenues and other income), and NEE stockholders' equity on the Item 8 path (66.5B,
total equity including noncontrolling interests, against 54.6B).

Against the gpt-oss-120b baseline, all three runs got Caterpillar's long-term debt right where
120b missed it, and Laguna also got Goldman's operating cash flow right where 120b missed it. The
other differences all go the baseline's way.

### Thinking on vs off, Gemma

| | Thinking on | Thinking off | Difference |
| --- | ---: | ---: | ---: |
| Score | 117/121 | 115/121 | +2 |
| Reasoning tokens | 32,696 (median 202 / question) | 0 | |
| Completion tokens, median | 216 | 12 | |
| Truncated | 0 | 0 | |
| Wall clock | 130 min | 61 min | +69 min |
| Decode | 7.9 t/s | 7.8 t/s | |

The 69 minutes are the reasoning tokens: 32,696 at 7.9 t/s is 69 minutes. Thinking recovered MSFT
shares and PLTR net income and changed nothing on Goldman. This is the first pair here where
thinking on scored higher than off (Qwen on coding and Nemotron went the other way), but the gain is
two questions on one seed inside overlapping intervals (0.93–0.99 vs 0.91–0.98).

### Speed

| Model | Active params, bits | Bytes / token (weights) | Decode | Cold prefill | Cold TTFT, 80K+ filings | Warm re-prefill gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-oss-120b | 5.1B × 4.25 | 2.7 GB | 30 t/s | 1,354 t/s | 57–101 s | 63 tokens |
| Laguna S 2.1 | ~8B × ~5.0 | ~5.0 GB | 17.2 t/s | 816 t/s | 97–157 s | 86 tokens |
| DeepSeek-V4-Flash-nothink | ~13B × ~2–2.5 | 3.3–4.5 GB | 16.0 t/s | 422 t/s | 190–306 s | — |
| Gemma 4 31B | 30.7B × ~4.6 | ~17.6 GB | 7.9 t/s | 367 t/s | 202–356 s | 86 tokens |

Laguna decodes where the bandwidth rule puts it: 2.7 / 5.0 of gpt-oss-120b's 30 t/s is 16.
Gemma is below the ~10 t/s its card predicted and below qwen3.8-27b's 9.8. The predicted
sliding-window re-prefill of one micro-batch per warm request (~2,028 tokens, as Qwen3.8 shows) did
not happen on either model: median warm gap 86 tokens, cache reuse 86.5–86.8%.

Laguna's 29-minute wall clock comes from three things at once: no reasoning, tiny completions
(median 13 tokens), and the second-fastest cold prefill on the box.

## What changes

- **On-prem default:** unchanged, `gpt-oss-120b`. Gemma trades 5x wall clock for two fewer correct
  answers; Laguna trades four answers for three minutes.
- **Dense slot:** Gemma 4 31B with thinking on is now the most accurate dense model on the Spark
  (117 vs Qwen's 116), at the same speed class. Both read every fitting filing correctly.
- **Laguna** is the fast second option and the first model here built for the SWE suite. Its
  extraction misses are two wrong-line accounting picks and the Goldman chunks; the coding suite is
  where it should be judged.

## Next runs, in order

1. `laguna-s-2.1-thinking` at 131K: does thinking fix the XOM and NEE line picks, or loop?
2. Laguna on SWE tier 1, thinking off then on: the reason it was registered.
3. Gemma and Laguna at their native windows (Gemma 262K, Laguna larger): does GS's Item 8 fit and do
   the four chunk misses close, as they did for Qwen at 262K on the 5090?
4. Gemma's MTP drafter and Laguna's DFlash drafter, once the parked speed plan resumes.

## Caveats and review notes

- Single run per row, one seed. Gemma-on vs Gemma-off vs Laguna are all inside overlapping 95%
  intervals; gpt-oss-120b's interval (0.96–1.00) touches Laguna's and Gemma-off's (0.91–0.98).
- The three models did not see the same context: gpt-oss-120b read NEE whole and GS's Item 8;
  Gemma and Laguna read NEE's Item 8 and GS's chunks. "Fits, of 104" (GS and STWD excluded) is the
  column that compares tokenizers fairly, and there Gemma-on is 104/104.
- No memorisation control (same questions, no document) has been run for any model.
- All four servers (both Gemma twins, Laguna, the gpt-oss-120b baseline) ran with 4 slots; the
  eval used one at a time.
- Laguna's checkpoint was registered as ~40 GB and is 73.4 GB; the registry comments and card were
  corrected in this commit.
