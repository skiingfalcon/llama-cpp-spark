# Email draft: Nemotron-3-Super on the Spark (2026-09-17)

> Internal team email. Every number traces to a run directory under `state/evals/` or to
> `docs/eval-report-2026-09-spark-cuda-nemotron.md`. Drafted with the `model-run-email` skill.

---

Good morning,

Two more overnight runs on the Spark, this time with NVIDIA's Nemotron 3 Super. Nemotron is less
well known than gpt-oss or Qwen, so a short introduction first.

**What Nemotron 3 Super is.** NVIDIA released it in March 2026 as open weights under the NVIDIA
Open Model License. It is a 120B-parameter mixture-of-experts model that activates about 12B
parameters per token, but with an unusual design: most of its layers are Mamba-2 state-space
layers rather than attention, with only eight attention layers in the whole stack. That is what
gives it a native 1M-token context window at almost no memory cost, because only those eight
layers keep a KV cache. It is a reasoning model that thinks in hidden `<think>` tokens before
answering. We tried it for one reason: it is the only local model whose window swallows every
10-K in our set whole, including Goldman Sachs at 274K tokens and Starwood at 204K, which every
other local model has to read in pieces. The 4-bit weights are 70 GB, so it fits the Spark but
not the 32 GB card.

Same task as before: 11 financial metrics from the latest 10-K of 12 companies, 121 questions,
scored against SEC XBRL ground truth at 0.5% tolerance. Single run per row. I ran Nemotron twice
on purpose: once capped at the same 131K input budget the gpt-oss and Qwen runs had, so the
comparison is fair, and once with the full 1M window and no cap, so every filing was read whole.

**What this says**

- **The 1M window is real, and it did not help.** With the full window the harness fell back on
  zero questions, the first time any local model has read all 12 filings whole. The score went
  from 113/121 capped to 112/121 unconstrained. Reading Goldman whole recovered its balance-sheet
  lines and lost its net income; NextEra, which was perfect from its financial-statements
  section, lost two questions when it was read whole. More context moved the misses around, it
  did not remove them.
- **The real problem is that it thinks too long.** In both runs six answers came back empty
  because the model spent the entire 4,096-token completion budget reasoning and never wrote a
  number. It emitted about 131K hidden reasoning tokens per run, six times gpt-oss-120b's 21K, and
  its median answer is five times longer. Take those six truncations out and it is 112 of 115,
  in the same band as the other local models. The lever to test next is a thinking cap, not a
  bigger window.
- **Speed sits where the architecture predicts.** Decode is 20 tokens per second, between dense
  Qwen at 10 and gpt-oss-120b at 30, because it reads about 12B active parameters per token. But
  six times the reasoning means six times the wall clock: about 150 minutes per run against 26 for
  gpt-oss-120b. Reading Goldman whole took six and a half minutes of prefill for the first question.
- **Its real mistakes are the familiar ones.** Exxon revenue (it picked the net-sales line),
  Palantir share count (a different date's figure), and Goldman net income in the full run. No
  scale errors, 20 of 20 on financial-statement sections under the cap.

**Recommendation** for document extraction is unchanged: gpt-oss-120b stays the on-prem default.
Nemotron's 95% interval barely touches 120b's, so this is close to a real gap and not noise, but
it is still one seed. Nemotron is the model to reach for only when a document genuinely exceeds
131K tokens and must be read whole, and even then the thinking cap should be tested first. Two
short runs settle it: the capped run again with `--reasoning-budget 2048` on the server, and the
same with the completion budget raised to 8,192. If the six blanks turn into six right answers,
Nemotron is at 118 of 121 and worth another look. If they turn into six wrong answers, we stop.

This is the second time this week that hidden reasoning has cost a model its score. On the RTX
5090, [colleague]'s Qwen coding runs, now merged and linked below, scored better with thinking
turned off than on, because one problem in eight spent its whole budget thinking. Reasoning
budgets need to be part of the serving configuration we compare, not an afterthought.

| Stack | Accuracy, 121 Qs | When the filing fits, of 104 Qs | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 262K ctx | 98.3% | 102/104 | 55 t/s | 40–70 s | 24 min | $0 marginal |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 131K ctx | 94.2% | 102/104 | 60 t/s | 38–54 s | 23 min | $0 marginal |
| **DGX Spark, Nemotron-3-Super, 131K budget** | **93.4%** | **99/104** | **20 t/s** | **105–160 s** | **148 min** | $0 marginal |
| **DGX Spark, Nemotron-3-Super, full 1M window** | **92.6%** | **97/104** | **19 t/s** | **105–160 s; GS whole 6.5 min** | **153 min** | $0 marginal |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | 45 t/s | 40–60 s | 19 min | $0 marginal |

"When the filing fits" counts the 104 questions on the ten filings that fit a 131K window, so
the column is comparable across tokenizers. Nemotron's tokenizer is about 13% fatter than
gpt-oss's, the same effect that pushed Qwen to the fallback path.

Two housekeeping notes: the machine-generated report shows only the latest Nemotron
configuration, so both run folders are kept on disk and the write-up covers both. The
memorisation control, same questions with no document, is still the outstanding item before any
of this goes outside the company.

**Write-ups**

- Nemotron-3-Super on the Spark: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-spark-cuda-nemotron.md
- Nemotron model card, with the configuration used and the open questions: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/models/nemotron-3-super.md
- Qwen on the RTX 5090, extraction and coding: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-rtx5090-cuda-qwen.md and https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/eval-report-2026-09-rtx5090-financebench-swe-qwen.md
- Which report covers what: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/docs/index.md
- Machine-generated table over all runs, with intervals: https://github.com/skiingfalcon/llama-cpp-spark/blob/master/state/evals/report-sec.md

---

## Fact trail

| Claim | Source |
| --- | --- |
| Released March 2026, 120B/12B active, hybrid Mamba-2 + latent MoE + attention, 1M context, NVIDIA Open Model License | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 , https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b |
| 8 attention layers, 40 Mamba-2, 40 latent-MoE; 69.9 GB Q4_K | `docs/models/nemotron-3-super.md` Identity |
| 113/121 capped, 148 min, 6 truncated, 133.6k reasoning tokens, 19.7 t/s | `state/evals/sec/nemotron-3-super-spark-cuda/20260915T122649Z-extract-full/run.json` |
| 112/121 full window, 153 min, fallback 0, 6 truncated, 131.3k reasoning tokens, 19.4 t/s | `state/evals/sec/nemotron-3-super-spark-cuda/20260916T103526Z-extract-full/run.json` |
| 99/104 and 97/104 on non-GS/STWD filings; cold TTFT 105–159 s on 100K+ filings; GS whole 397 s | computed from the two `results.jsonl` files (`.claude/skills/model-run-email/facts.py`) |
| gpt-oss-120b 21k reasoning tokens, completion median 154 vs Nemotron 729–741 | `state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full` |
| CI 0.87–0.97 (full) and 0.88–0.98 (capped) vs 120b 0.96–1.00 | `state/evals/report-sec.md`, Nemotron report Caveats |
| GS 274,176 tokens, STWD 203,910 tokens under Nemotron's tokenizer | `results.jsonl` `doc_tokens` |
| RTX 5090 rows | `docs/eval-report-2026-09-rtx5090-cuda-qwen.md` summary table |
| Qwen coding reasoning on vs off: 68 of 542 blank | `docs/eval-report-2026-09-rtx5090-financebench-swe-qwen.md` |
