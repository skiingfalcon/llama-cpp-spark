# SEC 10-K extraction — cross-stack comparison table

**Canonical, cumulative.** This is the one comparison table that goes in every model-run email.
It is hand-maintained here because not every drafted email gets committed to `docs/emails/`, so
scanning past emails for "prior rows" silently drops rows (this file exists because that
happened once: the 2026-09-19 DeepSeek draft missed the Bonsai rows added on 2026-09-18). Update
this file whenever a new row is added, in the same edit that adds the row to an email.

Task: 11 financial metrics from the latest 10-K of 12 companies, 121 questions, scored against
SEC XBRL ground truth at 0.5% tolerance. Every row is a single run, one seed. Read gaps inside
overlapping 95% intervals as noise, not a ranked difference.

| Stack | Accuracy, 121 Qs | When the filing fits, of 104 Qs | Decode | Cold prefill, new ~100K filing | Full run | Cost per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra | 99.2% | 103/104 | n/a | n/a | 20 min | ~$25 |
| DGX Spark, gpt-oss-120b (MoE) | 98.3% | 103/104 | 30 t/s | 40–60 s | 26 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 262K ctx | 98.3% | 102/104 | 55 t/s | 40–70 s | 24 min | $0 marginal |
| DGX Spark, DeepSeek-V4-Flash-nothink (MoE, 13B active, 2-bit) | 98.3% | 102/104 | 16 t/s | 190–306 s | 51 min | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 262K ctx | 96.7% | 100/104 | 93 t/s | 37–64 s | 20 min | $0 marginal |
| AMD Strix Halo, gpt-oss-120b, Vulkan | 96.7% | 103/104 | 29 t/s | 4–8 min | 77 min | $0 marginal |
| DGX Spark, Qwen3.8-27B (dense) | 95.9% | 104/104 | 10 t/s | 70–200 s | 125 min | $0 marginal |
| RTX 5090, Qwen3.8-27B (dense), 131K ctx | 94.2% | 102/104 | 60 t/s | 38–54 s | 23 min | $0 marginal |
| DGX Spark, Bonsai 2 27B (ternary), 131K ctx | 93.4% | 102/104 | 16 t/s | 120–200 s | 107 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, 131K budget | 93.4% | 99/104 | 20 t/s | 105–160 s | 148 min | $0 marginal |
| RTX 5090, Bonsai 2 27B (ternary), 131K ctx | 92.6% | 101/104 | 98 t/s | 37–53 s | 21 min | $0 marginal |
| DGX Spark, Nemotron-3-Super, full 1M window | 92.6% | 97/104 | 19 t/s | 105–160 s; GS whole 6.5 min | 153 min | $0 marginal |
| DGX Spark, gpt-oss-20b (MoE) | 87.6% | 94/104 | 45 t/s | 40–60 s | 19 min | $0 marginal |

"When the filing fits" counts the 104 questions on the ten filings that fit a 131K window, so the
column is comparable across tokenizers. Bonsai shares Qwen's tokenizer, so its filings are the
same size as Qwen's to the token.

## Row provenance

| Row | Source |
| --- | --- |
| OpenAI gpt-5.6-terra | `docs/eval-report-2026-09-spark-halo-terra.md` |
| DGX Spark / AMD Strix Halo, gpt-oss-120b and gpt-oss-20b | `docs/eval-report-2026-09-spark-cuda-rerun.md`, `docs/eval-report-2026-09-spark-halo-terra.md` |
| RTX 5090, Qwen3.8-27B (both ctx) | `docs/eval-report-2026-09-rtx5090-cuda-qwen.md` |
| DGX Spark, Qwen3.8-27B (dense) | `docs/eval-report-2026-09-spark-cuda-qwen.md` |
| DGX Spark, DeepSeek-V4-Flash-nothink | `docs/emails/2026-09-19-deepseek-v4-flash-nothink.md`, `state/evals/sec/deepseek-v4-flash-nothink-spark-cuda/20260919T121223Z-extract-full` |
| RTX 5090, Bonsai 2 27B (both ctx) | `docs/eval-report-2026-09-rtx5090-bonsai2.md` |
| DGX Spark, Bonsai 2 27B (ternary), 131K ctx | `state/evals/sec/bonsai-2-27b-spark-cuda/20260918T204818Z-extract-full` (no dedicated report yet; run on PrismML's fork, tag `prism-b10685`, not the pinned upstream build) |
| DGX Spark, Nemotron-3-Super (both windows) | `docs/eval-report-2026-09-spark-cuda-nemotron.md` |
