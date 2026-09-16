# Model reference cards

One card per model this project has served or scored. Each card records the exact serving
configuration used, the eval settings, the measured results with their source run, and an honest
strengths / weaknesses / when-to-use section. Cards are living documents: append to the
**Changelog** at the bottom of a card whenever a new run lands, and keep the summary table here
in step.

## Summary (SEC 10-K extraction, 121 questions, `scoring_version 2026-09-13.1`)

| Card | Class | Where run | Accuracy | 95% CI | Decode | Whole run | Role today |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| [gpt-oss-120b](gpt-oss-120b.md) | MoE 117B / 5.1B active, MXFP4 | Spark CUDA | 98.3% | 0.96–1.00 | 30 t/s | 26 min | **On-prem default** for extraction |
| ↳ same, via LM Studio | | Halo Vulkan | 96.7% | 0.93–0.99 | 29 t/s | 77 min | Halo reference |
| ↳ same, via LM Studio | | Halo ROCm | 95.0% | 0.91–0.98 | 19 t/s | 72 min | Not recommended on Windows |
| [qwen3.8-27b](qwen3.8-27b.md) | Dense 27B, UD-Q4_K_XL | Spark CUDA | 95.9% | 0.92–0.99 | 9.8 t/s | 125 min | Careful reader, slow; candidate for reasoning tasks |
| ↳ same, native 262K + q8_0 KV | | RTX 5090 CUDA (Windows) | 98.3% | 0.96–1.00 | 55 t/s | 24 min | Ties 120b on a 32 GB consumer card; GS recovered at 262K |
| ↳ same, 131K (Spark config) | | RTX 5090 CUDA (Windows) | 94.2% | 0.90–0.98 | 60 t/s | 23 min | Spark result replicated, 5x faster; GS still chunked 3/8 |
| [gpt-oss-20b](gpt-oss-20b.md) | MoE 21B / 3.6B active, MXFP4 | Spark CUDA | 87.6% | 0.81–0.93 | 45 t/s | 19 min | Latency/memory option, human review required |
| [nemotron-3-super](nemotron-3-super.md) | Hybrid Mamba/MoE 120B / 12.7B active, Q4_K | Spark CUDA, capped 131K | 93.4% | 0.88–0.98 | 20 t/s | 148 min | Over-thinks at 4096 (6 truncations); thinking cap + unconstrained run pending |
| [gpt-5.6-terra](gpt-5.6-terra.md) | Hosted frontier | OpenAI API | 99.2% | 0.98–1.00 | — | 20 min, ~$25 | Ceiling reference |

Intervals are percentile bootstraps over per-item verdicts from a single run; overlapping
intervals mean the models are not distinguishable at this sample size. The machine-generated
source is [`state/evals/report-sec.md`](../../state/evals/report-sec.md).

## How to update a card

1. Land the run (commit `state/evals/...`) and regenerate `report-sec.md`.
2. Copy the new numbers into the card's **Results** table with the run directory as the source.
3. Revise **Strengths / Weaknesses / When to use** if the evidence changed them.
4. Append a dated line to the card's **Changelog**, and update the summary table above.
5. New model: copy [`TEMPLATE.md`](TEMPLATE.md), fill every field you can, mark the rest
   `unknown` rather than deleting them.

Rules: every number cites a run directory or a report; no number lives only in a card.
Strengths and weaknesses describe what we measured, not the model card's claims.
