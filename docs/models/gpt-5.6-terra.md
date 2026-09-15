# gpt-5.6-terra (hosted reference)

**Status:** measured (OpenAI API). **Ceiling reference for the suite; not a local model.**

## Identity

| | |
| --- | --- |
| Family / vendor | OpenAI, hosted frontier model |
| Architecture | Undisclosed; reasoning model |
| Served via | OpenAI Chat Completions API, `--provider openai` |
| Context | 1,050,000 (as configured with `--context-window`) |
| License / terms | Provider terms; data leaves the network |
| Registry entry | none (API path); runs are named `openai:gpt-5.6-terra` |

## Configuration used

| Knob | Value |
| --- | --- |
| Decoding | provider defaults (frontier reasoning models reject fixed temperature/seed); `max_completion_tokens` 4096 |
| Parallelism | `--parallel 1` after the first run lost 10 items to HTTP 429 |
| Retry | `Retry-After` honoured, 8 retries, 60 s cap |
| Tokenizer for budgeting | `o200k_base` via tiktoken (12.39M prompt tokens per run vs 10.5M for gpt-oss) |

## Results

| Run | Task | Accuracy | 95% CI | Fallback | TTFT p50 | Total p50 / p95 | Wall clock | Cost (est.) | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| First run | extract-full, 10-K | 96.4% raw (106/110), 95.8% paired-96 | — | none; 10 items lost to 429 | 1.2 s | 1.2 s / 2.2 s | 24 min | ~$22 | `state/evals/sec/openai_gpt-5.6-terra/20260912T033135Z-extract-full` |
| **Current** | extract-full, 10-K | **99.2% (120/121)** | 0.98–1.00 | none (reads GS/STWD whole) | 1.44 s | 1.6 s / 2.9 s | 20 min (3.5 min request time) | ~$25 | `…/20260912T142344Z-extract-full` |

Cost is estimated from token counts at list price ($2/M input, $12/M output), not billed.
Prompt-cache reuse observed: ~0% on this path. Reasoning tokens reported: ~1.3k total (very little).

## Strengths (measured)

- Highest score and the only stack that read the two oversized filings whole; 17/17 on those items.
- Fastest per-answer latency (1.6 s median) with almost no visible reasoning.
- No serving to operate.

## Weaknesses (measured)

- Its one miss (HD share count, `unknown`) is on the cover page, which both local models read correctly.
- Rate limits stretched 3.5 minutes of compute into 20 minutes of wall clock even serially.
- No fixed temperature or seed; reruns are not bit-identical.
- ~$25 per suite run with no cache discount observed; cost scales with document size times question count.
- Data egress and provider retention terms apply.

## When to use / when not to

- **Use** as the accuracy ceiling when evaluating a new local model, and for documents that exceed local context.
- **Do not** use it as the default for routine extraction; the local 120B model matches it wherever the document fits, at zero marginal cost.

## Open questions

- Behaviour with prompt caching enabled explicitly (the 90% reuse seen locally would cut cost substantially if the API honoured it here).

## Changelog

- 2026-09-12 — first run (rate-limit degraded) and clean re-run.
- 2026-09-15 — card created.
