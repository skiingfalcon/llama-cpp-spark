# laguna-s-2.1

**Status:** measured (Spark, thinking-off row, 2026-09-20; `-thinking` twin not yet run).
**Poolside's agentic-coding MoE, 118B total / ~8B active, the closest active-parameter peer to
gpt-oss-120b and the first coding-specialist model in this roster. First run: 115/121 (95.0%) in
29 minutes, the fastest wall clock of any local model after gpt-oss-120b, at 17 t/s decode. Four
questions behind gpt-oss-120b; the misses are Goldman's BM25 chunks plus two wrong-line picks.**

## Identity

| | |
| --- | --- |
| Family / vendor | Poolside, Laguna S 2.1 (open weights, announced 2026-07-21; the family also has XS.2 at 33B-A3B and M.1 at 225B-A23B, released 2026-04-28) |
| Architecture | Mixture-of-experts, 118B total / ~8B active per token; 48 layers in a 1:3 global-to-sliding-window ratio (12 global attention layers, 36 sliding-window layers, window 512); grouped-query attention with 8 KV heads; 256 routed experts (top-10) plus 1 shared expert; native 1,048,576 context; reasoning via hidden thinking, **off by default** (opposite of every other model registered here), toggled per request through the chat template's `enable_thinking` kwarg |
| Checkpoint served | `unsloth/Laguna-S-2.1-GGUF`, quant `UD-Q4_K_XL` (3 shards, 73.4 GB) |
| Native context | 1,048,576 |
| License | OpenMDW-1.1 |
| Registry entry | `models.toml` and `models.halo.toml` `[models."laguna-s-2.1"]` and `[models."laguna-s-2.1-thinking"]`, port 8093 |

Not registered: Laguna XS.2 (33B-A3B, overlaps the class already covered by gpt-oss-20b /
glm-4.7-flash / gemma-4-26b-a4b) and Laguna M.1 (225B-A23B; at 4 bits the weights alone would
leave little Spark headroom and no Halo fit, and 23B active parameters would sit in the Nemotron /
Qwen3.5-122B-A10B speed band).

## Serving configuration used

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f` (`b869`). Laguna architecture support (`ggml-org/llama.cpp` PR #25165) merged upstream 2026-07-22, before the pin was cut (2026-09-11); no pin bump was needed | prebuilt zip at `LLAMA_CPP_RELEASE` (not yet served) |
| Backend | cuda | vulkan (or rocm) |
| ctx_size / n_parallel | 131072 / default (4 slots) | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-thinking` entry adds `--chat-template-kwargs {"enable_thinking": true}` | same, plus `--no-mmap` |
| Sampling (serving only) | model defaults | same |
| Chat template | served template sha `444819b8ad46` (recorded in `run.json`) | |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Spark CUDA, 131K, thinking off (vendor default) | extract-full | 95.0% (115/121) | 0.91–0.98 | 93 full (92/93) / 28 fallback (NEE + STWD Item 8: 19/20; GS BM25 chunks: 4/8) | 0 | 17.2 t/s | 816 t/s; 97–157 s TTFT on 80K+ filings | 29 min | [`20260920T003059Z-extract-full`](../../state/evals/sec/laguna-s-2.1-spark-cuda/20260920T003059Z-extract-full/) |

Report: [`eval-report-2026-09-spark-cuda-gemma-laguna.md`](../eval-report-2026-09-spark-cuda-gemma-laguna.md).

Misses (6): GS NetIncomeLoss (the model wrote a 608-token prose search instead of a number, with
thinking off: "I need to find the net income..."), GS Assets / Liabilities / StockholdersEquity
("unknown", BM25 chunks), XOM Revenues (323.9B, the sales line, against 332.2B total revenues and
other income), NEE StockholdersEquity (66.5B, total equity including noncontrolling interests,
against 54.6B).

## Strengths (measured)

- 29 minutes for the suite, three minutes behind gpt-oss-120b and faster than every other local
  model: DeepSeek-nothink 51, Gemma-nothink 61, Qwen 125. Completions are tiny (median 13 tokens,
  no reasoning) and cold prefill is 816 t/s, second only to gpt-oss-120b's 1,354.
- 17.2 t/s decode, where the bandwidth rule puts it: ~8B active at 5 bits per weight (73.4 GB /
  118B) is about 5 GB per token against gpt-oss-120b's 2.7 GB, so 30 t/s × 2.7 / 5 ≈ 16.
- Zero truncations; 92 of 93 on full-document questions; 19 of 20 on the Item 8 section path.
- Prompt-cache reuse 86.8% (9,005,904 of 10,378,894), warm re-prefill gap a median 86 tokens; the
  sliding-window layers did not force a checkpoint re-prefill on this pin.
- 73.4 GB of weights: fits the Halo's 96 GB cap as gpt-oss-120b's 63 GB does, so it can run on both
  platforms, unlike deepseek-v4-flash.

## Weaknesses (measured)

- Four questions behind gpt-oss-120b (115 vs 119); intervals 0.91–0.98 vs 0.96–1.00 touch, so this
  is at the edge of noise on one seed, but every other local model at this speed class also lost.
- The tokenizer is fatter than gpt-oss's, as Gemma's is: GS 274,903 tokens (+14%), NEE 138,457
  (+12%). NEE fell out of full-document mode and GS's Item 8 no longer fit, so Goldman went to BM25
  chunks and lost 4 of 8. Same shape as Gemma; gpt-oss kept GS on the section path (7/8).
- Two misses are wrong-line picks, not extraction failures: XOM's sales line for total revenues,
  and NEE's total equity including noncontrolling interests. Both are the kind of accounting
  judgment a coding-tuned model would not have been trained toward.
- With thinking off it once reasoned aloud in the answer field for 608 tokens and never produced
  the number (GS net income). The reported thinking-loop issue may show up more with thinking on.

## When to use / when not to

- The fast second option when gpt-oss-120b is busy or a coding-tuned model is wanted on the same
  box: same wall clock class, four questions behind on one seed. Not yet a replacement for
  gpt-oss-120b on extraction.
- The model to take to the SWE tier 1 suite first: it is the only one here built for that
  workload. The smoke gate's `eval swe check` output was not committed, so treat SWE readiness
  as unverified until that run lands.

## Open questions

- `laguna-s-2.1-thinking`: does thinking on fix the wrong-line picks or trigger the reported loops
  under a 4,096-token budget? Not yet run.
- SWE tier 1, thinking off then on: the reason this model was registered.
- Native window: does 262K (or more) on the Spark put GS's Item 8 back in budget and recover the
  four chunk misses, as it did for Qwen on the 5090?
- `poolside/Laguna-S-2.1-DFlash` drafter with the pin's `--spec-type draft-dflash`: acceptance rate
  on 10-K text and SWE tasks, once the parked Spark speed plan resumes.
- Halo run at 73.4 GB inside the 96 GB cap.

## Changelog

- 2026-09-20 — first Spark run recorded (thinking off, 115/121, 17 t/s, 29 min); checkpoint size
  corrected to 73.4 GB (registered as ~40 GB); architecture filled from the vendor card.
- 2026-09-19 — registered on both platforms with a thinking-on twin (`-thinking`, not `-nothink`,
  since vendor default is thinking off); card created.
