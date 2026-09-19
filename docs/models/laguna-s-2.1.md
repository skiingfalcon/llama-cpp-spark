# laguna-s-2.1

**Status:** planned (registered 2026-09-19, both platforms, no run yet). **Poolside's agentic-coding
MoE, 118B total / 8B active — the same active-parameter class as gpt-oss-120b, our best Spark
result, and the first coding-specialist model in this project's roster. Question to answer: does an
architecture built for SWE tasks read 10-Ks as well as gpt-oss-120b, and does it change the SWE-suite
picture the way nothing tested so far could?**

## Identity

| | |
| --- | --- |
| Family / vendor | Poolside, Laguna S 2.1 (open weights, released 2026-07-21; part of the Laguna family alongside XS.2 33B-A3B and M.1 225B-A23B, both released 2026-04-28) |
| Architecture | Mixture-of-experts, 118B total / ~8B active parameters; GQA with interleaved full and sliding-window attention layers (exact layer/head counts, expert count and top-k routing not yet verified against `poolside/Laguna-S-2.1`'s model card and the technical report, arXiv:2605.27605 — confirm on first real run); reasoning via hidden thinking, **off by default** (opposite of every other model registered here so far), toggled via `enable_thinking` in the chat template |
| Checkpoint served | `unsloth/Laguna-S-2.1-GGUF`, quant `UD-Q4_K_XL` (3 shards, ~40 GB) |
| Native context | Vendor sources disagree: some docs state 262,144, Poolside's own announcement claims up to 1,048,576 via extension. Unresolved — see Open questions |
| License | OpenMDW-1.1 (verify on the model card) |
| Registry entry | `models.toml` and `models.halo.toml` `[models."laguna-s-2.1"]` and `[models."laguna-s-2.1-thinking"]`, port 8093 |

Not registered: Laguna XS.2 (33B-A3B, overlaps the class already covered by gpt-oss-20b /
glm-4.7-flash / gemma-4-26b-a4b) and Laguna M.1 (225B-A23B, ~110GB+ at 4-bit with little Spark
headroom and no Halo fit, and 23B active parameters would sit in the Nemotron / Qwen3.5-122B-A10B
speed band).

## Serving configuration (planned)

| Knob | Spark | Halo |
| --- | --- | --- |
| llama.cpp | pinned `82d6bb284d1f`. Laguna architecture support (`ggml-org/llama.cpp` PR #25165) merged upstream 2026-07-22, before our pin was cut (2026-09-11) — expected to need no pin bump, the same way `gemma4` and `deepseek4` did not | prebuilt zip at `LLAMA_CPP_RELEASE` |
| Backend | cuda | vulkan (or rocm) |
| ctx_size / n_parallel | 131072 / default | 131072 / default |
| Batch / ubatch | 2048 / 2048 | 2048 / 512 |
| Flash attention / KV type | on / f16 (read "KV self size" from the server log once served) | on / f16, `--no-mmap` |
| Extra flags | `--jinja`; `-thinking` entry adds `--chat-template-kwargs {"enable_thinking": true}` | same, plus `--no-mmap` |
| Sampling (serving only) | model defaults (not yet verified) | same |

Eval decoding stays temperature 0, seed 42, `max_tokens` 4096. Plan: `scripts/spark-new-models-smoke.sh`
runs a 22-item gate (AAPL, XOM) for both twins — including `eval swe check`, since this is the first
coding-specialist model on the roster — then, if the gate passes, `extract-full` at 131K with
thinking off (the primary row, matching vendor default), then the `-thinking` twin.

## Results

| Run | Task | Accuracy | 95% CI | Fits / fallback | Truncated | Decode | Cold prefill | Wall clock | Source |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |

## Strengths (expected, unmeasured)

- 8B active parameters: the bandwidth rule predicts roughly gpt-oss-120b's ballpark (~25-30 t/s) on
  GB10, since the active-parameter counts are close (8B vs 5.1B) — the closest architectural peer to
  our best Spark result so far.
- ~40 GB of weights: comfortably fits the Spark alongside a large KV cache, and fits the Halo's 96 GB
  VGM cap — the first Laguna candidate (and only the second model after Gemma) to run on both
  platforms rather than being Spark-only.
- Purpose-built for agentic coding/SWE: the only model in this roster designed for that workload
  rather than adapted to it, so it's the natural one to give the SWE suite a real test rather than
  just a gate check.
- Poolside publishes an official self-speculative drafter, `poolside/Laguna-S-2.1-DFlash`, and our
  pin already carries generic `--spec-type draft-dflash` support — a plausible speed-up once the
  parked Spark speed plan resumes, though not part of this registration.

## Weaknesses (expected, unmeasured)

- Thinking off by default breaks the naming convention every other model here uses (base = thinking
  on); read `laguna-s-2.1` as the primary row and `laguna-s-2.1-thinking` as the experimental twin,
  not the other way around.
- An HF discussion on `poolside/Laguna-S-2.1` ("Thinking Loops, and Chat Template possible fix!")
  reports the model can loop in reasoning; if the `-thinking` twin shows this, it would explain any
  truncations the way it has for gemma/nemotron's default-on reasoning.
- Native context is unverified: if the real ceiling is 262,144 rather than 1M, the "read GS/STWD
  whole" experiment other long-context models got doesn't apply here.
- Architecture and quant are new to this harness; exact expert/layer counts, GQA ratio, and
  whether `UD-Q4_K_XL`'s dynamic quant holds up on numeric extraction are all unmeasured.

## When to use / when not to

- Not yet. If `laguna-s-2.1` lands at or above gpt-oss-120b (98.3%) at a comparable decode speed, it
  becomes a second on-prem default candidate and the first coding-specialist entry worth taking to
  the SWE suite seriously. If it lands well below that, or the thinking twin loops, the case for
  Laguna narrows to "SWE-only" rather than a dual-purpose model.

## Open questions

- Exact architecture: layer count, GQA head ratio, sliding-window pattern, expert count and
  top-k routing — verify against the vendor's model card and arXiv:2605.27605.
- Real native context ceiling (262K vs 1M) and "KV self size" at 131K.
- Whether `--chat-template-kwargs {"enable_thinking": true}` is honoured by the served template, or
  whether `--reasoning on` is needed instead.
- Whether the reported thinking-loop issue shows up under our 4096-token completion budget.
- SWE-suite results specifically, since this is the first model registered here for that reason.
- DFlash drafter compatibility and acceptance rate on 10-K text and SWE tasks, once the parked Spark
  speed plan resumes.

## Changelog

- 2026-09-19 — registered on both platforms with a thinking-on twin (`-thinking`, not `-nothink`,
  since vendor default is thinking off); card created.
