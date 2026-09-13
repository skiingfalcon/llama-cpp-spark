# gpt-oss-120b on a DGX Spark vs an AMD Strix Halo (Vulkan and ROCm) vs GPT-5.6 Terra, on a real 10-K extraction task: 98.3% / 96.7% / 99.2%, harness and raw results included

**Venue:** r/LocalLLaMA (Reddit)
**Audience:** people running open-weight models on their own hardware; many Spark and Strix Halo owners
**Source of numbers:** `docs/eval-report-2026-09-spark-halo-terra.md`, `docs/eval-report-2026-09-spark-cuda-rerun.md`, `state/evals/sec/*`
**Status:** draft

---

I wanted to know whether an open-weight model on a desk-sized box is good enough for a real
document workload, not a synthetic benchmark. The workload: pull 11 financial metrics (revenue,
net income, EPS, total assets, long-term debt, shares outstanding, and so on) out of the latest
10-K of 12 US companies, 121 questions, scored against the SEC's own XBRL ground truth at 0.5%
tolerance. Filings are 45K to 124K tokens each; two of them (Goldman Sachs, Starwood) do not
fit in 131K and get answered from their financial-statements section.

Same model file (`gpt-oss-120b-MXFP4.gguf`), same prompts, same sampling (temp 0, seed 42,
4096-token completion budget), three local stacks, plus a hosted frontier model for reference.

| Stack | Accuracy (121 Qs) | When the filing fits (104 Qs) | Decode | Cold prefill of a ~100K-token filing | Whole suite |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI gpt-5.6-terra (API, 1.05M ctx) | 99.2% | 103/104 | n/a | n/a | 20 min, ~$25 |
| NVIDIA DGX Spark, llama.cpp CUDA | 98.3% | 103/104 | 30 t/s | 40 to 60 s | 26 min |
| AMD Strix Halo, llama.cpp Vulkan | 96.7% | 103/104 | 29 t/s | 4 to 8 min | 77 min |
| AMD Strix Halo, llama.cpp ROCm | 95.0% | 101/104 | 19 t/s | 4 to 8 min | 72 min |

**What I took from it**

- On every filing that fits in context, the Spark, the Halo on Vulkan, and the frontier model
  score the same: 103 of 104. The frontier model's edge is reading the two oversized filings whole.
- Decode speed is basically identical on the two boxes (bandwidth-bound; both are around 256 to
  273 GB/s). The 3x wall-clock gap is entirely prefill, which is compute-bound, and the GB10 has a
  lot more of it. Once a filing is cached, follow-up questions cost about the same on both.
- ROCm on Windows was not the win the Linux benchmarks suggest: about 5% faster prefill than
  Vulkan, a third slower decode, and two more misses. Vulkan is the backend to use on this box
  today.
- The harness fixes mattered more than the hardware. The first run had gpt-oss losing 6 points to
  a 512-token completion budget that truncated its hidden reasoning, and four "misses" that were
  wrong ground truth (XBRL alias choices). The pre-fix and post-fix reports are both in the repo.

**Caveats, so you do not have to find them**

- The Halo box is locked down (no Python, no compiler), so its runs went through LM Studio
  2.37's bundled llama.cpp via a Node script, not the pinned build the Spark uses. Treat the
  accuracy columns as comparable and the speed columns as indicative.
- Everything ran serially, one request at a time. No concurrency numbers.
- Token counts on the Halo path are character estimates (LM Studio has no `/tokenize`).
- The frontier cost is an estimate from token counts at list price, not a bill.

**Repo:** https://github.com/skiingfalcon/llama-cpp-spark — harness, per-question results
(`results.jsonl` for every run), the three write-ups, and the Node script for locked-down boxes.
Start at `docs/index.md`.

**Questions for people who have gone further than I have**

1. Has anyone measured a rocWMMA-enabled ROCm build (or Linux + RADV) on gfx1151 with a
   120B MoE at 100K+ prompt depth? I would like to know whether the 3x prefill gap to the Spark
   is the hardware or the Windows stack.
2. Any experience with `--kv-unified` / `--swa-full` against host-RAM growth under heavy prompt
   caching on Strix Halo? My runs were fine, but they were short.

## Feedback

<!-- fill in after posting -->
