# Strix Halo (gfx1151, Windows): Vulkan vs ROCm on gpt-oss-120b with 45K–124K-token prompts, compared to a DGX Spark CUDA build

**Venue:** llama.cpp GitHub Discussions (post as a new discussion; cross-reference from the Vulkan performance thread #10879 and the ROCm/HIP thread #15021)
**Audience:** llama.cpp maintainers and backend contributors; expect questions about flags and builds
**Source of numbers:** `docs/eval-report-2026-09-spark-halo-terra.md`, `state/evals/sec/gpt-oss-120b-*`
**Status:** draft

---

Real-workload data point rather than `llama-bench`: long-prompt extraction over SEC 10-K filings,
121 requests per run, prompts 45K–124K tokens, gpt-oss-120b MXFP4, 131,072 context, temp 0,
seed 42, 4096 completion budget. Accuracy is scored against XBRL ground truth so the backends
can be compared on correctness as well as speed.

**Hardware / builds**

| | DGX Spark | Strix Halo (Ryzen AI Max+ 395, Radeon 8060S) |
| --- | --- | --- |
| OS | DGX OS (Ubuntu, aarch64) | Windows 11, AMD Adrenalin |
| Build | llama.cpp `82d6bb284d1f`, CUDA, `CMAKE_CUDA_ARCHITECTURES=121a-real` | LM Studio 2.37.0 runtime packs: `llama.cpp-win-x86_64-vulkan-avx2` and `…-amd-rocm-avx2` (stock builds; no rocWMMA FA as far as I can tell) |
| Serve flags | `-fa on -b 2048 -ub 2048 -c 131072 -ngl 999 --jinja` | GPU offload 36/36, eval batch 512, unified KV on, KV offloaded to GPU, keep in memory |
| Memory | 128 GB unified (~121 GiB usable) | 128 GB unified, VGM set to 96 GB |

**Results (same 121 requests, same model file)**

| Backend | Correct | Decode t/s (p50) | TTFT p50 (cached prefix) | Per-request p95 (= cold ~100K prefill) | Whole run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spark CUDA | 119/121 | 30 | 0.57 s | 63 s | 26 min |
| Halo Vulkan | 117/121 | 29 | 1.38 s | 300 s | 77 min |
| Halo ROCm | 115/121 | 19 | 1.19 s | 238 s | 72 min |

- Decode is within noise between the two boxes on Vulkan, consistent with both being bandwidth
  bound at roughly the same GB/s.
- Cold prefill of a ~100K-token filing: 40–60 s on the Spark, 4–8 min on the Halo on either
  backend. That is the whole wall-clock difference.
- ROCm vs Vulkan on this Windows stack: prefill ~5% faster, decode ~35% slower, and two
  additional wrong answers (both on the GS filing answered from an Item 8 excerpt). I assume the
  KV-cache-in-shared-memory behaviour from #18011 is part of the ROCm decode story; I have not
  confirmed it.
- Accuracy on the 104 requests whose filing fits in context is identical for Spark CUDA and Halo
  Vulkan (103/104); Halo ROCm is 101/104.

**What I would like to learn from this thread**

1. Does the official `win-rocm` release zip build with `GGML_HIP_ROCWMMA_FATTN`? If not, is
   anyone shipping a Windows gfx1151 build that does, and what does long-context prefill look like
   on it?
2. Is there a known reason for the ~35% decode penalty on the Windows ROCm path vs Vulkan on
   gfx1151, or is that expected given #18011?
3. Any recommended `-ub` for Vulkan at 65K–130K context on this chip? I used 512 after reading
   #20515; 2048 is what the Spark uses.

**Caveats**

- The Halo box is locked down (no Python), so its runs went through LM Studio's server and a Node
  harness that mirrors the Python one. Token counts on that path are character estimates; TTFT and
  t/s come from LM Studio's `stats`. Treat the speed columns as indicative, the accuracy columns
  as exact.
- Serial requests only; nothing under concurrency.

**Raw data:** per-request `results.jsonl` and `run.json` for every run are committed under
`state/evals/sec/` in https://github.com/skiingfalcon/llama-cpp-spark; the write-up is
`docs/eval-report-2026-09-spark-halo-terra.md`. Happy to run additional configurations on the
Spark if someone wants a specific comparison.

## Feedback

<!-- fill in after posting -->
