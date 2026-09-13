# local-llm — llama.cpp on DGX Spark and AMD Strix Halo

**llama.cpp** serving plus a workload-driven eval harness (SEC 10-K extraction, SWE tiers) for
local inference boxes, wrapped in a small **uv**-managed Python CLI. Two platforms are supported
and compared with the identical harness:

- **NVIDIA DGX Spark** (GB10 / sm_121, Linux): native CUDA source build. The original target.
- **AMD Strix Halo** (Ryzen AI Max, Radeon 8060S / gfx1151, **Windows**): prebuilt Vulkan and
  HIP zips. See [Windows / AMD Strix Halo](#windows--amd-strix-halo).

Models are data (`models.toml` / `models.halo.toml`), not code — add a GGUF by editing TOML or
passing `--hf` / `--model-path`. Everything platform-specific lives under
`src/spark_llm/platforms/{spark,halo}/`; the CLI, registry, argv merge and evals are shared.

> **Renamed from `spark-llm`.** The command is now `local-llm` and settings use the
> `LOCAL_LLM_*` prefix. `spark-llm` and `SPARK_LLM_*` keep working as deprecated aliases, so
> existing scripts and `.env` files need no change. The Python package is still `spark_llm`.

## Architecture

![Inference stacks: NVIDIA DGX Spark (CUDA) vs AMD Strix Halo (Vulkan / ROCm)](docs/inference-stack-spark-vs-halo.png)

Three ways the same `gpt-oss-120b` MXFP4 weights get served in this project:

- **NVIDIA DGX Spark (CUDA)** — `local-llm` reads [`models.toml`](models.toml), merges argv in
  `server.py`, and spawns a pinned `llama-server` built by [`scripts/build.sh`](scripts/build.sh)
  (`platforms/spark`, ggml-cuda).
- **AMD Strix Halo (Vulkan / ROCm)** — same CLI on Windows via `platforms/halo` and the
  release-zip installer (`--backend vulkan|hip`); intended path when the box allows Python.
- **AMD Strix Halo via LM Studio** — locked-down hosts that cannot install `uv` / a compiler
  still run the extract-full suite through [`scripts/lmstudio.mjs`](scripts/lmstudio.mjs)
  against LM Studio's bundled llama.cpp. See the
  [Halo eval report](docs/eval-report-2026-09-halo.md) for the September 2026 numbers.

## Requirements

**DGX Spark** (verified on this machine):

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB10 (compute capability 12.1 / sm_121) |
| Memory | 121 GB unified |
| CUDA | 13.0 toolkit + driver 580.x |
| CPU / OS | aarch64, Ubuntu |
| Tooling | uv, Python 3.12, cmake ≥ 3.30 (via `uv tool`), ninja |

Host cmake 3.28 is too old for the `121a` architecture suffix — install a current cmake with `uv tool install 'cmake>=3.30'` and `uv tool install ninja` (already done if you followed the project setup).

**AMD Strix Halo** (Windows) needs no compiler at all; see the dedicated section below.

| Item | Value |
| --- | --- |
| APU | AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151, RDNA 3.5) |
| Memory | 128 GB unified LPDDR5X; assign ≥ 96 GB to the iGPU via Variable Graphics Memory |
| OS | Windows 11 + AMD Adrenalin driver (≥ 26.6.4 for the ROCm/HIP backend) |
| Tooling | uv, Python 3.12, git — no CMake, no Visual Studio, no ROCm SDK install |

## Quickstart (DGX Spark)

```bash
cd ~/projects/llama-cpp-spark

# 1. Build llama.cpp (CUDA, sm_121a-real)
make build
# or: uv run local-llm build

# 2. Download gpt-oss-20b (~12 GB MXFP4 GGUF) into /opt/models
uv run local-llm download gpt-oss-20b

# 3. Serve (background; health-checked)
uv run local-llm serve gpt-oss-20b

# 4. Chat
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Say hello in one sentence."}]}'

uv run local-llm chat gpt-oss-20b -m "Say hello in one sentence."

# 5. Stop
uv run local-llm stop
```

Sanity checks: `uv run local-llm doctor`, `uv run local-llm models`.

## Windows / AMD Strix Halo

### Why this exists

The SEC extraction benchmark showed `gpt-oss-120b` on the DGX Spark at parity with a frontier
model. The next question was whether a cheaper AMD **Strix Halo** mini-PC (Ryzen AI Max+ 395,
Radeon 8060S / gfx1151, 128 GB unified memory) gets the same result. Answering that honestly
means running the *identical* harness, model file, sampling and prompts on both boxes, so the
Halo box is a first-class platform of this repo rather than a separate script. The box runs
Windows, so this section is written for PowerShell.

### Prerequisites

| Item | Why |
| --- | --- |
| AMD Adrenalin driver, current | Vulkan runtime ships with it; the HIP zips need **26.6.4 or newer** |
| **Variable Graphics Memory** set high (≥ 96 GB) | Adrenalin → Performance → Tuning. `gpt-oss-120b` is ~63 GB of weights plus KV cache. Windows caps the iGPU at 96 GB; the setting cannot be read by software, so `doctor` only reminds you |
| Python 3.12, `uv`, `git` | `uv sync` installs everything else; no CMake, Visual Studio or ROCm SDK |
| ~70 GB free disk | weights; another ~300 MB for the two llama.cpp zips |

### Install and run (PowerShell)

```powershell
git clone https://github.com/skiingfalcon/llama-cpp-spark; cd llama-cpp-spark; uv sync
uv run local-llm doctor                       # platform halo, backends, GPU, memory, VGM reminder
uv run local-llm build                        # downloads the pinned win-vulkan and win-rocm zips
uv run local-llm download gpt-oss-120b        # -> %LOCALAPPDATA%\local-llm\models
$env:LOCAL_LLM_EDGAR_USER_AGENT = "local-llm you@example.com"
uv run local-llm eval sec fetch               # SEC corpus (once)

# 1. Hardware sweep on both backends first (8K–100K tokens, cold vs warm cache).
uv run local-llm serve gpt-oss-120b --backend vulkan
uv run local-llm eval sec perf gpt-oss-120b; uv run local-llm stop
uv run local-llm serve gpt-oss-120b --backend hip
uv run local-llm eval sec perf gpt-oss-120b; uv run local-llm stop
uv run local-llm eval report --suite sec      # "Hardware" block: spark/cuda vs halo/vulkan vs halo/hip

# 2. Accuracy run on the backend that won at 100K prefill. Overnight for 120b.
uv run local-llm serve gpt-oss-120b --backend <vulkan|hip>
uv run local-llm eval sec run gpt-oss-120b --task extract-full --forms 10-K
uv run local-llm stop; uv run local-llm eval report --suite sec
```

`build` resolves the llama.cpp release tag from `LLAMA_CPP_RELEASE` (the nearest tag to the
Spark's pinned commit), downloads `llama-<tag>-bin-win-vulkan-x64.zip` and
`llama-<tag>-bin-win-rocm-<ver>-x64.zip`, unpacks them under `vendor/llama.cpp/<tag>-<backend>/`,
runs `--version` and `--list-devices`, and records everything in `vendor/llama.cpp/halo-build.json`.
If the official ROCm zip lists no GPU (gfx1151 not covered, or the driver is older than the
bundled HIP runtime), use AMD's Lemonade build, which has a dedicated gfx1151 target:
`uv run local-llm build --backend hip --source lemonade --force`.

Eval and report commands are the same as on the Spark. Run artifacts land in the same
`state/evals/` tree; `run.json` carries `platform`, `backend` and `llama_cpp_release`, and the
report keeps runs from different platforms side by side instead of letting a Halo run replace
the Spark run of the same model.

### Which backend: Vulkan or HIP

Both, measured. The Strix Halo community numbers (Linux, Sept 2026) say the answer depends on
context length, and this workload sits at the far end:

| | Vulkan | HIP (ROCm) |
| --- | --- | --- |
| Setup | zero dependencies beyond the driver | driver ≥ 26.6.4; gfx1151 coverage varies by build |
| Short context (≤ 8K) | fastest decode (~85 vs 64 t/s on a 30B MoE), equal prefill | slightly slower |
| 130K context depth | prefill collapses (~17 t/s) | ~3x faster prefill with rocWMMA (~51 t/s), equal decode |
| Windows-specific reports | shared-memory leak on long-running servers (driver 26.3.1) | KV cache lands in shared memory, hurting very long context |

Our filings are 45K–124K tokens, so the `perf` sweep above is not optional: it tells you which
backend to use for SEC work on *this* driver, and the report records both. Expect cold prefill
of a 100K-token filing to take minutes on either backend, versus 40–60 s on the Spark; the eval
request timeout is 3600 s on this platform for that reason.

`models.halo.toml` carries the Strix Halo serving defaults: `--no-mmap` (mmap'd weights are very
slow on this APU), `ubatch_size = 512` (2048 is implicated in Vulkan `DeviceLost` crashes at
65–80K context), `n_gpu_layers = 999`, `host = 127.0.0.1`. If `unified_mem` in `perf` output
climbs monotonically across prompts, add `--kv-unified` (llama.cpp #22372).

### What differs from the Spark

| | DGX Spark | Strix Halo (Windows) |
| --- | --- | --- |
| llama.cpp | pinned commit, CMake + CUDA `sm_121a` source build | pinned release tag, prebuilt `win-vulkan` / `win-rocm` zips |
| Backend | `cuda` | `vulkan` or `hip`, chosen per `serve --backend` |
| Registry | `models.toml` | `models.halo.toml` (same names and ports) |
| Weights | `/opt/models` | `%LOCALAPPDATA%\local-llm\models` (`LOCAL_LLM_MODELS_DIR`) |
| Process control | `setsid` + `SIGTERM` | detached process group + `taskkill /T` |
| Telemetry | `nvidia-smi` (VRAM, per-process) | CIM: GPU name/driver, **unified memory in use** (no per-process split) |
| Bind host | `0.0.0.0` | `127.0.0.1` (no firewall prompt) |
| Eval / report commands | identical | identical |

Everything in that table lives in `src/spark_llm/platforms/spark/` and
`src/spark_llm/platforms/halo/`. Shared modules call `platforms.current()` and never branch on
the OS themselves. `LOCAL_LLM_PLATFORM=spark|halo` forces detection (e.g. a Linux box driving
tests for the Halo code).

### Why we run llama-server directly and not LM Studio

LM Studio is the easy way to get a model running on this hardware, and it is worth ten minutes
as a smoke test that the weights load on the GPU and that Variable Graphics Memory is set.
It is not the benchmark engine, for three concrete reasons:

- **The harness depends on llama-server endpoints LM Studio does not expose in that form.**
  `/tokenize` (context budgeting per model tokenizer), `/props` (served context, build, chat
  template hash), per-request `timings` (prompt and decode tokens/s), `cache_prompt`, and
  `chat_template_kwargs` (reasoning effort). Going through LM Studio means writing an adapter
  and losing the throughput and cache columns that make the Spark comparison meaningful.
- **Reproducibility.** LM Studio bundles and auto-updates its own llama.cpp runtime. A hardware
  comparison needs the build pinned and recorded in every `run.json` (`LLAMA_CPP_VERSION`,
  `LLAMA_CPP_RELEASE`). Two boxes on two unknown engine versions is not a hardware comparison.
- **No performance upside.** LM Studio's AMD runtime *is* llama.cpp (Vulkan or ROCm). Same
  kernels, one more layer, fewer knobs.

### Does LM Studio give an easier path to production?

No. It shortens the demo, not the deployment.

- LM Studio is a desktop application. Its server runs inside a user session, has no
  authentication, no metrics endpoint, no container image, no multi-tenant or fleet story, and
  is closed source. Headless mode helps a developer laptop, not a service.
- Anything built against it gets swapped out at the first production step, so a benchmark run
  through it would have measured an engine nobody ships.
- `llama-server` is already the production-shaped component: one binary, `--api-key`,
  `--metrics` (Prometheus), `--parallel` slots, `--slot-save-path`, any reverse proxy in front,
  runnable as a Windows service (NSSM or Task Scheduler) or a Linux container. Bench, eval and
  serve in this repo all drive the same binary with the same flags, which is the point.
- Honest caveat: **Windows is a fine bench target and a poor production host for this stack.**
  Every serious Strix Halo deployment runs Linux (Ubuntu 24.04 HWE kernel ≥ 6.17, Mesa RADV,
  `amd-ttm` to give the GPU ≥ 110 GB), where the drivers, containers and systemd live. If the
  Windows numbers disappoint or hosting becomes real, put Linux on the same box; the
  `platforms/` split makes a `halo-linux` variant a small addition because the POSIX process
  handling already exists in `platforms/spark/`.

| | LM Studio | llama-server (this repo) | Ollama |
| --- | --- | --- | --- |
| Pinned, recorded build | no (auto-updating runtime) | yes (`LLAMA_CPP_VERSION` / `_RELEASE`) | partial (own release cadence) |
| API auth / metrics | no / no | `--api-key` / `--metrics` | no / no |
| Service / container | user session, headless mode | Windows service or Linux container | Windows service, Linux container |
| Harness compatibility | adapter needed, columns lost | native | adapter needed (no `/tokenize`, `/props`) |
| Flag control (ubatch, KV type, FA) | GUI subset | full | limited |
| Licence | proprietary, free for work use | MIT | MIT |

Ollama is the closest alternative and also wraps llama.cpp, but it hides the serving flags this
comparison depends on and keeps its own model store. vLLM is not an option here: no Windows
build, and its ROCm support for gfx1151 is immature.

### Locked-down machines: LM Studio extract-full workaround

Some Halo (or other) boxes are locked down: no Python, no compiler, no admin rights to install
`uv` / Visual Studio Build Tools, and only a browser-installable app like **LM Studio** is
allowed. In that case you cannot run `local-llm serve` or the Python SEC harness, but you can
still run a comparable **extract-full** accuracy check against LM Studio's OpenAI-compatible
endpoint with the single Node script checked in here:

[`scripts/lmstudio.mjs`](scripts/lmstudio.mjs) — Node 18+ only, no npm dependencies. It mirrors
the harness corpus (same 12 tickers, 1×10-K + 3×10-Q), XBRL ground truth, prompts, free-text
scoring, and full → section → BM25 top-k context degradation.

```powershell
# On the locked-down box (PowerShell). Node from nodejs.org is enough.
$env:EDGAR_UA = "local-llm you@example.com"   # SEC fair-access User-Agent
# Load a model in LM Studio, start the local server (default http://127.0.0.1:1234)
node scripts/lmstudio.mjs fetch
node scripts/lmstudio.mjs run --forms 10-K    # overnight for 120b; --limit N to smoke-test
node scripts/lmstudio.mjs report
```

Optional: `$env:LMS_URL`, `$env:LMS_MODEL`, `$env:LMS_TOKEN`, `--ticker`, `--tag`,
`--reasoning-effort low`, `--insecure` (corporate TLS inspection).

**What matches the Python harness:** companies, forms, tags/aliases/`accept_aliases`, question
wording with period-end dates, prompts, `parse_number` scoring (0.5% tolerance + off-by-scale),
and the context-degradation policy. Results land under
`state/evals/sec/<model>-halo-<vulkan|rocm>/<timestamp>-extract-full/` with
`served_via: lmstudio`, next to the Spark and Terra runs in the same `sec/` tree.

**What does not:** token counts are `chars/4.6` estimates (LM Studio has no `/tokenize`), so
full vs section vs chunked boundaries can differ slightly; the backend is LM Studio's bundled
runtime, not the pinned `LLAMA_CPP_RELEASE`; cached-prompt and exact prefill/decode columns are
indicative only. Use this for an accuracy signal on a locked box, not as a drop-in replacement
for the Spark hardware comparison. Prefer `local-llm eval sec …` whenever Python is available.

**Results (September 2026):** on the same 121-question 10-K suite,
**OpenAI hosted (`gpt-5.6-terra`)** scored **120/121 (99.2%)**, **NVIDIA DGX Spark (CUDA)**
**119/121 (98.3%)**, **AMD Strix Halo (Vulkan)** **117/121 (96.7%)**, and **AMD Strix Halo
(ROCm)** **115/121 (95.0%)**. Raw artifacts live under `state/evals/sec/` next to the Spark
and Terra runs (`gpt-oss-120b`, `gpt-oss-120b-halo-vulkan`, `gpt-oss-120b-halo-rocm`,
`openai_gpt-5.6-terra`). Full write-up:
[docs/eval-report-2026-09-halo.md](docs/eval-report-2026-09-halo.md) (see also
[eval-report-2026-09-rerun.md](docs/eval-report-2026-09-rerun.md) and
[eval-report-2026-09.md](docs/eval-report-2026-09.md)).

### Known gaps on Halo

- No per-process GPU memory accounting; `perf` records **unified memory in use** instead, which
  is the right number on a UMA box but also counts everything else on the machine.
- The "foreign GPU process" guard before `bench` is inert (nothing to list processes with).
- The Windows Vulkan shared-memory leak and the Windows ROCm KV-in-shared-memory placement are
  upstream/driver behaviours; the harness measures them (`unified_mem`, long-context TTFT), it
  does not fix them.
- SWE tiers 2–3 need Docker Desktop and are out of scope for this box; tier 1 works.

## Adding a model

Edit [`models.toml`](models.toml). No Python changes required.

**Chat model (file + repo):**

```toml
[models.my-chat]
repo = "org/Some-Model-GGUF"
file = "some-model-Q4_K_M.gguf"
kind = "chat"
ctx_size = 32768
port = 8084
extra_args = ["--jinja"]
```

**Chat model (quant via llama.cpp `-hf`):**

```toml
[models.qwen3-8b]
repo = "unsloth/Qwen3-8B-GGUF"
quant = "Q4_K_M"
kind = "chat"
ctx_size = 32768
port = 8081
extra_args = ["--jinja"]
sampling = { temp = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0 }
```

**Embedding model:**

```toml
[models.qwen3-embedding-4b]
repo = "Qwen/Qwen3-Embedding-4B-GGUF"
file = "Qwen3-Embedding-4B-Q4_K_M.gguf"
kind = "embedding"
port = 8090
extra_args = ["--embeddings"]
```

**Sharded or large GGUF (override layers / context as needed):**

```toml
[models.gpt-oss-120b]
repo = "ggml-org/gpt-oss-120b-GGUF"
file = "gpt-oss-120b-MXFP4.gguf"
kind = "chat"
ctx_size = 131072
n_gpu_layers = 70
port = 8082
extra_args = ["--jinja"]
```

(If a repo publishes multi-part `*-00001-of-00003.gguf` shards, point `file` at the first shard; llama.cpp loads siblings automatically. Download helpers match shard siblings by basename.)

Argv is merged in layers: **defaults ← kind flags ← per-model overrides ← CLI flags**.

### Escape hatches (no registry entry needed)

```bash
# Any Hugging Face GGUF repo (downloaded via huggingface_hub, then served with `-m`)
uv run local-llm serve --hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M --port 8099

# Local file already on disk
uv run local-llm serve --model-path /opt/models/mistral-7b-instruct.Q4_K_M.gguf --port 8083

# Pass unknown llama-server flags through
uv run local-llm serve gpt-oss-20b -- --verbose --metrics
```

## Why `121a-real` (Spark only)

gpt-oss models use **MXFP4**. Native block-scale MXFP4 CUDA kernels only compile for Blackwell’s **`sm_121a`** target. Building with plain `121` often fails at `ptxas` with:

`Feature '.block_scale' not supported on .target 'sm_121'`

[`scripts/build.sh`](scripts/build.sh) therefore configures:

```bash
-DCMAKE_CUDA_ARCHITECTURES=121a-real
```

If that still fails, it retries with `-DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_NATIVE=OFF` (works, but loses some prompt-processing speed). Do not “simplify” the primary arch back to `121` without knowing this.

Runtime also needs Blackwell compat libs on `LD_LIBRARY_PATH` (handled by `scripts/env.sh` / the CLI):

`/usr/local/cuda-13/compat` (or `/usr/local/cuda/compat`).

## Benchmarking and model comparison

Three different questions hide behind "which model is faster/better"; the CLI keeps them apart.

| Command | Measures | Does **not** measure |
| --- | --- | --- |
| `local-llm bench A B` | Raw pp/tg tokens per second via `llama-bench`, using the exact served batch/ubatch/ngl/flash-attn/KV settings. Saved to `state/bench/*.json` with build provenance. | Task quality, chat template, serving latency under load. |
| `local-llm eval sec …` | Accuracy on real 10-K/10-Q filings (XBRL ground truth), plus TTFT / prefill / decode by input length, cold vs warm prefix cache, and throughput under concurrency. | Anything outside filings. |
| `local-llm eval swe …` | Coding ability through the served endpoint in three tiers: HumanEval+/MBPP+ (evalplus), Aider polyglot (edit-format compliance), a fixed 50-instance SWE-bench Verified subset (mini-swe-agent + swebench harness). Records tool-call/format failure rates. | Full SWE-bench. |

Every eval run is written to `state/evals/<suite>/<model>/<timestamp>-<task>/` as `run.json`
(model, GGUF, llama.cpp commit, CUDA arch actually built, server `/props`, decoding settings,
config hash) plus `results.jsonl`. `local-llm eval report --suite sec|swe` tabulates the
latest finished run per model **per platform/backend** (`spark/cuda`, `halo/vulkan`,
`halo/hip`; API runs are platform-agnostic) and warns when runs used different configs. When
the latest gpt-oss-20b, gpt-oss-120b, and OpenAI/Terra extract runs are present, the report
adds a comparison: paired score, full-document vs oversized-filing fallback, per-tag /
per-company tables, and item-level disagreements. When the same model and task exist on more
than one platform/backend, a **Hardware** block puts them side by side: paired accuracy,
TTFT, total latency, prompt/decode throughput, context, GPU and build; for `perf` runs the
rows are per input length.

Rules the harness enforces so numbers stay comparable:

- **Bench refuses to run while a tracked server or foreign GPU process is alive** (`--force` to override).
- **Quality tasks decode at temperature 0 with a fixed seed**; per-model sampling from `models.toml` is only used for serving/perf.
- **The CUDA arch that actually built is recorded** (`build/spark-arch.txt`); a `121 + GGML_NATIVE=OFF` fallback build is flagged in reports rather than silently compared to `121a-real`.

### SEC suite

```bash
export LOCAL_LLM_EDGAR_USER_AGENT="local-llm you@example.com"   # EDGAR fair-access requirement
uv run local-llm eval sec fetch                                  # 12 companies × (1×10-K + 3×10-Q) + XBRL facts
uv run local-llm serve gpt-oss-20b
uv run local-llm eval sec run gpt-oss-20b --task extract-full    # whole filing in context
uv run local-llm eval sec run gpt-oss-20b --task extract-chunked # BM25 top-k chunks (works for 8k-ctx models)
uv run local-llm eval sec run gpt-oss-20b --task qa-financebench # needs the judge model served too
uv run local-llm eval sec perf gpt-oss-20b                       # 8k/32k/64k/100k tokens, cold vs warm, concurrency 1,4
uv run local-llm eval report --suite sec
```

Ground truth for `extract-*` comes from EDGAR's XBRL company facts (revenue, net income, EPS,
assets, cash flow, …), matched with 0.5 % tolerance; answers that are right except for a
thousands/millions scale error are counted separately (`off_by_scale`). In `extract-full`,
a filing that does not fit the served context degrades per filing rather than being skipped:
the financial-statements Item (8 for a 10-K) goes in if it fits, else the BM25 top-k chunks.
Each result records `mode` (`full` / `section` / `chunked`) and a `fallback` flag, and the
run summary breaks accuracy down `by_mode`, so partial-context answers (GS and STWD 10-Ks at
131K) stay distinguishable from full-document ones. Companies, tags, chunk size and tolerances
live in [`evals.toml`](evals.toml); prompts live in `evals/prompts/`.

#### Compare with an OpenAI frontier model

The OpenAI path reuses the exact fetched filings, prompts, numeric ground truth, tolerance,
and report format. The API key is read only from the environment and is never written to a
run artifact. Replace the model ID and context limit with values supported by your account:

```bash
export OPENAI_API_KEY="..."

# Cheap smoke test first: one extraction question.
uv run local-llm eval sec run <OPENAI_MODEL> \
  --provider openai --context-window 128000 \
  --task extract-full --forms 10-K --limit 1

# Full pinned 10-K set.
uv run local-llm eval sec run <OPENAI_MODEL> \
  --provider openai --context-window 128000 \
  --task extract-full --forms 10-K

# Shows the latest local and OpenAI runs together.
uv run local-llm eval report --suite sec
```

`LOCAL_LLM_OPENAI_CONTEXT_WINDOW` changes the default context budget, and `OPENAI_BASE_URL`
overrides `https://api.openai.com/v1`. OpenAI runs are named `openai:<model>` in reports.
They record score, skips, TTFT, end-to-end latency, input/output token usage, cached input
tokens, and reasoning tokens when the API returns them. External latency includes network
and provider queueing; local llama.cpp prompt throughput and remote API throughput are not
hardware-equivalent measurements. Context budgeting uses each model's own tokenizer, so
token counts—and therefore the set of oversized filings skipped—can differ. The report's
**Paired** table rescores every run of a task on the items all of them answered, so use it
rather than the raw `score` column when skip sets differ. Frontier reasoning models do not
consistently support fixed temperature or seed, so the OpenAI path records and uses provider
decoding defaults. Use `--limit 1` before a full run to validate model access, context size,
and likely cost.

Reasoning models (gpt-oss, frontier) spend hidden reasoning tokens from `max_tokens` before
the visible answer. The `truncated` column counts items that hit the budget with no answer;
if it is non-zero, raise `[quality].max_tokens` in `evals.toml` (or `--max-tokens`) or lower
`--reasoning-effort low`. Local runs record the streamed `reasoning_content` length and
token count per item so the split between thinking and answering is visible.

Filing work needs context: a 10-K is roughly 50k–150k tokens. Raise `ctx_size` in
`models.toml` (and consider `cache_type_k`/`cache_type_v = "q8_0"`, `n_parallel`) for the
models you want on the full-document path. llama-server splits `--ctx-size` across
`--parallel` slots.

### SWE suite

```bash
uv run local-llm eval swe check gpt-oss-20b              # tool-call smoke test through --jinja; run first
uv run local-llm eval swe run gpt-oss-20b --tier 1       # evalplus humaneval+mbpp (minutes)
uv run local-llm eval swe run gpt-oss-20b --tier 2       # aider polyglot (Docker, ~1 h)
uv run local-llm eval swe run gpt-oss-20b --tier 3       # SWE-bench Verified ×50 (Docker, hours)
uv run local-llm eval swe run gpt-oss-20b --tier 3 --dry-run   # print the harness commands only
uv run local-llm eval report --suite swe
```

External harnesses run via `uvx` with the versions pinned in `evals.toml` and are never
vendored. Tiers 2 and 3 need Docker; the SWE-bench scoring images are x86_64-first, so the
intended layout is: Spark serves the model, an x86 box runs `local-llm eval swe … --host <spark>`
(or `LOCAL_LLM_EVAL_HOST`). The Tier 3 instance list is a seeded sample committed in
`evals.toml` so every model sees the same 50 tasks. Tier 2/3 command lines were written from
the harnesses' documented interfaces but have not been executed on this dev box (no Docker);
run `--dry-run` and check them against the pinned versions before a long run.

## Running several models at once

Each registry entry has its own `port`. Unified memory on the Spark can hold a chat model and an embedding model together:

```bash
uv run local-llm serve gpt-oss-20b qwen3-embedding-4b
uv run local-llm stop          # all tracked
uv run local-llm stop gpt-oss-20b
```

PIDs and logs live under `state/`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `cmake` / CUDA not found | Toolkit not on PATH or old cmake | `export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH`; `uv tool install 'cmake>=3.30'` |
| Build errors about GPU arch / `block_scale` | Wrong `CMAKE_CUDA_ARCHITECTURES` | Use `121a-real`; allow build.sh fallback |
| CUDA OOM on start | Context or model too large | Lower `--ctx-size`, reduce `--n-gpu-layers`, or pick a smaller quant |
| High latency | Layers on CPU | Ensure `--n-gpu-layers` is high enough; watch `nvidia-smi` during a request |
| `curl: (7) Failed to connect` | Server not up / wrong port | Wait for health; `local-llm doctor`; check `state/<name>.log` |
| GGUF download stalls | Network / HF | Re-run `local-llm download …` (resumable) |

## Project layout

```
llama-cpp-spark/
  models.toml          # model registry, DGX Spark (edit this to add models)
  models.halo.toml     # same models, Strix Halo serving knobs (auto-selected on Windows)
  evals.toml           # eval suites: SEC companies/tags, SWE tiers (data, not code)
  evals/prompts/       # prompt templates used by the evals
  LLAMA_CPP_VERSION    # pinned llama.cpp commit (Spark source build)
  LLAMA_CPP_RELEASE    # matching llama.cpp release tag (Halo prebuilt zips)
  scripts/build.sh     # CUDA native build (Spark)
  scripts/env.sh       # LD_LIBRARY_PATH for GB10 (Spark)
  src/spark_llm/       # CLI + argv merge + download + bench (shared)
  src/spark_llm/platforms/spark/  # Linux/CUDA: process control, nvidia-smi, build.sh, doctor
  src/spark_llm/platforms/halo/   # Windows/AMD: zip installer, taskkill/CIM, doctor
  src/spark_llm/evals/ # endpoint client, run records, SEC + SWE suites, report
  state/bench, state/evals  # persisted results (gitignored)
  tests/               # no-GPU unit tests
  vendor/llama.cpp/    # gitignored clone + build tree
```

## License

Project glue code: use freely. llama.cpp and model weights retain their upstream licenses.
