# Eval reports — which one to read

| Read this when… | Report | Status |
| --- | --- | --- |
| You want the current headline: Spark vs AMD Strix Halo (Vulkan, ROCm) vs hosted Terra on gpt-oss-120b | [eval-report-2026-09-spark-halo-terra.md](eval-report-2026-09-spark-halo-terra.md) | **latest** (Sept 13, 2026) |
| You need Spark/Terra detail: per-tag and per-company tables, the 20b model, every miss itemised, cost | [eval-report-2026-09-spark-cuda-rerun.md](eval-report-2026-09-spark-cuda-rerun.md) | current for the Spark and Terra columns |
| You want Qwen3.8-27B on the same Spark suite, including why it was ~5× slower than gpt-oss-120b | [eval-report-2026-09-spark-cuda-qwen.md](eval-report-2026-09-spark-cuda-qwen.md) | Qwen add-on (Sept 15, 2026) |
| You want Nemotron-3-Super on the same suite (131K-budget vs full 1M window) | [eval-report-2026-09-spark-cuda-nemotron.md](eval-report-2026-09-spark-cuda-nemotron.md) | Nemotron add-on (Sept 16–17, 2026) |
| You want Qwen3.8-27B on a single RTX 5090 (Windows, CUDA): the 131K replication and the native-262K run that recovers Goldman Sachs | [eval-report-2026-09-rtx5090-cuda-qwen.md](eval-report-2026-09-rtx5090-cuda-qwen.md) | Qwen on RTX 5090 (Sept 15, 2026) |
| You want the first FinanceBench and coding tier-1 (HumanEval/MBPP) runs, Qwen3.8-27B on the RTX 5090, reasoning on vs off, and the harness fixes they needed | [eval-report-2026-09-rtx5090-financebench-swe-qwen.md](eval-report-2026-09-rtx5090-financebench-swe-qwen.md) | first runs of both suites (Sept 15, 2026) |
| You want the pre-fix run and the explanation of what the harness fixes changed | [eval-report-2026-09-spark-cuda.md](eval-report-2026-09-spark-cuda.md) | superseded; kept for the cross-check |

Each number has one home: the Halo report owns the AMD columns, the re-run owns the Spark and
Terra columns, and the first report owns the pre-fix baseline. The machine-generated table over
all committed runs is [`state/evals/report-sec.md`](../state/evals/report-sec.md)
(`uv run local-llm eval report --suite sec` regenerates it).

`inference-stack-spark-vs-halo.png` is a binary export; there is no diagram source in the repo.

Every report now carries a "Review notes" block under its Caveats listing what the numbers
cannot support (single run, no memorisation control yet, unequal context, revised ground truth).

Per-model reference cards (identity, exact serving config, measured results, strengths and
weaknesses, changelog) live in [`models/`](models/README.md). Update a card whenever a run lands.

A talk-through slide deck of the results lives in [`deck/`](deck/README.md)
(`local-inference-2026-09.pdf`; Marp source alongside).

Posts and write-ups published outside the repo, with links and the feedback they drew, are
logged in [`posts/README.md`](posts/README.md).
