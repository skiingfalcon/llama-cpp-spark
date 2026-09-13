# Published artifacts

Everything this project has published outside the repo, with where it went, when, and what came
back. One row per posting; drafts live next to this file and are committed before they go out,
so the text that was actually published is in git history.

## Log

| Date | Venue | Title / draft | Link | Status | Follow-ups captured |
| --- | --- | --- | --- | --- | --- |
| 2026-09-13 | r/LocalLLaMA (Reddit) | [gpt-oss-120b: DGX Spark vs AMD Strix Halo vs GPT-5.6 Terra on 10-K extraction](2026-09-13-localllama-spark-vs-halo.md) | _add after posting_ | draft — **blocked** on the `--no-document` control run and on quoting the bootstrap intervals | |
| 2026-09-13 | llama.cpp GitHub Discussions | [Strix Halo (gfx1151, Windows) Vulkan vs ROCm on gpt-oss-120b, 45K–124K-token prompts](2026-09-13-llamacpp-discussion-strix-halo.md) | _add after posting_ | draft | |

Status values: `draft` → `posted` → `closed` (feedback folded back into the reports or the code).

## How to add one

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `YYYY-MM-DD-<venue>-<slug>.md` and write the post there.
2. Add a row above with status `draft`; commit.
3. Post it. Paste the public link into the row, set status `posted`; commit.
4. When replies arrive, add the substantive ones to a `## Feedback` section at the bottom of the
   draft file (who said what, in one line each, no usernames beyond the handle they posted under)
   and note what changed in the repo because of them. When nothing more is coming, set `closed`.

## Where the numbers come from

Every figure in a post must trace to a committed run under `state/evals/` or to one of the
reports in [`docs/index.md`](../index.md). If a post needs a number that is not there yet, run
the eval first and commit the artifact, then write the post.
