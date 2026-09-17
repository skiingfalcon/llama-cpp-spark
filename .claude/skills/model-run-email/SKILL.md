---
name: model-run-email
description: Draft the team email that announces a new model eval run (SEC 10-K extraction or another suite) from committed run artifacts, in the house style, with every number traced to a run directory.
---

# Model-run email

Use when the user asks for "an email about the <model> run", "draft the update for the team",
or similar, after a run has landed under `state/evals/` and (ideally) has a report in `docs/`.

## Procedure

1. **Locate the evidence.** The run directories under `state/evals/<suite>/<model>-<platform>-<backend>/`,
   the report in `docs/eval-report-*.md`, the card in `docs/models/<model>.md`, and the
   machine table `state/evals/report-sec.md`. Pull master first if the run came from another box.
2. **Get the numbers from the artifacts, not from memory.** From the repo root:

   ```bash
   uv run python .claude/skills/model-run-email/facts.py <run-dir> [<run-dir> ...] \
       --baseline state/evals/sec/gpt-oss-120b-spark-cuda/20260912T132250Z-extract-full
   ```

   It prints, per run: headline and 95% CI, "when the filing fits" (104 questions, GS and STWD
   excluded), decode t/s, cold prefill t/s and cold TTFT range on 80K+ token filings, wall clock,
   truncations, reasoning tokens, by-mode counts, every miss, and the item-level diff against
   the baseline. Take the table row from this output. If a number the email needs is not in it,
   compute it from `results.jsonl` and say so in the fact trail.
3. **If the model is new or not widely known, introduce it** in one short paragraph before the
   results: who released it and when, open-weights licence, size (total and active parameters),
   the one architectural fact that matters for us (MoE vs dense, context window, hybrid layers),
   whether it reasons in hidden tokens, weight size on disk, and the single reason we tried it.
   Verify the release facts against the vendor's model card or Hugging Face page with a web
   search; cite the URL in the fact trail. Do not repeat the vendor's benchmark claims.
4. **Write the email** in the structure below, then save it to
   `docs/emails/YYYY-MM-DD-<model>.md` with a `## Fact trail` table (claim → run dir, report
   section, or URL) under a horizontal rule. Show the user the email body in chat.
5. **Fact-check pass before showing it:** every number in the prose appears in the facts output
   or the report; intervals are quoted when a gap is called real or noise; single run per row is
   stated; anything not yet run (memorisation control, judge pass) is named as outstanding.

## Structure (match the earlier emails)

- Greeting, one or two sentences on what ran and when (overnight on which machine), and what is
  being kicked off next. Leave a `[next run]` placeholder for the user to fill.
- Model introduction paragraph if needed (step 3).
- One sentence restating the task: "11 financial metrics from the latest 10-K of 12 companies,
  121 questions, scored against SEC XBRL ground truth at 0.5% tolerance. Single run per row;
  read gaps inside overlapping 95% intervals as noise."
- **What this says**: three to five bullets. Lead each with the finding in bold-free plain words,
  then the mechanism and the number that shows it. Cover: accuracy and where the misses are,
  what the fallback/context path did, speed and why (bandwidth, active parameters, reasoning
  length), and what the misses actually were.
- **Recommendation**: does the on-prem default change; when this model is the right pick; the one
  or two experiments that would change the answer, stated as runs with expected outcomes.
- The comparison table with the standard columns: Stack | Accuracy, 121 Qs | When the filing
  fits, of 104 Qs | Decode | Cold prefill, new ~100K filing | Full run | Cost per run. Keep all
  prior rows, add the new ones in bold, sort by accuracy. Explain the "fits" column in one line.
- Housekeeping notes (two at most) and the outstanding pre-publication items.
- **Write-ups**: report, model card, `docs/index.md`, `state/evals/report-sec.md`, as full
  GitHub URLs on `master`.

## Style

- Plain sentences, about 20 words, one idea each. No em-dashes, no parentheses for asides.
- Numbers belong in the table or at the end of a sentence, never three in one clause.
- Say "single run", "one seed", and quote the interval whenever a difference is interpreted.
- Name colleagues as "[colleague]" in the draft; the user fills in names.
- Never soften a bad result; never claim more than the run supports. If the model lost, say so
  in the first bullet.
- Do not regenerate `report-sec.md` or edit reports as part of this skill; if a report number
  disagrees with the artifact, flag it in chat instead.
