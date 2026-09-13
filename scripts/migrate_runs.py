#!/usr/bin/env python3
"""One-off: bring every committed state/evals/**/run.json onto the RunRecord schema.

- Flat files from early ``scripts/lmstudio.mjs`` versions are rewritten through
  ``RunRecord.from_flat``.
- Files with hand-added top-level ``platform`` / ``backend`` keys (the Sept 2026 folder relabel)
  get those values moved into ``provenance`` where the loader actually reads them.
- Everything else is re-serialised canonically (adds ``format`` / ``model_label``).

Idempotent. Run from the repo root: ``uv run python scripts/migrate_runs.py [--check]``.
``--check`` exits 1 if any file would change (for CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from spark_llm.config import get_settings
from spark_llm.evals.runs import (
    RunRecord,
    evals_root,
    is_flat_record,
    normalise_backend,
    run_label,
)


def migrate_one(path: Path) -> tuple[str, bool, str]:
    data = json.loads(path.read_text())
    if is_flat_record(data):
        rec = RunRecord.from_flat(data)
        note = "flat -> runrecord"
    else:
        prov = data.setdefault("provenance", {})
        moved = []
        for key in ("platform", "backend"):
            if key in data and not prov.get(key):
                prov[key] = data[key]
                moved.append(key)
        if prov.get("backend"):
            prov["backend"] = normalise_backend(prov["backend"])
        rec = RunRecord.model_validate(data)
        note = f"moved {', '.join(moved)} into provenance" if moved else "canonicalised"
    if not rec.model_label:
        rec.model_label = run_label(
            rec.model,
            rec.provenance.platform,
            rec.provenance.backend,
            api=bool(rec.server.get("provider")),
        )
    new = rec.model_dump_json(indent=2) + "\n"
    changed = new != path.read_text()
    return note, changed, new


def main(argv: list[str]) -> int:
    check = "--check" in argv
    root = evals_root(get_settings())
    rc = 0
    for path in sorted(root.glob("*/*/*/run.json")):
        note, changed, new = migrate_one(path)
        rel = path.relative_to(root)
        if not changed:
            print(f"ok       {rel}")
            continue
        if check:
            print(f"DIFFERS  {rel} ({note})")
            rc = 1
        else:
            path.write_text(new)
            print(f"rewrote  {rel} ({note})")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
