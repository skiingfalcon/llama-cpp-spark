"""Export the SEC suite configuration as JSON for non-Python runners.

``scripts/lmstudio.mjs`` (Node, locked-down boxes) cannot parse TOML, so it reads
``evals/generated/sec-config.json``. This module is the single source of that file; a test
asserts the committed copy matches ``evals.toml`` so the two cannot drift silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spark_llm.config import repo_root
from spark_llm.evals.config import EvalConfig
from spark_llm.evals.sec.questions import SCORING_VERSION

GENERATED_REL = Path("evals") / "generated" / "sec-config.json"


def export_sec_config(cfg: EvalConfig) -> dict[str, Any]:
    """The subset of evals.toml the extract-full task needs, in a stable key order."""
    q = cfg.quality
    return {
        "_generated_by": "local-llm eval sec export-config (do not edit; edit evals.toml)",
        "scoring_version": SCORING_VERSION,
        "companies": [{"ticker": c.ticker, "cik": c.cik} for c in cfg.sec.companies],
        "forms": dict(cfg.sec.forms),
        "tags": [
            {
                "tag": t.tag,
                "label": t.label,
                "kind": t.kind,
                "unit": t.unit,
                "aliases": list(t.aliases),
                "accept_aliases": t.accept_aliases,
            }
            for t in cfg.sec.xbrl_tags
        ],
        "tolerance": cfg.sec.tolerance,
        "chunk_tokens": cfg.sec.chunk_tokens,
        "top_k": cfg.sec.top_k,
        "quality": {
            "temperature": q.temperature,
            "seed": q.seed,
            "max_tokens": q.max_tokens,
            "reasoning_effort": q.reasoning_effort,
        },
    }


def render(cfg: EvalConfig) -> str:
    return json.dumps(export_sec_config(cfg), indent=2) + "\n"


def generated_path() -> Path:
    return repo_root() / GENERATED_REL


def write_generated(cfg: EvalConfig, path: Path | None = None) -> Path:
    path = path or generated_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cfg))
    return path
