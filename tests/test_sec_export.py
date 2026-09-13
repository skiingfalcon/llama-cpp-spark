"""The committed evals/generated/sec-config.json must match evals.toml (Node script reads it)."""

from __future__ import annotations

import json
from pathlib import Path

from spark_llm.evals.config import load_eval_config
from spark_llm.evals.sec.export import export_sec_config, generated_path, render, write_generated


def test_generated_config_is_current() -> None:
    cfg = load_eval_config()
    path = generated_path()
    assert path.is_file(), "run: uv run local-llm eval sec export-config"
    assert path.read_text() == render(cfg), (
        "evals/generated/sec-config.json is stale; run: uv run local-llm eval sec export-config"
    )


def test_export_shape(tmp_path: Path) -> None:
    cfg = load_eval_config()
    data = export_sec_config(cfg)
    assert {c["ticker"] for c in data["companies"]} >= {"AAPL", "GS", "STWD"}
    tags = {t["tag"]: t for t in data["tags"]}
    assert tags["Revenues"]["accept_aliases"] is True
    assert tags["LongTermDebtNoncurrent"]["aliases"] == ["LongTermDebtAndCapitalLeaseObligations"]
    assert data["quality"]["max_tokens"] == cfg.quality.max_tokens
    out = write_generated(cfg, tmp_path / "x" / "sec-config.json")
    assert json.loads(out.read_text())["top_k"] == cfg.sec.top_k
