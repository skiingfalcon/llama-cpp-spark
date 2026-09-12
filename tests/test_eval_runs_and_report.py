"""RunWriter persistence, config hashing, and cross-model report."""

from __future__ import annotations

from pathlib import Path

from spark_llm.config import Settings
from spark_llm.evals.config import load_eval_config
from spark_llm.evals.report import build_report, latest_runs
from spark_llm.evals.runs import RunWriter, config_hash, list_runs, load_results, load_run
from spark_llm.provenance import Provenance


def _settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state", vendor_dir=tmp_path / "vendor")


def _prov() -> Provenance:
    return Provenance(
        timestamp="2026-09-12T00:00:00+00:00",
        hostname="spark",
        machine="aarch64",
        spark_llm_version="0.1.0",
        llama_cpp_checkout="82d6bb284d1f",
        cuda_arch="121a-real",
    )


def test_run_writer_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    w = RunWriter(
        settings,
        "sec",
        "extract-full",
        "gpt-oss-20b",
        server={"n_ctx_per_slot": 131072},
        quality={"temperature": 0.0},
        task_config={"tags": ["Revenues"]},
        provenance=_prov(),
    )
    w.write({"id": "a", "correct": True})
    w.write({"id": "b", "correct": False})
    rec = w.finish({"n": 2, "score": 0.5})
    assert rec.n_results == 2 and rec.finished is not None
    loaded = load_run(w.dir)
    assert loaded.model == "gpt-oss-20b" and loaded.summary["score"] == 0.5
    assert loaded.provenance.cuda_arch == "121a-real"
    assert [r["id"] for r in load_results(w.dir)] == ["a", "b"]
    assert list_runs(settings, "sec") == [w.dir]
    assert list_runs(settings, "swe") == []


def test_config_hash_is_stable_and_sensitive() -> None:
    a = config_hash("t", {"x": 1}, {"y": [1, 2]})
    assert a == config_hash("t", {"x": 1}, {"y": [1, 2]})
    assert a != config_hash("t", {"x": 2}, {"y": [1, 2]})


def test_report_picks_latest_per_model_and_flags_config_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for model, score, tags in (
        ("a", 0.4, ["Revenues"]),
        ("a", 0.6, ["Revenues"]),
        ("b", 0.9, ["Assets"]),
    ):
        w = RunWriter(
            settings,
            "sec",
            "extract-full",
            model,
            task_config={"tags": tags},
            provenance=_prov(),
            server={"n_ctx_per_slot": 8192},
        )
        w.write({"id": "x", "correct": True})
        w.finish({"n": 1, "score": score, "ttft_p50_s": 1.5})
    dirs = latest_runs(settings, "sec")
    assert len(dirs) == 2
    table, md = build_report(dirs)
    assert "| b | extract-full | 1 | 0.9 |" in md
    assert "| a | extract-full | 1 | 0.6 |" in md  # latest run for a, not 0.4
    assert "different configs" in md  # a and b used different tag lists


def test_evals_toml_loads() -> None:
    cfg = load_eval_config()
    assert len(cfg.sec.companies) == 12
    assert {t.kind for t in cfg.sec.xbrl_tags} <= {"instant", "duration", "ytd"}
    assert cfg.swe.tier3.limit == 50 and len(cfg.swe.tier3.instances) == 50
    assert cfg.swe.tier1.datasets == ["humaneval", "mbpp"]
    assert cfg.quality.temperature == 0.0
