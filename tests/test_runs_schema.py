"""run.json schema: flat Node adapter, tolerant loading, directory labels, backend names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from spark_llm.config import Settings
from spark_llm.evals.report import build_report, latest_runs
from spark_llm.evals.runs import (
    RunRecord,
    RunWriter,
    load_run,
    normalise_backend,
    run_label,
)
from spark_llm.provenance import Provenance

FLAT = {
    "task": "extract-full",
    "model": "openai/gpt-oss-120b",
    "model_label": "gpt-oss-120b-halo-rocm",
    "platform": "halo",
    "backend": "llama.cpp-win-x86_64-amd-rocm-avx2",
    "runtime_version": "2.37.0",
    "served_via": "lmstudio",
    "quant": "MXFP4",
    "context": 131072,
    "decoding": {"temperature": 0, "seed": 42, "max_tokens": 4096},
    "tolerance": 0.005,
    "chunk_tokens": 8000,
    "top_k": 6,
    "token_estimate": "chars/4.6 (approximate)",
    "started": "20260913T044559Z",
    "n": 121,
    "correct": 115,
    "skipped": 0,
    "truncated": 0,
    "fallback": 17,
    "off_by_scale": 1,
    "score": 0.9504,
    "by_mode": {"full": {"n": 104, "correct": 101}, "section": {"n": 17, "correct": 14}},
    "by_form": {"10-K": {"n": 121, "correct": 115}},
    "ttft_p50": 1.193,
    "total_p50": 7.427,
    "total_p95": 238.354,
    "prompt_tps_p50": 68791.0,
    "decode_tps_p50": 19.4,
    "prompt_tokens": 9976288,
    "reasoning_chars": 57469,
    "wall_clock_s": 4311.501,
    "cached_prompt_tokens": None,
    "config_hash": "58b2946660ca",
}


def _settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state", vendor_dir=tmp_path / "vendor")


def _prov(platform: str | None = None, backend: str | None = None) -> Provenance:
    return Provenance(
        timestamp="2026-09-13T00:00:00+00:00",
        hostname="box",
        machine="x",
        spark_llm_version="0.1.0",
        platform=platform,
        backend=backend,
    )


def test_from_flat_maps_summary_provenance_and_times() -> None:
    rec = RunRecord.from_flat(FLAT)
    assert rec.suite == "sec" and rec.task == "extract-full"
    assert rec.model == "gpt-oss-120b" and rec.model_label == "gpt-oss-120b-halo-rocm"
    assert rec.provenance.build_id == "rocm:lmstudio-2.37.0"
    assert rec.provenance.platform == "halo" and rec.provenance.backend == "rocm"
    assert rec.provenance.llama_cpp_release == "2.37.0"
    assert rec.started == "2026-09-13T04:45:59+00:00"
    assert rec.finished == "2026-09-13T05:57:50+00:00"  # started + 4311.5 s
    s = rec.summary
    assert s["n"] == 121 and s["score"] == pytest.approx(0.9504)
    assert s["ttft_p50_s"] == 1.193 and s["total_p95_s"] == 238.354
    assert s["total_tokens"] == 9976288 and s["by_mode"]["section"]["correct"] == 14
    assert rec.server["n_ctx_per_slot"] == 131072 and rec.server["served_via"] == "lmstudio"
    assert rec.runtime["model_id"] == "openai/gpt-oss-120b" and rec.runtime["quant"] == "MXFP4"
    assert rec.quality["max_tokens"] == 4096 and rec.task_config["tolerance"] == 0.005
    assert rec.n_results == 121 and rec.config_hash == "58b2946660ca"


def test_load_run_adapts_flat_file_and_rejects_garbage(tmp_path: Path) -> None:
    d = tmp_path / "flat"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(FLAT))
    assert load_run(d).provenance.backend == "rocm"
    g = tmp_path / "garbage"
    g.mkdir()
    (g / "run.json").write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValidationError):
        load_run(g)


def test_report_skips_unloadable_run_instead_of_crashing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    w = RunWriter(
        settings, "sec", "extract-full", "m", provenance=_prov("spark", "cuda"), server={}
    )
    w.write({"id": "1", "correct": True, "skipped": False})
    w.finish({"n": 1, "score": 1.0})
    bad = settings.state_dir / "evals" / "sec" / "junk" / "20260101T000000Z-extract-full"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text('{"hello": "world"}')
    dirs = latest_runs(settings, "sec")
    assert dirs == [w.dir]
    _, md = build_report(dirs + [bad])  # a bad dir passed explicitly is skipped too
    assert "| m | extract-full |" in md


def test_run_writer_labels_directory_by_platform_and_backend(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    w = RunWriter(settings, "sec", "extract-full", "gpt-oss-120b", provenance=_prov("halo", "rocm"))
    assert w.dir.parent.name == "gpt-oss-120b-halo-rocm"
    assert w.record.model_label == "gpt-oss-120b-halo-rocm" and w.record.format == "runrecord/1"
    w.finish({"n": 0})
    assert load_run(w.dir).model_label == "gpt-oss-120b-halo-rocm"
    api = RunWriter(
        settings,
        "sec",
        "extract-full",
        "openai:gpt-5.6-terra",
        provenance=_prov("spark", "cuda"),
        server={"provider": "openai"},
    )
    assert api.dir.parent.name == "openai_gpt-5.6-terra"  # API runs are platform-agnostic
    legacy = RunWriter(settings, "sec", "extract-full", "gpt-oss-20b", provenance=_prov())
    assert legacy.dir.parent.name == "gpt-oss-20b"  # no platform recorded -> plain name


def test_run_label_and_backend_normalisation() -> None:
    assert run_label("gpt-oss-120b", "spark", "cuda", api=False) == "gpt-oss-120b-spark-cuda"
    assert run_label("openai:x", "halo", "vulkan", api=True) == "openai_x"
    assert normalise_backend("llama.cpp-win-x86_64-amd-rocm-avx2") == "rocm"
    assert normalise_backend("llama.cpp-win-x86_64-vulkan-avx2") == "vulkan"
    assert normalise_backend("HIP") == "rocm" and normalise_backend("cuda13") == "cuda"
    assert normalise_backend(None) is None and normalise_backend("metal") == "metal"


def test_run_writer_context_manager_marks_aborted_runs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(RuntimeError):
        with RunWriter(
            settings, "sec", "extract-full", "m", provenance=_prov("spark", "cuda")
        ) as w:
            w.write({"id": "1", "correct": True, "skipped": False})
            raise RuntimeError("boom")
    rec = load_run(w.dir)
    assert rec.finished is not None and rec.summary["aborted"].startswith("RuntimeError: boom")
    assert rec.summary["n"] == 1
    # a normal run through the context manager is untouched
    with RunWriter(settings, "sec", "extract-full", "m2", provenance=_prov("spark", "cuda")) as w2:
        w2.finish({"n": 0, "score": None})
    assert "aborted" not in load_run(w2.dir).summary
