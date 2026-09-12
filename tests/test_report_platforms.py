"""Report behaviour with runs from more than one platform/backend."""

from __future__ import annotations

from pathlib import Path

from spark_llm.config import Settings
from spark_llm.evals.report import build_report, latest_runs, platform_label
from spark_llm.evals.runs import RunWriter, load_run
from spark_llm.provenance import Provenance


def _settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state", vendor_dir=tmp_path / "vendor")


def _prov(platform: str | None, backend: str | None, gpu: str = "GB10") -> Provenance:
    return Provenance(
        timestamp="2026-09-13T00:00:00+00:00",
        hostname="box",
        machine="x",
        spark_llm_version="0.1.0",
        llama_cpp_pinned="82d6bb284d1f",
        platform=platform,
        backend=backend,
        llama_cpp_release="b10919" if platform == "halo" else None,
        gpu={"name": gpu},
    )


def _run(settings, model, task, prov, items, summary_extra=None, server=None):  # type: ignore[no-untyped-def]
    w = RunWriter(
        settings, "sec", task, model, provenance=prov, server=server or {"n_ctx_per_slot": 131072}
    )
    for rec in items:
        w.write(rec)
    correct = [r for r in items if r.get("correct")]
    scored = [r for r in items if r.get("correct") is not None and not r.get("skipped")]
    w.finish(
        {
            "n": len(items),
            "score": len(correct) / len(scored) if scored else None,
            "truncated": 0,
            "ttft_p50_s": 1.0,
            "total_p50_s": 5.0,
            "total_p95_s": 50.0,
            "prompt_tps_p50": 250.0,
            "decode_tps_p50": 30.0,
            **(summary_extra or {}),
        }
    )
    return w.dir


def _items(*verdicts: bool | None) -> list[dict]:
    return [
        {
            "id": f"AAPL:10-K:{i}",
            "ticker": "AAPL",
            "tag": "Assets",
            "skipped": v is None,
            "correct": v,
        }
        for i, v in enumerate(verdicts, 1)
    ]


def test_legacy_runs_are_spark_cuda_and_api_runs_are_platform_agnostic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    d = _run(settings, "gpt-oss-120b", "extract-full", _prov(None, None), _items(True))
    assert platform_label(load_run(d)) == "spark/cuda"
    d = _run(
        settings,
        "openai:gpt-5.6-terra",
        "extract-full",
        _prov("halo", "vulkan"),
        _items(True),
        server={"provider": "openai"},
    )
    assert platform_label(load_run(d)) == "api"


def test_latest_runs_keeps_one_per_platform_backend(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _run(settings, "gpt-oss-120b", "extract-full", _prov("spark", "cuda"), _items(True, True))
    _run(settings, "gpt-oss-120b", "extract-full", _prov("halo", "vulkan"), _items(True, False))
    _run(settings, "gpt-oss-120b", "extract-full", _prov("halo", "hip"), _items(True, True))
    dirs = latest_runs(settings, "sec")
    assert len(dirs) == 3
    labels = sorted(platform_label(load_run(d)) for d in dirs)
    assert labels == ["halo/hip", "halo/vulkan", "spark/cuda"]


def test_hardware_block_lists_each_platform_with_paired_scores(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    # Spark answered 3 (all right); Halo/vulkan skipped item 3 and missed item 2.
    _run(settings, "gpt-oss-120b", "extract-full", _prov("spark", "cuda"), _items(True, True, True))
    _run(
        settings,
        "gpt-oss-120b",
        "extract-full",
        _prov("halo", "vulkan", gpu="AMD Radeon(TM) 8060S Graphics"),
        _items(True, False, None),
    )
    table_md = build_report(latest_runs(settings, "sec"))[1]
    assert "| platform |" in table_md.splitlines()[0]
    assert "## Hardware: same model, different box" in table_md
    assert "### gpt-oss-120b extract-full" in table_md
    # paired on items 1-2: spark 2/2, halo 1/2
    assert "| spark/cuda | 1 | 2/2 (1.000) |" in table_md
    assert "| halo/vulkan | 0.5 | 1/2 (0.500) |" in table_md
    assert "AMD Radeon(TM) 8060S Graphics" in table_md and "b10919/" in table_md


def test_hardware_block_absent_with_single_platform(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _run(settings, "gpt-oss-120b", "extract-full", _prov("spark", "cuda"), _items(True))
    _run(settings, "gpt-oss-20b", "extract-full", _prov("spark", "cuda"), _items(True))
    md = build_report(latest_runs(settings, "sec"))[1]
    assert "Hardware:" not in md


def test_perf_hardware_block_is_per_length(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for plat, backend, pp in (("spark", "cuda", 900.0), ("halo", "vulkan", 300.0)):
        w = RunWriter(
            settings, "sec", "perf", "gpt-oss-120b", provenance=_prov(plat, backend), server={}
        )
        for length in (8192, 32768):
            w.write(
                {
                    "kind": "cold",
                    "length": length,
                    "fits": True,
                    "ttft_s": 1.5,
                    "prompt_tps": pp,
                    "decode_tps": 30.0,
                }
            )
            w.write(
                {"kind": "warm", "length": length, "fits": True, "ttft_s": 0.2, "prompt_tps": None}
            )
        w.finish({"n": 4})
    md = build_report(latest_runs(settings, "sec"))[1]
    assert "### gpt-oss-120b perf" in md
    assert "| length | halo/vulkan | spark/cuda |" in md
    assert "| 8192 | ttft 1.5s / pp 300 / tg 30 t/s | ttft 1.5s / pp 900 / tg 30 t/s |" in md


def test_comparison_groups_split_by_platform(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for model in ("gpt-oss-20b", "gpt-oss-120b"):
        _run(settings, model, "extract-full", _prov("spark", "cuda"), _items(True, True))
        _run(settings, model, "extract-full", _prov("halo", "vulkan"), _items(True, False))
    _run(
        settings,
        "openai:gpt-5.6-terra",
        "extract-full",
        _prov("spark", "cuda"),
        _items(True, True),
        server={"provider": "openai"},
    )
    md = build_report(latest_runs(settings, "sec"))[1]
    assert "### extract-full [spark/cuda]: gpt-oss vs Terra" in md
    assert "### extract-full [halo/vulkan]: gpt-oss vs Terra" in md
