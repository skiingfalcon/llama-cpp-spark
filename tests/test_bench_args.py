"""llama-bench argv must mirror the served configuration — no GPU required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spark_llm.bench import BenchOptions, bench_argv, parse_bench_json, usable_depths
from spark_llm.config import Settings
from spark_llm.download import resolve_weights
from spark_llm.registry import Defaults, ModelKind, ModelSpec
from spark_llm.server import build_argv


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    vendor = tmp_path / "vendor" / "llama.cpp"
    bindir = vendor / "build" / "bin"
    bindir.mkdir(parents=True)
    for b in ("llama-server", "llama-bench"):
        (bindir / b).write_text("#!/bin/true\n")
    models = tmp_path / "models"
    models.mkdir()
    return Settings(
        models_dir=models,
        vendor_dir=vendor,
        state_dir=tmp_path / "state",
        models_toml=tmp_path / "models.toml",
    )


def _val(argv: list[str], flag: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def test_bench_uses_served_batch_ngl_and_kv(settings: Settings) -> None:
    gguf = settings.models_dir / "gpt-oss-120b-MXFP4.gguf"
    gguf.write_bytes(b"fake")
    spec = ModelSpec(
        name="gpt-oss-120b",
        file=gguf.name,
        kind=ModelKind.chat,
        n_gpu_layers=70,
        ctx_size=65536,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        port=8082,
    )
    defaults = Defaults(batch_size=2048, ubatch_size=2048, flash_attn="on")
    argv, rt, skipped = bench_argv(spec, defaults, settings, BenchOptions())
    assert _val(argv, "-m") == str(gguf)
    assert _val(argv, "-ngl") == "70"
    assert _val(argv, "-b") == "2048" and _val(argv, "-ub") == "2048"
    assert _val(argv, "-fa") == "1"
    assert _val(argv, "-ctk") == "q8_0" and _val(argv, "-ctv") == "q8_0"
    assert _val(argv, "-o") == "json"
    assert _val(argv, "-d") == "0,16384"
    assert skipped == [65536]  # 65536 + pp4096 would exceed ctx 65536
    # Same merged values the server sees
    serve = build_argv(spec, defaults, settings)
    assert _val(serve, "--n-gpu-layers") == "70"
    assert _val(serve, "--batch-size") == "2048"
    assert _val(serve, "--cache-type-k") == "q8_0"
    assert rt.n_gpu_layers == 70


def test_embedding_kind_sets_embd(settings: Settings) -> None:
    gguf = settings.models_dir / "emb.gguf"
    gguf.write_bytes(b"fake")
    spec = ModelSpec(name="emb", file="emb.gguf", kind=ModelKind.embedding, port=8090)
    argv, _, _ = bench_argv(spec, Defaults(), settings)
    assert _val(argv, "-embd") == "1"


def test_quant_only_spec_resolves_from_snapshot(settings: Settings) -> None:
    spec = ModelSpec(name="qwen3-8b", repo="unsloth/Qwen3-8B-GGUF", quant="Q4_K_M", port=8081)
    snap = settings.models_dir / "unsloth__Qwen3-8B-GGUF"
    snap.mkdir()
    (snap / "Qwen3-8B-Q8_0.gguf").write_bytes(b"x")
    wanted = snap / "Qwen3-8B-Q4_K_M.gguf"
    wanted.write_bytes(b"x")
    assert resolve_weights(spec, settings.models_dir) == wanted
    argv, _, _ = bench_argv(spec, Defaults(), settings)
    assert _val(argv, "-m") == str(wanted)
    # serve now prefers the local snapshot over -hf too
    serve = build_argv(spec, Defaults(), settings)
    assert _val(serve, "-m") == str(wanted)
    assert "-hf" not in serve


def test_quant_only_spec_missing_is_clear_error(settings: Settings) -> None:
    spec = ModelSpec(name="qwen3-8b", repo="unsloth/Qwen3-8B-GGUF", quant="Q4_K_M", port=8081)
    with pytest.raises(FileNotFoundError, match="local-llm download qwen3-8b"):
        bench_argv(spec, Defaults(), settings)


def test_sharded_snapshot_picks_first_shard(settings: Settings) -> None:
    spec = ModelSpec(name="big", repo="org/Big-GGUF", quant="Q4_K_M")
    snap = settings.models_dir / "org__Big-GGUF"
    snap.mkdir()
    for n in ("00002-of-00002", "00001-of-00002"):
        (snap / f"Big-Q4_K_M-{n}.gguf").write_bytes(b"x")
    assert resolve_weights(spec, settings.models_dir).name == "Big-Q4_K_M-00001-of-00002.gguf"


def test_n_parallel_flows_to_server(settings: Settings) -> None:
    gguf = settings.models_dir / "m.gguf"
    gguf.write_bytes(b"x")
    spec = ModelSpec(name="m", file="m.gguf", n_parallel=4, port=8080)
    serve = build_argv(spec, Defaults(), settings)
    assert _val(serve, "--parallel") == "4"


@pytest.mark.parametrize(
    ("depths", "pp", "ctx", "kept", "skipped"),
    [
        ([0, 16384, 65536], [512, 4096], None, [0, 16384, 65536], []),
        ([0, 16384, 65536], [512, 4096], 0, [0, 16384, 65536], []),
        ([0, 16384, 65536], [512, 4096], 8192, [0], [16384, 65536]),
        ([0, 16384], [512], 32768, [0, 16384], []),
    ],
)
def test_usable_depths(depths, pp, ctx, kept, skipped) -> None:
    assert usable_depths(depths, pp, ctx) == (kept, skipped)


def test_parse_bench_json_tolerates_noise() -> None:
    payload = [
        {
            "model_filename": "x.gguf",
            "n_prompt": 512,
            "n_gen": 0,
            "n_depth": 0,
            "avg_ts": 1234.5,
            "stddev_ts": 3.2,
        }
    ]
    raw = "ggml_cuda_init: found 1 CUDA devices\n" + json.dumps(payload)
    assert parse_bench_json(raw) == payload
