"""Table-driven argv merge tests across model kinds — no GPU required."""

from __future__ import annotations

from pathlib import Path

import pytest

from spark_llm.config import Settings
from spark_llm.registry import Defaults, ModelKind, ModelSpec, Sampling
from spark_llm.server import ServeOverrides, build_argv, kind_flags


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    vendor = tmp_path / "vendor" / "llama.cpp"
    bindir = vendor / "build" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "llama-server").write_text("#!/bin/true\n")
    (bindir / "llama-server").chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    return Settings(
        models_dir=models,
        vendor_dir=vendor,
        state_dir=tmp_path / "state",
        models_toml=tmp_path / "models.toml",
    )


def _has_flag_value(argv: list[str], flag: str, value: str) -> bool:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv) and argv[i + 1] == value:
            return True
    return False


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ModelKind.chat, []),
        (ModelKind.embedding, ["--embeddings"]),
        (ModelKind.rerank, ["--reranking"]),
        (ModelKind.multimodal, ["--jinja"]),
    ],
)
def test_kind_flags(kind: ModelKind, expected: list[str]) -> None:
    assert kind_flags(kind) == expected


def test_chat_model_argv(settings: Settings) -> None:
    gguf = settings.models_dir / "gpt-oss-20b-mxfp4.gguf"
    gguf.write_bytes(b"fake")
    spec = ModelSpec(
        name="gpt-oss-20b",
        file="gpt-oss-20b-mxfp4.gguf",
        kind=ModelKind.chat,
        ctx_size=0,
        port=8080,
        extra_args=["--jinja"],
    )
    argv = build_argv(spec, Defaults(), settings)
    assert _has_flag_value(argv, "-m", str(gguf))
    assert _has_flag_value(argv, "--port", "8080")
    assert _has_flag_value(argv, "--ctx-size", "0")
    assert _has_flag_value(argv, "--n-gpu-layers", "999")
    assert _has_flag_value(argv, "--flash-attn", "on")
    assert "--jinja" in argv
    assert "--embeddings" not in argv


def test_embedding_model_argv(settings: Settings) -> None:
    gguf = settings.models_dir / "emb.gguf"
    gguf.write_bytes(b"fake")
    spec = ModelSpec(
        name="emb",
        file="emb.gguf",
        kind=ModelKind.embedding,
        port=8090,
        extra_args=["--embeddings"],
    )
    argv = build_argv(spec, Defaults(), settings)
    assert argv.count("--embeddings") == 1  # deduped
    assert _has_flag_value(argv, "--port", "8090")


def test_sampling_and_ngl_override(settings: Settings) -> None:
    gguf = settings.models_dir / "q.gguf"
    gguf.write_bytes(b"fake")
    spec = ModelSpec(
        name="qwen",
        file="q.gguf",
        kind=ModelKind.chat,
        n_gpu_layers=70,
        sampling=Sampling(temp=0.6, top_p=0.95, top_k=20, min_p=0.0),
        extra_args=["--jinja"],
        port=8081,
    )
    argv = build_argv(spec, Defaults(), settings)
    assert _has_flag_value(argv, "--n-gpu-layers", "70")
    assert _has_flag_value(argv, "--temp", "0.6")
    assert _has_flag_value(argv, "--top-p", "0.95")
    assert _has_flag_value(argv, "--top-k", "20")
    assert _has_flag_value(argv, "--min-p", "0.0")


def test_cli_overrides_win(settings: Settings) -> None:
    gguf = settings.models_dir / "m.gguf"
    gguf.write_bytes(b"fake")
    other = settings.models_dir / "other.gguf"
    other.write_bytes(b"fake")
    spec = ModelSpec(name="m", file="m.gguf", kind=ModelKind.chat, port=8080, ctx_size=4096)
    overrides = ServeOverrides(
        port=9999,
        ctx_size=2048,
        n_gpu_layers=10,
        model_path=other,
        extra_args=["--verbose"],
    )
    argv = build_argv(spec, Defaults(), settings, overrides)
    assert _has_flag_value(argv, "-m", str(other))
    assert _has_flag_value(argv, "--port", "9999")
    assert _has_flag_value(argv, "--ctx-size", "2048")
    assert _has_flag_value(argv, "--n-gpu-layers", "10")
    assert "--verbose" in argv


def test_hf_passthrough(settings: Settings) -> None:
    overrides = ServeOverrides(hf="unsloth/Qwen3-14B-GGUF:Q4_K_M", port=8080)
    anon = ModelSpec(name="adhoc", kind=ModelKind.chat, port=8080, extra_args=["--jinja"])
    argv = build_argv(anon, Defaults(), settings, overrides)
    assert _has_flag_value(argv, "-hf", "unsloth/Qwen3-14B-GGUF:Q4_K_M")
    assert "-m" not in argv


def test_large_model_uses_file(settings: Settings) -> None:
    name = "gpt-oss-120b-MXFP4.gguf"
    gguf = settings.models_dir / name
    gguf.write_bytes(b"fake")
    spec = ModelSpec(
        name="gpt-oss-120b",
        file=name,
        kind=ModelKind.chat,
        n_gpu_layers=70,
        port=8082,
        extra_args=["--jinja"],
    )
    argv = build_argv(spec, Defaults(), settings)
    assert _has_flag_value(argv, "-m", str(gguf))
    assert _has_flag_value(argv, "--n-gpu-layers", "70")
