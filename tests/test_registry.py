"""Registry loading and ModelSpec validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from spark_llm.registry import ModelKind, ModelSpec, load_registry


def test_load_seeded_registry() -> None:
    reg = load_registry()
    assert "gpt-oss-20b" in reg.models
    assert "qwen3-embedding-4b" in reg.models
    assert "gpt-oss-120b" in reg.models
    assert "qwen3-8b" in reg.models
    assert reg.defaults.n_gpu_layers == 999
    assert reg.models["gpt-oss-20b"].kind is ModelKind.chat
    assert reg.models["qwen3-embedding-4b"].kind is ModelKind.embedding
    assert "--jinja" in reg.models["gpt-oss-20b"].extra_args
    assert "--embeddings" in reg.models["qwen3-embedding-4b"].extra_args
    assert reg.models["gpt-oss-120b"].n_gpu_layers == 70
    assert reg.models["qwen3-8b"].quant == "Q4_K_M"
    assert reg.models["qwen3-8b"].hf_ref() == "unsloth/Qwen3-8B-GGUF:Q4_K_M"
    assert reg.models["gpt-oss-20b"].file == "gpt-oss-20b-MXFP4.gguf"


def test_unknown_model() -> None:
    reg = load_registry()
    with pytest.raises(KeyError, match="unknown model"):
        reg.get("nope")


def test_model_spec_allows_adhoc() -> None:
    # Used with ServeOverrides.hf / model_path; source comes from overrides.
    spec = ModelSpec(name="adhoc", kind=ModelKind.chat, port=8080)
    assert spec.file is None and spec.repo is None


def test_registry_entry_should_have_source() -> None:
    reg = load_registry()
    for name, spec in reg.models.items():
        assert spec.file or spec.repo or spec.quant, name


def test_large_model_file_on_120b() -> None:
    reg = load_registry()
    spec = reg.get("gpt-oss-120b")
    assert (spec.file or "").endswith(".gguf")
    assert "120b" in (spec.file or "").lower()
    assert spec.local_path(Path("/opt/models")).name.endswith(".gguf")
    assert spec.n_gpu_layers == 70


def test_halo_registry_mirrors_model_names_and_ports() -> None:
    from spark_llm.config import Settings, repo_root

    spark = load_registry()
    halo = load_registry(settings=Settings(models_toml=repo_root() / "models.halo.toml"))
    for name in (
        "gpt-oss-20b",
        "gpt-oss-120b",
        "qwen3-8b",
        "qwen3-embedding-4b",
        "nemotron-3-super",
    ):
        assert name in halo.models, name
        assert halo.models[name].port == spark.models[name].port
        assert halo.models[name].file == spark.models[name].file
    # Strix Halo serving knobs, not Spark's
    assert halo.defaults.host == "127.0.0.1" and halo.defaults.ubatch_size == 512
    assert halo.models["gpt-oss-120b"].n_gpu_layers is None  # inherits 999 from defaults
    assert "--no-mmap" in halo.models["gpt-oss-120b"].extra_args
    assert "--jinja" in halo.models["gpt-oss-120b"].extra_args


def test_nemotron_entry_keeps_the_whole_window_on_one_slot() -> None:
    reg = load_registry()
    spec = reg.get("nemotron-3-super")
    assert spec.repo == "ggml-org/nemotron-3-super-120b-GGUF"
    assert spec.ctx_size == 524288 and spec.n_parallel == 1 and spec.port == 8085
    assert spec.sampling is not None and spec.sampling.temp == 0.6
