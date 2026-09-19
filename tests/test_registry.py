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
        "gemma-4-31b",
        "gemma-4-31b-nothink",
        "gemma-4-26b-a4b",
        "gemma-4-26b-a4b-nothink",
        "laguna-s-2.1",
        "laguna-s-2.1-thinking",
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


def _both_registries():
    from spark_llm.config import Settings, repo_root

    return {
        "spark": load_registry(),
        "halo": load_registry(settings=Settings(models_toml=repo_root() / "models.halo.toml")),
    }


def test_shared_ports_serve_the_same_weights() -> None:
    """Two entries may share a port only when they are twins of the same weights (they cannot
    run at the same time); anything else is a collision."""
    for label, reg in _both_registries().items():
        by_port: dict[int, set[tuple[str | None, str | None, str | None]]] = {}
        for spec in reg.models.values():
            if spec.port is None:
                continue
            by_port.setdefault(spec.port, set()).add((spec.repo, spec.file, spec.quant))
        clashes = {port: srcs for port, srcs in by_port.items() if len(srcs) > 1}
        assert not clashes, f"{label}: different weights on one port: {clashes}"


def test_nothink_twins_match_their_base() -> None:
    for label, reg in _both_registries().items():
        for name, twin in reg.models.items():
            if not name.endswith("-nothink"):
                continue
            base = reg.models.get(name.removesuffix("-nothink"))
            assert base is not None, f"{label}: {name} has no base entry"
            for field in ("port", "repo", "file", "quant", "ctx_size", "n_parallel"):
                assert getattr(twin, field) == getattr(base, field), f"{label}: {name}.{field}"
            args = twin.extra_args
            assert "--reasoning" in args and args[args.index("--reasoning") + 1] == "off", name


def test_gemma4_entries() -> None:
    reg = load_registry()
    dense, moe = reg.get("gemma-4-31b"), reg.get("gemma-4-26b-a4b")
    assert dense.repo == "google/gemma-4-31B-it-qat-q4_0-gguf" and dense.port == 8089
    assert moe.repo == "google/gemma-4-26B-A4B-it-qat-q4_0-gguf" and moe.port == 8091
    for spec in (dense, moe):
        assert (spec.file or "").endswith(".gguf") and spec.ctx_size == 131072
        assert spec.sampling is not None and spec.sampling.top_k == 64


def test_deepseek_entry_is_spark_only() -> None:
    regs = _both_registries()
    spec = regs["spark"].get("deepseek-v4-flash")
    assert spec.repo == "unsloth/DeepSeek-V4-Flash-0731-GGUF" and spec.quant == "UD-Q2_K_XL"
    assert spec.ctx_size == 131072 and spec.n_parallel == 1 and spec.port == 8092
    assert "deepseek-v4-flash" not in regs["halo"].models  # over the Halo's 96 GB VGM cap
    assert not any(s.port == 8092 for s in regs["halo"].models.values())  # port stays reserved


def test_laguna_entry() -> None:
    """Laguna's vendor default is thinking off, so the base entry is the primary row and the
    twin is named "-thinking" (turns thinking on), inverting every other model's "-nothink"
    convention. It also fits both platforms, unlike deepseek-v4-flash."""
    for label, reg in _both_registries().items():
        base, thinking = reg.get("laguna-s-2.1"), reg.get("laguna-s-2.1-thinking")
        assert base.repo == "unsloth/Laguna-S-2.1-GGUF" and base.quant == "UD-Q4_K_XL"
        assert base.ctx_size == 131072 and base.port == 8093 == thinking.port
        assert "--reasoning" not in base.extra_args, label
        args = thinking.extra_args
        assert "--chat-template-kwargs" in args, label
        assert args[args.index("--chat-template-kwargs") + 1] == '{"enable_thinking": true}', label
