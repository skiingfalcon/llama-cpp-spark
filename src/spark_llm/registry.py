"""Load and validate models.toml — models are data, not code."""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spark_llm.config import Settings, get_settings


class ModelKind(StrEnum):
    chat = "chat"
    embedding = "embedding"
    multimodal = "multimodal"
    rerank = "rerank"


class Sampling(BaseModel):
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


class Defaults(BaseModel):
    n_gpu_layers: int = 999
    flash_attn: str = "on"
    batch_size: int = 2048
    ubatch_size: int = 2048
    host: str = "0.0.0.0"
    # llama-server splits --ctx-size across -np slots; keep in mind when raising n_parallel.
    n_parallel: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None


class ModelSpec(BaseModel):
    """One entry under [models.<name>] in models.toml."""

    name: str
    repo: str | None = None
    file: str | None = None
    quant: str | None = None
    kind: ModelKind = ModelKind.chat
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    mmproj: str | None = None
    sampling: Sampling | None = None
    extra_args: list[str] = Field(default_factory=list)
    port: int | None = None
    flash_attn: str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    host: str | None = None
    n_parallel: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None

    def local_path(self, models_dir: Path) -> Path | None:
        if self.file:
            return models_dir / self.file
        return None

    def hf_ref(self) -> str | None:
        """llama.cpp -hf style reference, e.g. org/repo:Q4_K_M."""
        if not self.repo:
            return None
        if self.quant:
            return f"{self.repo}:{self.quant}"
        return self.repo

    def snapshot_dir(self, models_dir: Path) -> Path | None:
        """Where ``download_model`` places a repo/quant snapshot (no explicit file)."""
        if not self.repo:
            return None
        return models_dir / self.repo.replace("/", "__")


class Registry(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    models: dict[str, ModelSpec] = Field(default_factory=dict)

    def get(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise KeyError(f"unknown model {name!r}; known: {known}") from exc


def _parse_models_section(raw: dict[str, Any], defaults: Defaults) -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"[models.{name}] must be a table, got {type(body).__name__}")
        data = dict(body)
        data["name"] = name
        # Fill port from base if missing — caller can still override
        if "port" not in data or data["port"] is None:
            # leave None; server layer assigns
            pass
        out[name] = ModelSpec.model_validate(data)
    return out


def load_registry(path: Path | None = None, settings: Settings | None = None) -> Registry:
    settings = settings or get_settings()
    path = path or settings.models_toml
    if not path.is_file():
        raise FileNotFoundError(f"models.toml not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    defaults = Defaults.model_validate(data.get("defaults") or {})
    models_raw = data.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError("[models] must be a table")
    models = _parse_models_section(models_raw, defaults)
    return Registry(defaults=defaults, models=models)
