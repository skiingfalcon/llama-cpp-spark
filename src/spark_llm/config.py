"""Application settings (SPARK_LLM_* env vars / .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """Return the project root (directory containing models.toml)."""
    # src/spark_llm/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPARK_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: Path = Path("/opt/models")
    models_toml: Path = Field(default_factory=lambda: repo_root() / "models.toml")
    vendor_dir: Path = Field(default_factory=lambda: repo_root() / "vendor" / "llama.cpp")
    state_dir: Path = Field(default_factory=lambda: repo_root() / "state")
    host: str = "0.0.0.0"
    base_port: int = 8080
    n_gpu_layers: int = 999
    health_timeout_s: float = 300.0
    health_poll_s: float = 1.0

    # Evaluation harness
    evals_toml: Path = Field(default_factory=lambda: repo_root() / "evals.toml")
    evals_data_dir: Path = Field(default_factory=lambda: repo_root() / "evals" / "data")
    eval_host: str = "127.0.0.1"  # set to the Spark's address when driving from another box
    # EDGAR requires a descriptive UA: "App Name contact@email" (SPARK_LLM_EDGAR_USER_AGENT)
    edgar_user_agent: str | None = None


def get_settings() -> Settings:
    return Settings()
