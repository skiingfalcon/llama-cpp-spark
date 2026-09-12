"""Application settings (LOCAL_LLM_* env vars / .env).

Renamed from ``SPARK_LLM_*`` when the AMD Strix Halo box joined; the old prefix is still read at
lower priority so existing ``.env`` files keep working. Platform-dependent defaults (models dir,
registry file, eval timeout) come from :mod:`spark_llm.platforms`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from spark_llm import platforms

ENV_PREFIX = "LOCAL_LLM_"
LEGACY_ENV_PREFIX = "SPARK_LLM_"


def repo_root() -> Path:
    """Return the project root (directory containing models.toml)."""
    # src/spark_llm/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _platform():  # type: ignore[no-untyped-def]
    return platforms.current()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Platform-dependent defaults: /opt/models + models.toml on the Spark,
    # %LOCALAPPDATA%\local-llm\models + models.halo.toml on the Halo box.
    models_dir: Path = Field(default_factory=lambda: _platform().default_models_dir())
    models_toml: Path = Field(default_factory=lambda: _platform().default_models_toml())
    vendor_dir: Path = Field(default_factory=lambda: repo_root() / "vendor" / "llama.cpp")
    state_dir: Path = Field(default_factory=lambda: repo_root() / "state")
    host: str = "0.0.0.0"
    base_port: int = 8080
    n_gpu_layers: int = 999
    health_timeout_s: float = 300.0
    health_poll_s: float = 1.0
    # GPU backend of the llama.cpp binary to run: cuda (Spark), vulkan | hip (Halo). None = the
    # platform default. Recorded in provenance so runs on different backends stay separate.
    backend: str | None = None
    # Directory holding llama-server(.exe); overrides the platform's discovered location.
    llama_bin_dir: Path | None = None

    # Evaluation harness
    evals_toml: Path = Field(default_factory=lambda: repo_root() / "evals.toml")
    evals_data_dir: Path = Field(default_factory=lambda: repo_root() / "evals" / "data")
    eval_host: str = "127.0.0.1"  # set to the server box's address when driving from another box
    # Per-request timeout for eval calls; a cold 124K-token prefill on an iGPU can take a while.
    eval_timeout_s: float = Field(default_factory=lambda: _platform().eval_timeout_s())
    # EDGAR requires a descriptive UA: "App Name contact@email" (LOCAL_LLM_EDGAR_USER_AGENT)
    edgar_user_agent: str | None = None
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_context_window: int = 128000

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init > LOCAL_LLM_* env > SPARK_LLM_* env > .env LOCAL_LLM_* > .env SPARK_LLM_*
        legacy_env = EnvSettingsSource(settings_cls, env_prefix=LEGACY_ENV_PREFIX)
        legacy_dotenv = DotEnvSettingsSource(
            settings_cls,
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix=LEGACY_ENV_PREFIX,
        )
        return (
            init_settings,
            env_settings,
            legacy_env,
            dotenv_settings,
            legacy_dotenv,
            file_secret_settings,
        )


def get_settings() -> Settings:
    return Settings()
