"""Load and validate evals.toml — suites are data, not code."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from spark_llm.config import Settings, get_settings


class QualitySettings(BaseModel):
    """Fixed decoding for quality runs so models are compared on equal footing."""

    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512


class JudgeSettings(BaseModel):
    model: str = "gpt-oss-120b"
    max_tokens: int = 256


class Company(BaseModel):
    ticker: str
    cik: int


class XbrlTag(BaseModel):
    tag: str
    label: str
    kind: str = "duration"  # duration | instant
    unit: str = "USD"
    aliases: list[str] = Field(default_factory=list)


class FinanceBenchSettings(BaseModel):
    enabled: bool = True
    repo: str = "PatronusAI/financebench"
    file: str = "financebench_open_source.jsonl"
    license: str = "CC-BY-NC-4.0"


class SecSuite(BaseModel):
    companies: list[Company] = Field(default_factory=list)
    forms: dict[str, int] = Field(default_factory=lambda: {"10-K": 1, "10-Q": 3})
    chunk_tokens: int = 8000
    top_k: int = 6
    tolerance: float = 0.005
    max_input_tokens: int = 0  # 0 = model's served ctx
    perf_lengths: list[int] = Field(default_factory=lambda: [8192, 32768, 65536, 100000])
    xbrl_tags: list[XbrlTag] = Field(default_factory=list)
    financebench: FinanceBenchSettings = Field(default_factory=FinanceBenchSettings)


class SweTier(BaseModel):
    tool: str
    version: str | None = None
    datasets: list[str] = Field(default_factory=list)
    repo: str | None = None
    ref: str | None = None
    exercises_repo: str | None = None
    exercises_ref: str | None = None
    edit_format: str | None = None
    dataset: str | None = None
    limit: int | None = None
    seed: int | None = None
    instances: list[str] = Field(default_factory=list)
    workers: int = 4
    step_limit: int | None = None


class SweSuite(BaseModel):
    tier1: SweTier
    tier2: SweTier
    tier3: SweTier


class EvalConfig(BaseModel):
    quality: QualitySettings = Field(default_factory=QualitySettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    sec: SecSuite = Field(default_factory=SecSuite)
    swe: SweSuite


def load_eval_config(path: Path | None = None, settings: Settings | None = None) -> EvalConfig:
    settings = settings or get_settings()
    path = path or settings.evals_toml
    if not path.is_file():
        raise FileNotFoundError(f"evals.toml not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return EvalConfig.model_validate(data)


def prompts_dir() -> Path:
    from spark_llm.config import repo_root

    return repo_root() / "evals" / "prompts"


def load_prompt(name: str) -> str:
    path = prompts_dir() / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template missing: {path}")
    return path.read_text().strip()
