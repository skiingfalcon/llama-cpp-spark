"""Persist eval runs: state/evals/<suite>/<model>/<timestamp>/{run.json, results.jsonl}."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spark_llm.config import Settings
from spark_llm.provenance import Provenance, collect

_UNSAFE = re.compile(r'[:\\/*?"<>|]')


def path_safe(name: str) -> str:
    """Model names such as 'openai:gpt-5.6-terra' must not reach the filesystem verbatim."""
    return _UNSAFE.sub("_", name)


def config_hash(*parts: Any) -> str:
    """Stable short hash of everything that should be equal for two runs to be comparable."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class RunRecord(BaseModel):
    suite: str
    task: str
    model: str
    started: str
    finished: str | None = None
    provenance: Provenance
    server: dict[str, Any] = Field(default_factory=dict)  # /props snapshot: n_ctx, slots, path
    runtime: dict[str, Any] = Field(default_factory=dict)  # merged RuntimeParams
    quality: dict[str, Any] = Field(default_factory=dict)  # decoding settings used
    task_config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    tools: dict[str, str] = Field(default_factory=dict)  # external harness versions
    summary: dict[str, Any] = Field(default_factory=dict)
    n_results: int = 0


def evals_root(settings: Settings) -> Path:
    return settings.state_dir / "evals"


class RunWriter:
    """Create the run directory, stream results to JSONL, finalise run.json."""

    def __init__(
        self,
        settings: Settings,
        suite: str,
        task: str,
        model: str,
        *,
        server: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        task_config: dict[str, Any] | None = None,
        tools: dict[str, str] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dir = evals_root(settings) / suite / path_safe(model) / f"{stamp}-{task}"
        self.dir.mkdir(parents=True, exist_ok=True)
        prov = provenance or collect(settings, with_gpu=True)
        self.record = RunRecord(
            suite=suite,
            task=task,
            model=model,
            started=prov.timestamp,
            provenance=prov,
            server=server or {},
            runtime=runtime or {},
            quality=quality or {},
            task_config=task_config or {},
            tools=tools or {},
        )
        self.record.config_hash = config_hash(
            task, self.record.quality, self.record.task_config, self.record.runtime
        )
        self._results = (self.dir / "results.jsonl").open("a")
        self.flush_record()

    @property
    def results_path(self) -> Path:
        return self.dir / "results.jsonl"

    def flush_record(self) -> None:
        (self.dir / "run.json").write_text(self.record.model_dump_json(indent=2) + "\n")

    def write(self, result: dict[str, Any]) -> None:
        self._results.write(json.dumps(result, default=str) + "\n")
        self._results.flush()
        self.record.n_results += 1

    def finish(self, summary: dict[str, Any]) -> RunRecord:
        self.record.summary = summary
        self.record.finished = datetime.now(UTC).isoformat(timespec="seconds")
        self._results.close()
        self.flush_record()
        return self.record


def load_run(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json((run_dir / "run.json").read_text())


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def list_runs(settings: Settings, suite: str | None = None) -> list[Path]:
    root = evals_root(settings)
    if not root.is_dir():
        return []
    pattern = f"{suite}/*/*/run.json" if suite else "*/*/*/run.json"
    return sorted(p.parent for p in root.glob(pattern))
