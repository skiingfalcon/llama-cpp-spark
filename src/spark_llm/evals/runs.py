"""Persist eval runs under state/evals/<suite>/<model-label>/<timestamp>-<task>/.

Two writers produce ``run.json``: this module (Python harness) and ``scripts/lmstudio.mjs``
(Node, for locked-down boxes). Both emit the :class:`RunRecord` shape (``format: runrecord/1``).
Runs written by earlier versions of the Node script used a flat schema; :func:`load_run`
adapts those transparently via :meth:`RunRecord.from_flat`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from spark_llm.config import Settings
from spark_llm.provenance import Provenance, collect

RECORD_FORMAT = "runrecord/1"
_UNSAFE = re.compile(r'[:\\/*?"<>|]')
# One name per GPU backend across CLI, provenance, folders and docs. LM Studio reports its
# runtime pack name (e.g. "llama.cpp-win-x86_64-amd-rocm-avx2"); llama.cpp calls the same
# backend "HIP". We say "rocm".
BACKEND_ALIASES = {"hip": "rocm", "amd-rocm": "rocm", "cuda12": "cuda", "cuda13": "cuda"}


def path_safe(name: str) -> str:
    """Model names such as 'openai:gpt-5.6-terra' must not reach the filesystem verbatim."""
    return _UNSAFE.sub("_", name)


def config_hash(*parts: Any) -> str:
    """Stable short hash of everything that should be equal for two runs to be comparable."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def normalise_backend(raw: str | None) -> str | None:
    """Map vendor/runtime strings onto the canonical backend names (cuda | vulkan | rocm)."""
    if not raw:
        return None
    s = raw.lower()
    if s in BACKEND_ALIASES:
        return BACKEND_ALIASES[s]
    for needle, canon in (
        ("vulkan", "vulkan"),
        ("rocm", "rocm"),
        ("hip", "rocm"),
        ("cuda", "cuda"),
    ):
        if needle in s:
            return canon
    return s


def run_label(model: str, platform: str | None, backend: str | None, api: bool) -> str:
    """Directory name for a run: ``<model>-<platform>-<backend>`` locally, ``<model>`` for APIs."""
    base = path_safe(model)
    if api or not platform:
        return base
    return f"{base}-{platform}-{backend}" if backend else f"{base}-{platform}"


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
    # Added with the multi-box work; absent on older files.
    format: str = RECORD_FORMAT
    model_label: str | None = None  # directory name, e.g. gpt-oss-120b-halo-rocm

    @classmethod
    def from_flat(cls, data: dict[str, Any]) -> RunRecord:
        """Adapt the flat run.json written by early versions of ``scripts/lmstudio.mjs``.

        That schema put summary metrics at the top level (``ttft_p50`` without the ``_s``
        suffix), had no ``suite``/``provenance``, used ``model_label`` for the directory name and
        ``model`` for the LM Studio model id, and a compact ``started`` stamp.
        """
        started = _iso(data.get("started"))
        wall = data.get("wall_clock_s")
        finished = None
        if started and isinstance(wall, int | float):
            finished = (datetime.fromisoformat(started) + timedelta(seconds=wall)).isoformat(
                timespec="seconds"
            )
        backend_raw = data.get("backend")
        platform = data.get("platform") or "halo"
        backend = normalise_backend(backend_raw)
        version = data.get("runtime_version")
        prov = Provenance(
            timestamp=started or "",
            hostname=str(data.get("hostname") or "?"),
            machine=str(data.get("machine") or "x86_64"),
            spark_llm_version=f"lmstudio.mjs/{version or '?'}",
            cuda_arch=f"{backend}:lmstudio-{version or '?'}",
            platform=platform,
            backend=backend,
            llama_cpp_release=version,
        )
        # ``model`` must be the bare model name so the same model on different boxes groups
        # together in the report; the directory label carries platform/backend.
        label = data.get("model_label")
        model = str(data["model"]).split("/")[-1]
        if label and backend and label.endswith(f"-{platform}-{backend}"):
            model = label[: -len(f"-{platform}-{backend}")]
        summary_keys = (
            "n",
            "correct",
            "score",
            "skipped",
            "truncated",
            "fallback",
            "off_by_scale",
            "by_mode",
            "by_form",
            "prompt_tps_p50",
            "decode_tps_p50",
            "cached_prompt_tokens",
            "reasoning_chars",
            "wall_clock_s",
        )
        summary: dict[str, Any] = {k: data[k] for k in summary_keys if k in data}
        for old, new in (
            ("ttft_p50", "ttft_p50_s"),
            ("total_p50", "total_p50_s"),
            ("total_p95", "total_p95_s"),
            ("prompt_tokens", "total_tokens"),
        ):
            if old in data:
                summary[new] = data[old]
        return cls(
            suite=data.get("suite") or "sec",
            task=data["task"],
            model=model,
            model_label=label,
            started=started or "",
            finished=finished,
            provenance=prov,
            server={
                "n_ctx_per_slot": data.get("context"),
                "served_via": data.get("served_via"),
                "runtime": backend_raw,
                "runtime_version": data.get("runtime_version"),
            },
            runtime={
                "served_via": data.get("served_via"),
                "model_id": data.get("model"),
                "quant": data.get("quant"),
                "backend_raw": backend_raw,
            },
            quality=dict(data.get("decoding") or {}),
            task_config={
                k: data[k]
                for k in ("tolerance", "chunk_tokens", "top_k", "token_estimate")
                if k in data
            },
            config_hash=data.get("config_hash", ""),
            summary=summary,
            n_results=int(data.get("n", 0)) + int(data.get("skipped", 0)),
        )


def _iso(stamp: Any) -> str | None:
    """Accept ISO-8601 or the compact ``20260913T044559Z`` form; return ISO with offset."""
    if not stamp:
        return None
    s = str(stamp)
    if re.fullmatch(r"\d{8}T\d{6}Z", s):
        return (
            datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat(timespec="seconds")
        )
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat(timespec="seconds")
    except ValueError:
        return s


def is_flat_record(data: dict[str, Any]) -> bool:
    return "provenance" not in data and ("model_label" in data or "served_via" in data)


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
        prov = provenance or collect(settings, with_gpu=True)
        server = server or {}
        label = run_label(model, prov.platform, prov.backend, api=bool(server.get("provider")))
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = evals_root(settings) / suite / label / f"{stamp}-{task}"
        self.dir = base
        n = 1
        while self.dir.exists():  # two runs in the same second must not share a directory
            n += 1
            self.dir = base.with_name(f"{base.name}-{n}")
        self.dir.mkdir(parents=True, exist_ok=False)
        self.record = RunRecord(
            suite=suite,
            task=task,
            model=model,
            model_label=label,
            started=prov.timestamp,
            provenance=prov,
            server=server,
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
    """Load run.json; flat (early Node script) files are adapted, anything else raises."""
    text = (run_dir / "run.json").read_text()
    try:
        return RunRecord.model_validate_json(text)
    except ValidationError:
        data = json.loads(text)
        if isinstance(data, dict) and is_flat_record(data):
            return RunRecord.from_flat(data)
        raise


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
