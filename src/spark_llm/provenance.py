"""Build/runtime provenance captured with every bench and eval run.

Numbers without this block are not comparable across rebuilds: build.sh can silently fall
back from ``121a-real`` to ``121 + GGML_NATIVE=OFF``, which changes prompt-processing speed.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from spark_llm import __version__
from spark_llm.config import Settings, repo_root
from spark_llm.gpu import gpu_info

ARCH_FILE = "spark-arch.txt"


class Provenance(BaseModel):
    timestamp: str
    hostname: str
    machine: str
    spark_llm_version: str
    llama_cpp_pinned: str | None = None
    llama_cpp_checkout: str | None = None
    cuda_arch: str | None = None
    binary_version: str | None = None
    gpu: dict[str, str] = Field(default_factory=dict)


def pinned_commit() -> str | None:
    path = repo_root() / "LLAMA_CPP_VERSION"
    return path.read_text().strip() if path.is_file() else None


def checkout_commit(settings: Settings) -> str | None:
    if not (settings.vendor_dir / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(settings.vendor_dir), "rev-parse", "--short=12", "HEAD"],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def built_cuda_arch(settings: Settings) -> str | None:
    """CUDA arch recorded by scripts/build.sh; None for builds predating that change."""
    path = settings.vendor_dir / "build" / ARCH_FILE
    return path.read_text().strip() if path.is_file() else None


def binary_version(binary: Path, env: dict[str, str] | None = None) -> str | None:
    if not binary.is_file():
        return None
    try:
        out = subprocess.check_output(
            [str(binary), "--version"], text=True, stderr=subprocess.STDOUT, env=env, timeout=20
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        out = getattr(exc, "output", None) or ""
    lines = [ln for ln in str(out).splitlines() if ln.strip()]
    return lines[0][:160] if lines else None


def collect(
    settings: Settings,
    binary: Path | None = None,
    env: dict[str, str] | None = None,
    with_gpu: bool = True,
) -> Provenance:
    return Provenance(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        hostname=platform.node(),
        machine=platform.machine(),
        spark_llm_version=__version__,
        llama_cpp_pinned=pinned_commit(),
        llama_cpp_checkout=checkout_commit(settings),
        cuda_arch=built_cuda_arch(settings),
        binary_version=binary_version(binary, env) if binary else None,
        gpu=gpu_info() if with_gpu else {},
    )
