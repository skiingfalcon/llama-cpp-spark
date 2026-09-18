"""Build/runtime provenance captured with every bench and eval run.

Numbers without this block are not comparable across rebuilds: build.sh can silently fall
back from ``121a-real`` to ``121 + GGML_NATIVE=OFF``, which changes prompt-processing speed.
With two machines in play the block also names the platform (spark / halo), the GPU backend
(cuda / vulkan / rocm) and, for prebuilt zips, the llama.cpp release tag.

Field names are platform-neutral. Files written before the rename used ``spark_llm_version`` and
``cuda_arch``; both are still accepted on load (aliases) and re-serialised under the new names.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from spark_llm import __version__
from spark_llm.config import Settings, repo_root
from spark_llm.gpu import gpu_info
from spark_llm.platforms import current

VERSION_FILE = "LLAMA_CPP_VERSION"


class Provenance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str
    hostname: str
    machine: str
    # Version of the harness that wrote the run (was ``spark_llm_version``).
    tool_version: str = Field(validation_alias=AliasChoices("tool_version", "spark_llm_version"))
    llama_cpp_pinned: str | None = None
    llama_cpp_checkout: str | None = None
    # What was built/installed: Spark "121a-real"; Halo "<backend>:<tag>:<source>"
    # (was ``cuda_arch``).
    build_id: str | None = Field(
        default=None, validation_alias=AliasChoices("build_id", "cuda_arch")
    )
    binary_version: str | None = None
    gpu: dict[str, str] = Field(default_factory=dict)
    # Added for the multi-box comparison; None on runs recorded before it existed (= spark/cuda).
    platform: str | None = None
    backend: str | None = None
    llama_cpp_release: str | None = None


def pinned_commit() -> str | None:
    path = repo_root() / VERSION_FILE
    return path.read_text().strip() if path.is_file() else None


def checkout_commit(settings: Settings) -> str | None:
    """HEAD of the llama.cpp tree the served binary came from: the LOCAL_LLM_LLAMA_BIN_DIR
    override (e.g. a fork build) when set, else the pinned vendor checkout."""
    tree = settings.llama_bin_dir or settings.vendor_dir

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(tree), *args], text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    if top == repo_root().resolve():
        return None  # a bare bin dir inside this project, not a llama.cpp clone
    try:
        return git("rev-parse", "--short=12", "HEAD")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def build_identity(settings: Settings) -> str | None:
    """Build identity recorded by the platform's build step (Spark: CUDA arch; Halo: zip tag)."""
    return current().build_arch(settings)


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
    plat = current()
    try:
        backend: str | None = plat.backend(settings)
    except ValueError:
        backend = settings.backend
    return Provenance(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        hostname=platform.node(),
        machine=platform.machine(),
        tool_version=__version__,
        llama_cpp_pinned=pinned_commit(),
        llama_cpp_checkout=checkout_commit(settings),
        build_id=build_identity(settings),
        binary_version=binary_version(binary, env) if binary else None,
        gpu=gpu_info() if with_gpu else {},
        platform=plat.name,
        backend=backend,
        llama_cpp_release=plat.release_tag(settings),
    )
