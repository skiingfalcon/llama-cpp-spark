"""The platform contract plus small shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from spark_llm.config import Settings


@dataclass
class GpuProcess:
    pid: int | None
    name: str
    used_mib: str
    raw: str


@dataclass
class DoctorCheck:
    label: str
    ok: bool
    detail: str
    note: bool = False  # informational line; never fails the doctor


@dataclass
class BuildOptions:
    """Options for ``local-llm build``; only the Halo installer reads most of them."""

    backends: list[str] = field(default_factory=list)  # empty = platform default
    tag: str | None = None
    source: str = "official"  # official | lemonade (rocm only)
    asset: str | None = None
    force: bool = False


class Platform(Protocol):
    """Everything that differs between boxes. Shared code calls these and nothing else."""

    name: str
    memory_metric: str  # "gpu_mem" (dedicated VRAM) or "unified_mem" (UMA boxes)

    def supported_backends(self) -> list[str]: ...
    def backend(self, settings: Settings) -> str: ...
    def release_tag(self, settings: Settings) -> str | None: ...

    def default_models_dir(self) -> Path: ...
    def default_models_toml(self) -> Path: ...
    def eval_timeout_s(self) -> float: ...

    def binary_path(self, settings: Settings) -> Path: ...
    def bench_binary(self, settings: Settings) -> Path: ...
    def runtime_env(self, settings: Settings) -> dict[str, str]: ...
    def popen_kwargs(self) -> dict[str, Any]: ...
    def terminate(self, pid: int) -> bool: ...
    def pid_alive(self, pid: int) -> bool: ...

    def gpu_info(self) -> dict[str, str]: ...
    def gpu_memory_used_mib(self) -> int | None: ...
    def compute_apps(self) -> list[GpuProcess]: ...

    def build_arch(self, settings: Settings) -> str | None: ...
    def build(self, settings: Settings, opts: BuildOptions) -> int: ...
    def doctor_checks(self, settings: Settings) -> list[DoctorCheck]: ...
