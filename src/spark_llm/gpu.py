"""GPU telemetry and process liveness shared by doctor, bench, and the eval harness.

Thin dispatch layer: the nvidia-smi implementation lives in ``platforms/spark``, the Windows CIM
implementation in ``platforms/halo``. Callers keep importing these names.
"""

from __future__ import annotations

from spark_llm.config import Settings
from spark_llm.platforms import current
from spark_llm.platforms.base import GpuProcess

__all__ = [
    "GpuProcess",
    "busy_reasons",
    "compute_apps",
    "gpu_info",
    "gpu_memory_used_mib",
    "live_tracked_servers",
    "pid_alive",
    "split_gpu_processes",
]


def gpu_info() -> dict[str, str]:
    """Static + live GPU facts for provenance. Empty dict when no telemetry tool is available."""
    return current().gpu_info()


def gpu_memory_used_mib() -> int | None:
    """Dedicated VRAM in use (Spark) or unified memory in use (Halo); see Platform.memory_metric."""
    return current().gpu_memory_used_mib()


def compute_apps() -> list[GpuProcess]:
    return current().compute_apps()


def pid_alive(pid: int) -> bool:
    return current().pid_alive(pid)


def live_tracked_servers(settings: Settings) -> dict[str, int]:
    """name -> pid for servers in servers.json whose process still exists."""
    from spark_llm.server import load_state  # local import: server imports gpu lazily too

    return {
        name: int(info["pid"])
        for name, info in load_state(settings).items()
        if pid_alive(int(info["pid"]))
    }


def split_gpu_processes(settings: Settings) -> tuple[list[GpuProcess], list[GpuProcess]]:
    """(ours, foreign) compute apps, keyed on servers.json PIDs."""
    tracked = set(live_tracked_servers(settings).values())
    ours: list[GpuProcess] = []
    foreign: list[GpuProcess] = []
    for proc in compute_apps():
        (ours if proc.pid in tracked else foreign).append(proc)
    return ours, foreign


def busy_reasons(settings: Settings) -> list[str]:
    """Human-readable reasons the GPU is not idle; empty list means safe to benchmark."""
    reasons: list[str] = []
    for name, pid in live_tracked_servers(settings).items():
        reasons.append(f"tracked server {name} is running (pid {pid}); run: local-llm stop")
    _, foreign = split_gpu_processes(settings)
    for proc in foreign:
        reasons.append(f"foreign GPU process: {proc.raw}")
    return reasons
