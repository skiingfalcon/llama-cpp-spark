"""nvidia-smi helpers shared by doctor, bench, and the eval harness."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from spark_llm.config import Settings


@dataclass
class GpuProcess:
    pid: int | None
    name: str
    used_mib: str
    raw: str


def _smi(args: list[str]) -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        return subprocess.check_output([smi, *args], text=True, timeout=10).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def gpu_info() -> dict[str, str]:
    """Static + live GPU facts for provenance. Empty dict when nvidia-smi is unavailable."""
    fields = [
        "name",
        "driver_version",
        "compute_cap",
        "memory.total",
        "memory.used",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.draw",
    ]
    out = _smi([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not out:
        return {}
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(fields, values, strict=False))


def gpu_memory_used_mib() -> int | None:
    out = _smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        return int(out.splitlines()[0].strip())
    except ValueError:
        return None


def compute_apps() -> list[GpuProcess]:
    out = _smi(["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"])
    procs: list[GpuProcess] = []
    if not out:
        return procs
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        pid: int | None
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            pid = None
        procs.append(
            GpuProcess(
                pid=pid,
                name=parts[1] if len(parts) > 1 else "",
                used_mib=parts[2] if len(parts) > 2 else "",
                raw=line,
            )
        )
    return procs


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        reasons.append(f"tracked server {name} is running (pid {pid}); run: spark-llm stop")
    _, foreign = split_gpu_processes(settings)
    for proc in foreign:
        reasons.append(f"foreign GPU process: {proc.raw}")
    return reasons
