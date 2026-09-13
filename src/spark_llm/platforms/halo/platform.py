"""Strix Halo on Windows: prebuilt llama-server zips, Windows process control, CIM telemetry.

Design notes (see README "Windows / AMD Strix Halo"):
- Two backends may be installed side by side (``vendor/llama.cpp/halo-build.json``);
  ``settings.backend`` picks one per serve so Vulkan and ROCm runs are recorded separately.
- On a unified-memory APU the number that matters is total memory in use, not "GPU memory";
  :meth:`gpu_memory_used_mib` returns that and the perf task labels it ``unified_mem``.
- Everything here imports on Linux so it can be unit-tested; Windows-only calls are guarded.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spark_llm.config import Settings, repo_root
from spark_llm.platforms.base import BuildOptions, DoctorCheck, GpuProcess

BUILD_FILE = "halo-build.json"
BACKENDS = ("vulkan", "rocm")
BACKEND_ALIASES = {
    "hip": "rocm"
}  # llama.cpp calls the backend HIP; zips, docs and LM Studio say ROCm
DEFAULT_BACKEND = "vulkan"
SERVER_EXE = "llama-server.exe"
BENCH_EXE = "llama-bench.exe"
# subprocess exposes these only on Windows; the numeric values are stable Win32 constants.
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x8)
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MIN_UNIFIED_MEMORY_MIB = 64 * 1024

Runner = Callable[..., subprocess.CompletedProcess[str]]


def load_build_info(settings: Settings) -> dict[str, Any]:
    path = settings.vendor_dir / BUILD_FILE
    if not path.is_file():
        return {"backends": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"backends": {}}
    data.setdefault("backends", {})
    return data


def save_build_info(settings: Settings, info: dict[str, Any]) -> Path:
    settings.vendor_dir.mkdir(parents=True, exist_ok=True)
    path = settings.vendor_dir / BUILD_FILE
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path


class HaloPlatform:
    name = "halo"
    memory_metric = "unified_mem"

    def __init__(self, runner: Runner | None = None) -> None:
        self._run: Runner = runner or subprocess.run

    # -- identity ------------------------------------------------------------------------
    def supported_backends(self) -> list[str]:
        return list(BACKENDS)

    def backend(self, settings: Settings) -> str:
        value = (settings.backend or DEFAULT_BACKEND).lower()
        value = BACKEND_ALIASES.get(value, value)
        if value not in BACKENDS:
            choices = " | ".join(BACKENDS)
            raise ValueError(f"backend {value!r} not supported on halo; use {choices}")
        return value

    def backend_entry(self, settings: Settings) -> dict[str, Any] | None:
        return load_build_info(settings)["backends"].get(self.backend(settings))

    def release_tag(self, settings: Settings) -> str | None:
        entry = self.backend_entry(settings)
        return entry.get("tag") if entry else None

    # -- defaults -------------------------------------------------------------------------
    def default_models_dir(self) -> Path:
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home()
        return root / "local-llm" / "models"

    def default_models_toml(self) -> Path:
        return repo_root() / "models.halo.toml"

    def eval_timeout_s(self) -> float:
        return 3600.0  # a 124K-token cold prefill on this iGPU can take tens of minutes

    # -- binaries and process control --------------------------------------------------------
    def bin_dir(self, settings: Settings) -> Path:
        if settings.llama_bin_dir is not None:
            return settings.llama_bin_dir
        entry = self.backend_entry(settings)
        if entry and entry.get("bin_dir"):
            return Path(entry["bin_dir"])
        return settings.vendor_dir / "bin"

    def binary_path(self, settings: Settings) -> Path:
        return self.bin_dir(settings) / SERVER_EXE

    def bench_binary(self, settings: Settings) -> Path:
        return self.bin_dir(settings) / BENCH_EXE

    def runtime_env(self, settings: Settings) -> dict[str, str]:
        """DLLs (Vulkan loader, HIP runtime) sit next to the exe; put that dir first on PATH."""
        env = os.environ.copy()
        bin_dir = str(self.bin_dir(settings))
        existing = env.get("PATH", "")
        env["PATH"] = bin_dir + (os.pathsep + existing if existing else "")
        return env

    def popen_kwargs(self) -> dict[str, Any]:
        # Detach from the console so the server survives the CLI exiting; no setsid on Windows.
        # Forcing the halo platform on another OS is for introspection/tests only; Popen on
        # POSIX rejects creationflags, so hand back nothing there.
        if sys.platform != "win32":
            return {}
        return {"creationflags": CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS}

    def terminate(self, pid: int) -> bool:
        """taskkill the process tree; False when it was already gone (or never ours)."""
        try:
            res = self._run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True
            )
        except OSError:
            return False
        return res.returncode == 0

    def pid_alive(self, pid: int) -> bool:
        if sys.platform == "win32":
            return self._pid_alive_win32(pid)
        return self._pid_alive_tasklist(pid)

    @staticmethod
    def _pid_alive_win32(pid: int) -> bool:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    def _pid_alive_tasklist(self, pid: int) -> bool:
        try:
            res = self._run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        return res.returncode == 0 and f'"{pid}"' in (res.stdout or "")

    # -- telemetry (PowerShell CIM; no nvidia-smi / rocm-smi on Windows) -----------------------
    def _powershell(self, script: str) -> Any | None:
        exe = shutil.which("pwsh") or shutil.which("powershell")
        if not exe:
            return None
        try:
            res = self._run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode != 0 or not (res.stdout or "").strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def video_controllers(self) -> list[dict[str, Any]]:
        data = self._powershell(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,DriverVersion,DriverDate,AdapterRAM | ConvertTo-Json -Compress"
        )
        if data is None:
            return []
        return data if isinstance(data, list) else [data]

    def os_memory(self) -> dict[str, Any] | None:
        data = self._powershell(
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"
        )
        return data if isinstance(data, dict) else None

    @staticmethod
    def _pick_gpu(controllers: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not controllers:
            return None
        for c in controllers:
            name = str(c.get("Name", ""))
            if "Radeon" in name or "AMD" in name:
                return c
        return controllers[0]

    def gpu_info(self) -> dict[str, str]:
        gpu = self._pick_gpu(self.video_controllers())
        info: dict[str, str] = {}
        if gpu:
            info["name"] = str(gpu.get("Name", ""))
            info["driver_version"] = str(gpu.get("DriverVersion", ""))
            if gpu.get("DriverDate"):
                info["driver_date"] = str(gpu["DriverDate"])
        mem = self.os_memory()
        if mem and mem.get("TotalVisibleMemorySize"):
            # KiB from CIM: the unified pool the iGPU draws from (VGM is a cap, not a split).
            info["memory.total"] = str(int(int(mem["TotalVisibleMemorySize"]) / 1024))
        return info

    def gpu_memory_used_mib(self) -> int | None:
        mem = self.os_memory()
        if not mem:
            return None
        try:
            total = int(mem["TotalVisibleMemorySize"])
            free = int(mem["FreePhysicalMemory"])
        except (KeyError, TypeError, ValueError):
            return None
        return int((total - free) / 1024)

    def compute_apps(self) -> list[GpuProcess]:
        return []  # no per-process GPU accounting exposed on Windows for the iGPU

    # -- build (release-zip install) ----------------------------------------------------------
    def build_arch(self, settings: Settings) -> str | None:
        entry = self.backend_entry(settings)
        if not entry:
            return None
        return f"{self.backend(settings)}:{entry.get('tag', '?')}:{entry.get('source', '?')}"

    def build(self, settings: Settings, opts: BuildOptions) -> int:
        from spark_llm.platforms.halo.install import install

        return install(settings, opts)

    # -- doctor -----------------------------------------------------------------------------
    def doctor_checks(self, settings: Settings) -> list[DoctorCheck]:
        checks: list[DoctorCheck] = []
        checks.append(DoctorCheck("os", sys.platform == "win32", sys.platform))
        info = load_build_info(settings)
        backends = info["backends"]
        checks.append(
            DoctorCheck(
                "backends installed",
                bool(backends),
                ", ".join(
                    f"{b} ({e.get('tag', '?')}, {e.get('source', '?')})"
                    for b, e in backends.items()
                )
                or "none (run: local-llm build)",
            )
        )
        for backend, entry in backends.items():
            exe = Path(entry.get("bin_dir", "")) / SERVER_EXE
            checks.append(DoctorCheck(f"{backend} llama-server", exe.is_file(), str(exe)))
            devices = entry.get("devices") or []
            checks.append(
                DoctorCheck(
                    f"{backend} devices",
                    any("Radeon" in d or "AMD" in d for d in devices),
                    "; ".join(devices) or "none listed at install time",
                )
            )
        try:
            selected = self.backend(settings)
        except ValueError as exc:
            checks.append(DoctorCheck("selected backend", False, str(exc)))
        else:
            checks.append(
                DoctorCheck(
                    "selected backend",
                    selected in backends or not backends,
                    selected + ("" if selected in backends else " (not installed)"),
                )
            )
        gpu = self._pick_gpu(self.video_controllers())
        if gpu:
            name = str(gpu.get("Name", ""))
            checks.append(
                DoctorCheck(
                    "gpu",
                    any(k in name for k in ("Radeon", "8060S", "8050S", "AMD")),
                    f"{name} driver {gpu.get('DriverVersion', '?')}",
                )
            )
            if "rocm" in backends:
                checks.append(
                    DoctorCheck(
                        "rocm driver note",
                        True,
                        "ROCm 7.14+ zips need Adrenalin 26.6.4 or newer; CIM reports the WDDM "
                        f"driver ({gpu.get('DriverVersion', '?')}); check the Adrenalin version "
                        "by hand",
                        note=True,
                    )
                )
        else:
            checks.append(DoctorCheck("gpu", False, "Win32_VideoController query failed"))
        mem = self.os_memory()
        if mem and mem.get("TotalVisibleMemorySize"):
            total_mib = int(int(mem["TotalVisibleMemorySize"]) / 1024)
            checks.append(
                DoctorCheck(
                    "unified memory",
                    total_mib >= MIN_UNIFIED_MEMORY_MIB,
                    f"{total_mib} MiB visible to Windows",
                )
            )
        checks.append(
            DoctorCheck(
                "variable graphics memory",
                True,
                "cannot be read programmatically: Adrenalin > Performance > Tuning; set high "
                "(>= 96 GB) before serving gpt-oss-120b",
                note=True,
            )
        )
        return checks


def parse_version_output(text: str) -> tuple[str | None, str | None]:
    """(first line, short commit) from ``llama-server --version`` output."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None
    m = re.search(r"\(([0-9a-f]{7,40})\)", text)
    return lines[0][:160], (m.group(1) if m else None)


def parse_devices_output(text: str) -> list[str]:
    """Device lines from ``llama-server --list-devices`` (``Vulkan0: AMD Radeon ...``)."""
    devices: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^(Vulkan|ROCm|HIP|CUDA|SYCL|OpenCL|CPU)\d*:", s):
            devices.append(s)
    return devices
