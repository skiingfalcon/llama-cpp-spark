"""DGX Spark platform: Linux process handling, CUDA compat libs, nvidia-smi telemetry, and the
CMake source build in ``scripts/build.sh``. These bodies were moved here verbatim from
``server.py`` / ``gpu.py`` / ``provenance.py`` / ``cli.doctor``; behaviour is unchanged.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from spark_llm.config import Settings, repo_root
from spark_llm.platforms.base import BuildOptions, DoctorCheck, GpuProcess

console = Console(stderr=True)

ARCH_FILE = "spark-arch.txt"
_SMI_FIELDS = [
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


def _smi(args: list[str]) -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        return subprocess.check_output([smi, *args], text=True, timeout=10).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


class SparkPlatform:
    name = "spark"
    memory_metric = "gpu_mem"

    # -- identity ------------------------------------------------------------------------
    def supported_backends(self) -> list[str]:
        return ["cuda"]

    def backend(self, settings: Settings) -> str:
        return "cuda"

    def release_tag(self, settings: Settings) -> str | None:
        return None  # source build pinned by commit, see LLAMA_CPP_VERSION

    # -- defaults -------------------------------------------------------------------------
    def default_models_dir(self) -> Path:
        return Path("/opt/models")

    def default_models_toml(self) -> Path:
        return repo_root() / "models.toml"

    def eval_timeout_s(self) -> float:
        return 1800.0

    # -- binaries and process control --------------------------------------------------------
    def binary_path(self, settings: Settings) -> Path:
        return settings.vendor_dir / "build" / "bin" / "llama-server"

    def bench_binary(self, settings: Settings) -> Path:
        return settings.vendor_dir / "build" / "bin" / "llama-bench"

    def runtime_env(self, settings: Settings) -> dict[str, str]:
        env = os.environ.copy()
        bin_dir = str(settings.vendor_dir / "build" / "bin")
        compat = "/usr/local/cuda-13/compat"
        if not Path(compat).is_dir():
            compat = "/usr/local/cuda/compat"
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{compat}:{existing}".rstrip(":")
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        return env

    def popen_kwargs(self) -> dict[str, Any]:
        return {"start_new_session": True}

    def terminate(self, pid: int) -> bool:
        """SIGTERM; False when the process was already gone."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    # -- telemetry --------------------------------------------------------------------------
    def gpu_info(self) -> dict[str, str]:
        """Static + live GPU facts for provenance. Empty dict when nvidia-smi is unavailable."""
        out = _smi([f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"])
        if not out:
            return {}
        values = [v.strip() for v in out.splitlines()[0].split(",")]
        return dict(zip(_SMI_FIELDS, values, strict=False))

    def gpu_memory_used_mib(self) -> int | None:
        out = _smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if not out:
            return None
        try:
            return int(out.splitlines()[0].strip())
        except ValueError:
            return None

    def compute_apps(self) -> list[GpuProcess]:
        out = _smi(
            ["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]
        )
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

    # -- build ------------------------------------------------------------------------------
    def build_arch(self, settings: Settings) -> str | None:
        """CUDA arch recorded by scripts/build.sh; None for builds predating that change."""
        path = settings.vendor_dir / "build" / ARCH_FILE
        return path.read_text().strip() if path.is_file() else None

    def build_script(self) -> Path:
        return repo_root() / "scripts" / "build.sh"

    def build(self, settings: Settings, opts: BuildOptions) -> int:
        if opts.backends and opts.backends != ["cuda"]:
            console.print(f"[red]spark builds CUDA only[/red] (got {', '.join(opts.backends)})")
            return 2
        script = self.build_script()
        if not script.is_file():
            console.print(f"[red]missing[/red] {script}")
            return 1
        return subprocess.call(["bash", str(script)])

    # -- doctor -----------------------------------------------------------------------------
    def doctor_checks(self, settings: Settings) -> list[DoctorCheck]:
        from spark_llm.gpu import split_gpu_processes

        checks: list[DoctorCheck] = []
        machine = os.uname().machine if hasattr(os, "uname") else "unknown"
        checks.append(DoctorCheck("arch", machine == "aarch64", machine))
        nvcc = shutil.which("nvcc")
        checks.append(DoctorCheck("nvcc", bool(nvcc), nvcc or "not found"))
        if nvcc:
            ver = subprocess.check_output([nvcc, "--version"], text=True)
            line = [ln for ln in ver.splitlines() if "release" in ln.lower()]
            checks.append(
                DoctorCheck("cuda toolkit", bool(line), line[-1].strip() if line else ver.strip())
            )

        smi = shutil.which("nvidia-smi")
        checks.append(DoctorCheck("nvidia-smi", bool(smi), smi or "not found"))
        if smi:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            checks.append(DoctorCheck("gpu", "GB10" in out or "12.1" in out, out))
            ours, foreign = split_gpu_processes(settings)
            if foreign:
                checks.append(DoctorCheck("gpu free", False, "; ".join(p.raw for p in foreign)))
            elif ours:
                checks.append(
                    DoctorCheck(
                        "gpu in use by local-llm",
                        True,
                        "; ".join(p.raw for p in ours),
                        note=True,
                    )
                )
            else:
                checks.append(DoctorCheck("gpu free", True, "no compute apps"))

        cmake = shutil.which("cmake")
        checks.append(
            DoctorCheck("cmake", bool(cmake), cmake or "not found (uv tool install 'cmake>=3.30')")
        )
        server = self.binary_path(settings)
        checks.append(DoctorCheck("llama-server", server.is_file(), str(server)))
        if server.is_file():
            try:
                ver = subprocess.check_output(
                    [str(server), "--version"],
                    text=True,
                    stderr=subprocess.STDOUT,
                    env=self.runtime_env(settings),
                )
                checks.append(DoctorCheck("llama-server version", True, ver.splitlines()[0][:120]))
            except Exception as exc:  # noqa: BLE001
                checks.append(DoctorCheck("llama-server version", False, str(exc)))
            arch = self.build_arch(settings)
            checks.append(
                DoctorCheck(
                    "cuda arch",
                    arch is not None and arch.startswith("121a"),
                    arch or "unknown (rebuild with make build to record it)",
                )
            )
        return checks
