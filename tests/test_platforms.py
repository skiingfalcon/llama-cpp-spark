"""Platform detection and the Spark / Halo implementations (no GPU, no Windows required)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spark_llm import platforms
from spark_llm.config import Settings
from spark_llm.platforms.halo.platform import (
    BUILD_FILE,
    HaloPlatform,
    parse_devices_output,
    parse_version_output,
)
from spark_llm.platforms.spark.platform import SparkPlatform


@pytest.fixture(autouse=True)
def _reset_platform():
    platforms.force(None)
    yield
    platforms.force(None)


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        models_dir=tmp_path / "models",
        vendor_dir=tmp_path / "vendor",
        state_dir=tmp_path / "state",
        **kw,
    )


# -- detection -----------------------------------------------------------------------------------


def test_detect_by_os(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_LLM_PLATFORM", raising=False)
    monkeypatch.delenv("SPARK_LLM_PLATFORM", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert platforms.detect() == "spark"
    monkeypatch.setattr(sys, "platform", "win32")
    assert platforms.detect() == "halo"


def test_detect_env_override_and_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LOCAL_LLM_PLATFORM", "halo")
    assert platforms.detect() == "halo"
    monkeypatch.delenv("LOCAL_LLM_PLATFORM")
    monkeypatch.setenv("SPARK_LLM_PLATFORM", "halo")  # legacy prefix still honoured
    assert platforms.detect() == "halo"
    platforms.force("spark")
    assert platforms.detect() == "spark" and platforms.current().name == "spark"


def test_unknown_platform_rejected() -> None:
    with pytest.raises(ValueError, match="unknown platform"):
        platforms.current("amiga")


# -- spark: behaviour moved verbatim ------------------------------------------------------------


def test_spark_paths_env_and_popen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plat = SparkPlatform()
    settings = _settings(tmp_path)
    assert plat.binary_path(settings) == tmp_path / "vendor" / "build" / "bin" / "llama-server"
    assert plat.bench_binary(settings) == tmp_path / "vendor" / "build" / "bin" / "llama-bench"
    monkeypatch.setenv("LD_LIBRARY_PATH", "/x")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = plat.runtime_env(settings)
    bin_dir = str(tmp_path / "vendor" / "build" / "bin")
    compat = (
        "/usr/local/cuda-13/compat"
        if Path("/usr/local/cuda-13/compat").is_dir()
        else "/usr/local/cuda/compat"
    )
    assert env["LD_LIBRARY_PATH"] == f"{bin_dir}:{compat}:/x"
    assert env["PATH"] == f"{bin_dir}:/usr/bin"
    assert plat.popen_kwargs() == {"start_new_session": True}
    assert plat.backend(settings) == "cuda" and plat.supported_backends() == ["cuda"]
    assert plat.default_models_dir() == Path("/opt/models")
    assert plat.default_models_toml().name == "models.toml"
    assert plat.eval_timeout_s() == 1800.0
    assert plat.memory_metric == "gpu_mem"


def test_spark_terminate_and_pid_alive_on_dead_pid() -> None:
    plat = SparkPlatform()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert plat.terminate(proc.pid) is False  # already gone -> "already dead" path
    assert plat.pid_alive(proc.pid) is False
    assert plat.pid_alive(os.getpid()) is True


def test_spark_build_arch_reads_arch_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert SparkPlatform().build_arch(settings) is None
    (tmp_path / "vendor" / "build").mkdir(parents=True)
    (tmp_path / "vendor" / "build" / "spark-arch.txt").write_text("121a-real\n")
    assert SparkPlatform().build_arch(settings) == "121a-real"


# -- halo ------------------------------------------------------------------------------------------


def _halo_build(tmp_path: Path, backends: dict[str, dict]) -> None:
    (tmp_path / "vendor").mkdir(parents=True, exist_ok=True)
    (tmp_path / "vendor" / BUILD_FILE).write_text(json.dumps({"backends": backends}))


def test_halo_binary_resolution_order(tmp_path: Path) -> None:
    plat = HaloPlatform()
    settings = _settings(tmp_path)
    # 1. nothing installed -> vendor/bin fallback
    assert plat.binary_path(settings) == tmp_path / "vendor" / "bin" / "llama-server.exe"
    # 2. halo-build.json entry for the selected backend
    _halo_build(
        tmp_path,
        {
            "vulkan": {"tag": "b10919", "source": "official", "bin_dir": str(tmp_path / "vk")},
            "hip": {"tag": "b10919", "source": "official", "bin_dir": str(tmp_path / "hip")},
        },
    )
    assert plat.binary_path(settings) == tmp_path / "vk" / "llama-server.exe"
    assert (
        plat.bench_binary(_settings(tmp_path, backend="hip"))
        == tmp_path / "hip" / "llama-bench.exe"
    )
    assert plat.build_arch(_settings(tmp_path, backend="hip")) == "hip:b10919:official"
    assert plat.release_tag(settings) == "b10919"
    # 3. explicit bin dir wins over everything
    assert (
        plat.binary_path(_settings(tmp_path, llama_bin_dir=tmp_path / "custom"))
        == tmp_path / "custom" / "llama-server.exe"
    )


def test_halo_backend_validation(tmp_path: Path) -> None:
    plat = HaloPlatform()
    assert plat.backend(_settings(tmp_path)) == "vulkan"
    assert plat.backend(_settings(tmp_path, backend="HIP")) == "hip"
    with pytest.raises(ValueError, match="not supported on halo"):
        plat.backend(_settings(tmp_path, backend="cuda"))
    assert plat.supported_backends() == ["vulkan", "hip"]


def test_halo_runtime_env_and_popen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plat = HaloPlatform()
    _halo_build(
        tmp_path, {"vulkan": {"tag": "t", "source": "official", "bin_dir": str(tmp_path / "vk")}}
    )
    monkeypatch.setenv("PATH", "C:\\Windows")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/should/not/matter")
    env = plat.runtime_env(_settings(tmp_path))
    assert env["PATH"].startswith(str(tmp_path / "vk") + os.pathsep)
    assert env["PATH"].endswith("C:\\Windows")
    assert env.get("LD_LIBRARY_PATH") == "/should/not/matter"  # untouched, not our concern
    assert plat.popen_kwargs() == {"creationflags": 0x208}
    assert plat.eval_timeout_s() == 3600.0 and plat.memory_metric == "unified_mem"
    assert plat.default_models_toml().name == "models.halo.toml"


def test_halo_default_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    assert HaloPlatform().default_models_dir() == tmp_path / "AppData" / "local-llm" / "models"
    monkeypatch.delenv("LOCALAPPDATA")
    assert HaloPlatform().default_models_dir() == Path.home() / "local-llm" / "models"


class _Runner:
    """Records commands and replays canned CompletedProcess results."""

    def __init__(self, responses: dict[str, tuple[int, str]]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(list(argv))
        for key, (rc, out) in self.responses.items():
            if key in " ".join(argv):
                return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")


def test_halo_terminate_and_tasklist_liveness() -> None:
    runner = _Runner({"taskkill": (0, "SUCCESS"), "PID eq 42": (0, '"llama-server.exe","42"')})
    plat = HaloPlatform(runner=runner)
    assert plat.terminate(42) is True
    assert runner.calls[0][:3] == ["taskkill", "/PID", "42"] and "/T" in runner.calls[0]
    assert plat._pid_alive_tasklist(42) is True
    assert plat._pid_alive_tasklist(43) is False
    dead = HaloPlatform(runner=_Runner({"taskkill": (128, "not found")}))
    assert dead.terminate(7) is False


def test_halo_cim_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = json.dumps(
        [
            {"Name": "Microsoft Basic Display", "DriverVersion": "1.0", "AdapterRAM": 0},
            {
                "Name": "AMD Radeon(TM) 8060S Graphics",
                "DriverVersion": "32.0.13031",
                "DriverDate": "2026-08-01",
            },
        ]
    )
    mem = json.dumps(
        {"TotalVisibleMemorySize": 128 * 1024 * 1024, "FreePhysicalMemory": 96 * 1024 * 1024}
    )
    runner = _Runner({"Win32_VideoController": (0, gpu), "Win32_OperatingSystem": (0, mem)})
    monkeypatch.setattr(
        "shutil.which", lambda name: "pwsh" if name in {"pwsh", "powershell"} else None
    )
    plat = HaloPlatform(runner=runner)
    info = plat.gpu_info()
    assert info["name"].startswith("AMD Radeon") and info["driver_version"] == "32.0.13031"
    assert info["memory.total"] == str(128 * 1024)
    assert plat.gpu_memory_used_mib() == 32 * 1024
    assert plat.compute_apps() == []


def test_halo_telemetry_degrades_without_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    plat = HaloPlatform()
    assert plat.gpu_info() == {} and plat.gpu_memory_used_mib() is None


def test_halo_doctor_reports_missing_install_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    checks = HaloPlatform().doctor_checks(_settings(tmp_path))
    labels = {c.label: c for c in checks}
    assert labels["backends installed"].ok is False
    assert "local-llm build" in labels["backends installed"].detail
    assert labels["variable graphics memory"].note is True


def test_halo_doctor_flags_invalid_backend_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    checks = HaloPlatform().doctor_checks(_settings(tmp_path, backend="cuda"))
    sel = next(c for c in checks if c.label == "selected backend")
    assert sel.ok is False and "not supported on halo" in sel.detail


def test_parse_helpers() -> None:
    ver, commit = parse_version_output("version: 10919 (d3146f2b5)\nbuilt with MSVC for x86_64\n")
    assert ver == "version: 10919 (d3146f2b5)" and commit == "d3146f2b5"
    devs = parse_devices_output(
        "Available devices:\n  Vulkan0: AMD Radeon(TM) 8060S Graphics (98304 MiB, 98304 MiB free)\n"
    )
    assert devs == ["Vulkan0: AMD Radeon(TM) 8060S Graphics (98304 MiB, 98304 MiB free)"]
    assert parse_devices_output("") == []


# -- settings --------------------------------------------------------------------------------------


def test_settings_defaults_follow_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("LOCAL_LLM_MODELS_DIR", "SPARK_LLM_MODELS_DIR", "LOCAL_LLM_MODELS_TOML"):
        monkeypatch.delenv(var, raising=False)
    platforms.force("spark")
    assert Settings().models_dir == Path("/opt/models")
    assert Settings().models_toml.name == "models.toml" and Settings().eval_timeout_s == 1800.0
    platforms.force("halo")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert Settings().models_dir == tmp_path / "local-llm" / "models"
    assert Settings().models_toml.name == "models.halo.toml" and Settings().eval_timeout_s == 3600.0


def test_settings_read_new_and_legacy_prefixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no repo .env in play
    monkeypatch.delenv("LOCAL_LLM_MODELS_DIR", raising=False)
    monkeypatch.setenv("SPARK_LLM_MODELS_DIR", "/legacy")
    assert Settings().models_dir == Path("/legacy")
    monkeypatch.setenv("LOCAL_LLM_MODELS_DIR", "/new")
    assert Settings().models_dir == Path("/new")  # new prefix wins when both are set
    monkeypatch.setenv("LOCAL_LLM_BACKEND", "hip")
    assert Settings().backend == "hip"
