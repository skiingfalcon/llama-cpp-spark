"""Coverage for small pure units that had none: gpu helpers, server state, provenance, CLI smoke,
platform Protocol conformance."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spark_llm import gpu, platforms, provenance
from spark_llm.cli import app
from spark_llm.config import Settings
from spark_llm.platforms.base import GpuProcess, Platform
from spark_llm.platforms.halo.platform import HaloPlatform
from spark_llm.platforms.spark.platform import SparkPlatform
from spark_llm.server import load_state, save_state


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        state_dir=tmp_path / "state", vendor_dir=tmp_path / "vendor", models_dir=tmp_path / "m"
    )


class _FakePlatform:
    name = "fake"
    memory_metric = "gpu_mem"

    def __init__(self, alive: set[int], apps: list[GpuProcess]) -> None:
        self.alive = alive
        self.apps = apps

    def pid_alive(self, pid: int) -> bool:
        return pid in self.alive

    def compute_apps(self) -> list[GpuProcess]:
        return self.apps

    def gpu_info(self) -> dict[str, str]:
        return {"name": "FakeGPU"}

    def gpu_memory_used_mib(self) -> int | None:
        return 42

    def backend(self, settings: Settings) -> str:
        return "cuda"

    def release_tag(self, settings: Settings) -> str | None:
        return "b1"

    def build_arch(self, settings: Settings) -> str | None:
        return "fake-arch"


def test_platform_protocol_conformance() -> None:
    required = set(Platform.__protocol_attrs__)  # type: ignore[attr-defined]
    for cls in (SparkPlatform, HaloPlatform):
        missing = {a for a in required if not hasattr(cls, a)}
        assert not missing, f"{cls.__name__} lacks {sorted(missing)}"


def test_gpu_busy_reasons_split_ours_from_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    save_state(settings, {"srv": {"pid": 111, "port": 8080}, "dead": {"pid": 222, "port": 8081}})
    fake = _FakePlatform(
        alive={111},
        apps=[
            GpuProcess(pid=111, name="llama-server", used_mib="900", raw="111, llama-server, 900"),
            GpuProcess(pid=999, name="python", used_mib="100", raw="999, python, 100"),
        ],
    )
    monkeypatch.setattr(gpu, "current", lambda: fake)
    assert gpu.live_tracked_servers(settings) == {"srv": 111}  # dead pid dropped
    ours, foreign = gpu.split_gpu_processes(settings)
    assert [p.pid for p in ours] == [111] and [p.pid for p in foreign] == [999]
    reasons = gpu.busy_reasons(settings)
    assert any("tracked server srv" in r and "local-llm stop" in r for r in reasons)
    assert any("foreign GPU process: 999" in r for r in reasons)
    assert gpu.gpu_memory_used_mib() == 42 and gpu.gpu_info() == {"name": "FakeGPU"}


def test_server_state_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert load_state(settings) == {}
    save_state(settings, {"a": {"pid": 1, "port": 2, "argv": ["x"], "log": "l"}})
    assert load_state(settings)["a"]["port"] == 2
    assert json.loads((settings.state_dir / "servers.json").read_text())["a"]["pid"] == 1


def test_provenance_checkout_follows_bin_dir_override(tmp_path: Path) -> None:
    fork = tmp_path / "fork"
    (fork / "build" / "bin").mkdir(parents=True)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ["PATH"],
    }
    subprocess.run(["git", "init", "-q", str(fork)], check=True, env=env)
    (fork / "f").write_text("x")
    subprocess.run(["git", "-C", str(fork), "add", "f"], check=True, env=env)
    subprocess.run(["git", "-C", str(fork), "commit", "-q", "-m", "m"], check=True, env=env)
    head = subprocess.check_output(
        ["git", "-C", str(fork), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()

    def settings(bin_dir: Path) -> Settings:
        return Settings(
            state_dir=tmp_path / "state",
            vendor_dir=tmp_path / "vendor",
            models_dir=tmp_path / "m",
            llama_bin_dir=bin_dir,
        )

    # no vendor checkout, but the override points inside a clone: report that clone's HEAD
    assert provenance.checkout_commit(settings(fork / "build" / "bin")) == head
    # an override that is just a directory (not a clone) reports nothing rather than this repo
    bare = tmp_path / "bare-bin"
    bare.mkdir()
    assert provenance.checkout_commit(settings(bare)) is None


def test_provenance_collect_uses_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePlatform(alive=set(), apps=[])
    monkeypatch.setattr(provenance, "current", lambda: fake)
    monkeypatch.setattr(provenance, "gpu_info", fake.gpu_info)
    settings = _settings(tmp_path)
    prov = provenance.collect(settings, with_gpu=True)
    assert prov.platform == "fake" and prov.backend == "cuda" and prov.llama_cpp_release == "b1"
    assert prov.build_id == "fake-arch" and prov.gpu == {"name": "FakeGPU"}
    assert prov.tool_version and prov.llama_cpp_checkout is None  # no vendor checkout in tmp
    # legacy field names still load and re-serialise under the new names
    legacy = provenance.Provenance.model_validate(
        {
            "timestamp": "t",
            "hostname": "h",
            "machine": "m",
            "spark_llm_version": "0.0",
            "cuda_arch": "121a",
        }
    )
    dumped = legacy.model_dump()
    assert dumped["tool_version"] == "0.0" and dumped["build_id"] == "121a"
    assert "spark_llm_version" not in dumped and "cuda_arch" not in dumped


@pytest.fixture(autouse=True)
def _reset_platform():
    platforms.force(None)
    yield
    platforms.force(None)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Rich styles and wraps help text depending on the terminal; compare plain, unwrapped text."""
    return re.sub(r"\s+", " ", _ANSI.sub("", text))


def test_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("LOCAL_LLM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOCAL_LLM_VENDOR_DIR", str(tmp_path / "vendor"))
    monkeypatch.setenv("LOCAL_LLM_MODELS_DIR", str(tmp_path / "models"))
    res = runner.invoke(app, ["build", "--help"])
    assert res.exit_code == 0 and "--backend" in _plain(res.output)
    res = runner.invoke(app, ["models"])
    assert res.exit_code == 0 and "gpt-oss-120b" in _plain(res.output)
    for plat in ("spark", "halo"):
        monkeypatch.setenv("LOCAL_LLM_PLATFORM", plat)
        res = runner.invoke(app, ["doctor"])
        assert res.exit_code in (0, 1), res.output  # failing checks are fine; tracebacks are not
        assert f"platform {plat}" in _plain(res.output) and "Traceback" not in res.output
    monkeypatch.setenv("LOCAL_LLM_PLATFORM", "halo")
    res = runner.invoke(app, ["serve", "gpt-oss-120b", "--backend", "cuda"])
    assert res.exit_code == 2 and "not available on halo" in _plain(res.output)
