"""Per-platform logic: everything that differs between the DGX Spark (Linux/CUDA) and the
AMD Strix Halo box (Windows, Vulkan or HIP) lives under ``platforms/<name>/``.

Shared modules (server, gpu, provenance, config, cli) call :func:`current` and delegate; they
never branch on ``sys.platform`` themselves. Detection is by OS with an env override so tests
and odd hosts can force a platform. This module imports only ``os``/``sys`` at import time so
``config.py`` can use it for defaults without a cycle.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spark_llm.platforms.base import Platform

PLATFORMS = ("spark", "halo")
ENV_VARS = ("LOCAL_LLM_PLATFORM", "SPARK_LLM_PLATFORM")

_forced: str | None = None
_cache: dict[str, Platform] = {}


def detect() -> str:
    """``halo`` on Windows, ``spark`` everywhere else; env or :func:`force` overrides."""
    if _forced:
        return _forced
    for var in ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value.strip().lower()
    return "halo" if sys.platform == "win32" else "spark"


def current(name: str | None = None) -> Platform:
    name = (name or detect()).lower()
    if name not in _cache:
        if name == "spark":
            from spark_llm.platforms.spark.platform import SparkPlatform

            _cache[name] = SparkPlatform()
        elif name == "halo":
            from spark_llm.platforms.halo.platform import HaloPlatform

            _cache[name] = HaloPlatform()
        else:
            raise ValueError(f"unknown platform {name!r}; choose from {', '.join(PLATFORMS)}")
    return _cache[name]


def force(name: str | None) -> None:
    """Pin the detected platform (tests, or scripts driving a remote box). None resets."""
    global _forced
    _forced = name.lower() if name else None
