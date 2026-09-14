#!/usr/bin/env python3
"""Regenerate the deck's charts from the numbers in docs/eval-report-2026-09-spark-halo-terra.md.

Run: uv run --with matplotlib python docs/deck/make_charts.py

Chart rules (see the dataviz method): magnitude across nominal stacks -> horizontal bars, one
series, one hue, no legend; two measures of different scale -> two panels, one axis each; thin
bars; direct value labels in text ink, never in the series colour; hairline gridlines; light
surface only (projected deck).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "charts"

# Palette (reference instance, light mode): series slot 1 + text tokens + surface.
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e9e8e4"  # one step off the surface

# --- numbers: docs/eval-report-2026-09-spark-halo-terra.md + state/evals/report-sec.md ----------
STACKS = ["OpenAI gpt-5.6-terra", "DGX Spark · CUDA", "Lunch box · Vulkan", "Lunch box · ROCm"]
ACCURACY = [99.2, 98.3, 96.7, 95.0]  # % of 121 questions
CI = [(98.0, 100.0), (96.0, 100.0), (93.0, 99.0), (91.0, 98.0)]  # bootstrap 95%, report-sec.md
LOCAL = ["DGX Spark · CUDA", "Lunch box · Vulkan", "Lunch box · ROCm"]
COLD_PREFILL_S = [63, 300, 238]  # per-request p95 = cold prefill of a ~100K-token filing
DECODE_TPS = [30, 29, 19]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.edgecolor": GRID,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)

BAR_H = 0.30  # fraction of the category band; the rest is air


def _style(ax, xmax: float, xlabel: str, ticks: list[float]) -> None:
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.tick_params(axis="x", length=0, labelsize=11)
    ax.set_xlabel(xlabel, color=TEXT_2, labelpad=8, fontsize=12)


def _bars(ax, labels: list[str], values: list[float], xmax: float, fmt) -> None:
    """Thin single-hue bars with the value at the tip in text ink."""
    ax.barh(range(len(labels)), values, height=BAR_H, color=SERIES, edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.012, i, fmt(v), va="center", ha="left", color=TEXT, fontsize=13)


def accuracy_chart() -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.9), dpi=200, layout="constrained")
    _bars(ax, STACKS, ACCURACY, 112, lambda v: f"{v:.1f}%")
    for i, (lo, hi) in enumerate(CI):
        y = i + BAR_H / 2 + 0.14
        ax.plot([lo, hi], [y, y], color=TEXT_2, linewidth=1, solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.05, y + 0.05], color=TEXT_2, linewidth=1)
    _style(ax, 112, "Accuracy on 121 questions (%) · thin line = bootstrap 95% interval", [0, 25, 50, 75, 100])
    ax.set_ylim(len(STACKS) - 0.45, -0.55)
    out = OUT / "accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def prefill_decode_chart() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0), dpi=200, layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)
    _bars(ax1, LOCAL, COLD_PREFILL_S, 345, lambda v: f"{v:.0f} s" if v < 120 else f"{v / 60:.1f} min")
    _style(ax1, 345, "Cold prefill of a ~100K-token filing (seconds, p95 per request)", [0, 60, 120, 180, 240, 300])
    ax1.set_title("Reading the document (prefill)", loc="left", color=TEXT, fontsize=14, pad=12)
    _bars(ax2, LOCAL, DECODE_TPS, 37, lambda v: f"{v:.0f} t/s")
    _style(ax2, 37, "Decode speed (tokens per second, median)", [0, 10, 20, 30])
    ax2.set_title("Writing the answer (decode)", loc="left", color=TEXT, fontsize=14, pad=12)
    for ax in (ax1, ax2):
        ax.set_ylim(len(LOCAL) - 0.45, -0.55)
    out = OUT / "prefill-decode.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (accuracy_chart(), prefill_decode_chart()):
        print(path)
