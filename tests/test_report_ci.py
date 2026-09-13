"""Bootstrap intervals, unreliable-throughput masking, and the no-document control label."""

from __future__ import annotations

from spark_llm.evals.config import load_prompt
from spark_llm.evals.report import bootstrap_ci, ci_text
from spark_llm.evals.sec.run import SecRunOptions, task_label


def test_bootstrap_ci_shape_and_determinism() -> None:
    assert bootstrap_ci([True] * 3) is None  # too few items
    assert bootstrap_ci([True] * 50) == (1.0, 1.0)
    flags = [True] * 117 + [False] * 4  # the Halo Vulkan result
    lo, hi = bootstrap_ci(flags)
    assert 0.90 < lo < 0.97 < hi <= 1.0
    assert bootstrap_ci(flags) == (lo, hi)  # seeded
    lo2, hi2 = bootstrap_ci([True] * 119 + [False] * 2)  # the Spark result
    assert lo2 < 0.983 < hi2 and lo < 0.983  # the two intervals overlap
    assert ci_text(flags) == f"{lo:.2f}–{hi:.2f}" and ci_text([True]) is None


def test_no_document_control_has_its_own_task_label_and_prompts() -> None:
    assert task_label(SecRunOptions(task="extract-full")) == "extract-full"
    assert task_label(SecRunOptions(task="extract-full", no_document=True)) == "extract-full-nodoc"
    user = load_prompt("sec_extract_nodoc_user").format(
        form="10-K", company="Apple", period_end="2025-09-27", question="Q?", format_hint="H"
    )
    assert "No document is attached" in user and "{document}" not in user
    assert "from memory" in load_prompt("sec_extract_nodoc_system")
