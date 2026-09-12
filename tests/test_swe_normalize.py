"""External harness outputs -> results.jsonl schema."""

from __future__ import annotations

import json
from pathlib import Path

from spark_llm.evals.swe.normalize import (
    normalize_aider,
    normalize_evalplus,
    normalize_swebench,
    parse_evalplus_stdout,
)

EVALPLUS_STDOUT = """
Computing expected output...
humaneval (base tests)
pass@1: 0.854
humaneval (base + extra tests)
pass@1: 0.799
"""


def test_evalplus(tmp_path: Path) -> None:
    assert parse_evalplus_stdout(EVALPLUS_STDOUT) == {
        "humaneval_base": 0.854,
        "humaneval_plus": 0.799,
    }
    d = tmp_path / "humaneval" / "m_openai_temp_0.0"
    d.mkdir(parents=True)
    (d / "eval_results.json").write_text(
        json.dumps(
            {
                "eval": {
                    "HumanEval/0": [{"base_status": "pass", "plus_status": "pass"}],
                    "HumanEval/1": [{"base_status": "pass", "plus_status": "fail"}],
                    "HumanEval/2": [{"base_status": "fail", "plus_status": "fail"}],
                }
            }
        )
    )
    results, summary = normalize_evalplus(tmp_path, "humaneval", EVALPLUS_STDOUT)
    assert [r["correct"] for r in results] == [True, False, False]
    assert summary["n"] == 3 and abs(summary["score"] - 1 / 3) < 1e-9
    assert summary["plus_pass_at_1"] == 0.799


def test_aider(tmp_path: Path) -> None:
    for lang, ex, outcomes, malformed in (
        ("python", "anagram", [False, True], 1),
        ("rust", "bowling", [True], 0),
        ("go", "clock", [False, False], 2),
    ):
        d = tmp_path / lang / "exercises" / "practice" / ex
        d.mkdir(parents=True)
        (d / ".aider.results.json").write_text(
            json.dumps(
                {
                    "tests_outcomes": outcomes,
                    "num_malformed_responses": malformed,
                    "syntax_errors": 0,
                    "duration": 12.5,
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                }
            )
        )
    results, summary = normalize_aider(tmp_path)
    assert summary["n"] == 3
    assert abs(summary["score"] - 2 / 3) < 1e-9
    assert abs(summary["pass_rate_first_try"] - 1 / 3) < 1e-9
    assert abs(summary["format_failure_rate"] - 2 / 3) < 1e-9
    assert summary["by_language"]["python"] == {"n": 1, "correct": 1}
    assert {r["id"] for r in results} == {"python/anagram", "rust/bowling", "go/clock"}


def test_swebench() -> None:
    ids = ["a__a-1", "b__b-2", "c__c-3"]
    preds = {"a__a-1": {"model_patch": "diff"}, "b__b-2": {"model_patch": "diff"}}
    report = {"resolved_ids": ["a__a-1"], "error_ids": []}
    trajs = {
        "a__a-1": {
            "info": {
                "exit_status": "Submitted",
                "model_stats": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            "messages": [{}] * 8,
        },
        "c__c-3": {"info": {"exit_status": "FormatError"}, "messages": [{}] * 4},
    }
    results, summary = normalize_swebench(preds, report, trajs, ids)
    assert [r["correct"] for r in results] == [True, False, False]
    assert results[2]["format_failure"] and not results[2]["submitted"]
    assert results[0]["steps"] == 3
    assert abs(summary["score"] - 1 / 3) < 1e-9
    assert abs(summary["format_failure_rate"] - 1 / 3) < 1e-9
    assert summary["exit_statuses"] == {"Submitted": 2, "FormatError": 1}
    # Without a harness report nothing is scored, but nothing is silently zero either
    _, unscored = normalize_swebench(preds, None, trajs, ids)
    assert unscored["score"] is None and unscored["skipped"] == 3
