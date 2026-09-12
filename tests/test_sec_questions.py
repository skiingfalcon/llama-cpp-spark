"""XBRL fact selection and numeric scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spark_llm.evals.config import load_eval_config
from spark_llm.evals.sec.questions import (
    answer_has_single_number,
    parse_number,
    questions_for_filing,
    score_numeric,
)

FIXTURE = Path(__file__).parent / "fixtures" / "companyfacts_aapl_trimmed.json"


@pytest.fixture(scope="module")
def facts() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def tags():
    return load_eval_config().sec.xbrl_tags


def _by_tag(qs):
    return {q.tag: q for q in qs}


def test_10q_questions_pick_quarter_and_instant_values(facts, tags) -> None:
    qs = _by_tag(
        questions_for_filing(
            facts,
            tags,
            form="10-Q",
            accession="0000320193-26-000020",
            report_date="2026-06-27",
            key="AAPL:10-Q:2026-06-27",
        )
    )
    assert qs["Revenues"].expected == 109_417_000_000  # three months, via alias concept
    assert qs["Revenues"].source["concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert "3-month period ended June 27, 2026" in qs["Revenues"].text
    assert qs["NetIncomeLoss"].expected == 29_789_000_000
    assert qs["EarningsPerShareDiluted"].expected == 2.02
    assert qs["NetCashProvidedByUsedInOperatingActivities"].expected == 116_996_000_000  # YTD
    assert "9-month" in qs["NetCashProvidedByUsedInOperatingActivities"].text
    assert qs["Assets"].expected == 383_266_000_000
    assert "as of June 27, 2026" in qs["Assets"].text
    assert qs["CommonStockSharesOutstanding"].unit == "shares"


def test_10k_questions_pick_fiscal_year_values(facts, tags) -> None:
    qs = _by_tag(
        questions_for_filing(
            facts,
            tags,
            form="10-K",
            accession="0000320193-25-000079",
            report_date="2025-09-27",
            key="AAPL:10-K:2025-09-27",
        )
    )
    # The fixture's FY 10-K tags both a total-revenues line (canonical) and the contract-revenue
    # line; the canonical concept sets ``expected`` and the other becomes an accepted alternate.
    assert qs["Revenues"].expected == 420_000_000_000
    assert qs["Revenues"].source["concept"] == "Revenues"
    assert qs["Revenues"].alternates == [416_161_000_000]
    assert "fiscal year ended September 27, 2025" in qs["Revenues"].text
    assert qs["NetIncomeLoss"].expected == 112_010_000_000
    assert qs["NetIncomeLoss"].alternates == []  # strict tag: no accept_aliases
    assert qs["StockholdersEquity"].expected == 73_733_000_000
    assert len(qs) == 11


def test_long_term_debt_prefers_noncurrent_and_ignores_total_debt(facts, tags) -> None:
    qs = _by_tag(
        questions_for_filing(
            facts,
            tags,
            form="10-K",
            accession="0000320193-25-000079",
            report_date="2025-09-27",
            key="AAPL:10-K:2025-09-27",
        )
    )
    q = qs["LongTermDebtNoncurrent"]
    assert q.source["concept"] == "LongTermDebtNoncurrent" and q.expected == 78_328_000_000
    # Lease-inclusive noncurrent line is an acceptable alternate; LongTermDebt (total incl.
    # current portion) is not part of the family and must never be accepted.
    assert q.alternates == [79_000_000_000]
    assert 90_000_000_000 not in q.alternates


def test_alias_declared_order_breaks_ties(facts) -> None:
    from spark_llm.evals.config import XbrlTag

    tag = XbrlTag(
        tag="Missing",
        label="x",
        kind="instant",
        aliases=["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    )
    q = questions_for_filing(
        facts,
        [tag],
        form="10-K",
        accession="0000320193-25-000079",
        report_date="2025-09-27",
        key="k",
    )[0]
    assert q.source["concept"] == "LongTermDebt"  # first declared alias wins
    tag.aliases.reverse()
    q = questions_for_filing(
        facts,
        [tag],
        form="10-K",
        accession="0000320193-25-000079",
        report_date="2025-09-27",
        key="k",
    )[0]
    assert q.source["concept"] == "LongTermDebtAndCapitalLeaseObligations"


def test_missing_period_yields_no_question(facts, tags) -> None:
    qs = questions_for_filing(
        facts, tags, form="10-Q", accession="x", report_date="1999-01-01", key="k"
    )
    assert qs == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("109417000000", 109_417_000_000),
        ("$109,417 million", 109_417_000_000),
        ("$109.4 billion", 109_400_000_000),
        ("(1,234)", -1234),
        ("-1,234 thousand", -1_234_000),
        ("2.02", 2.02),
        ("The value is 14,608,963,000 shares.", 14_608_963_000),
        ("unknown", None),
        ("Unknown.", None),
        ("**$39,544 million**", 39_544_000_000),
        ("−5,000", -5000),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_score_numeric_tolerance_and_scale() -> None:
    assert score_numeric("109,417,000,000", 109_417_000_000, 0.005).correct
    assert score_numeric("$109.4 billion", 109_417_000_000, 0.005).correct
    s = score_numeric("109,417", 109_417_000_000, 0.005)  # forgot "in millions"
    assert not s.correct and s.off_by_scale
    s = score_numeric("95,000,000,000", 109_417_000_000, 0.005)
    assert not s.correct and not s.off_by_scale
    assert not score_numeric("unknown", 1.0, 0.005).correct
    assert score_numeric("-1,234", -1234, 0.005).correct
    assert score_numeric("0", 0, 0.005).correct


def test_score_numeric_accepts_alternates_but_reports_error_vs_expected() -> None:
    s = score_numeric("706,413,000,000", 713_163_000_000, 0.005, alternates=[706_413_000_000])
    assert s.correct and s.matched == 706_413_000_000
    assert s.rel_error is not None and s.rel_error > 0.005  # still measured against expected
    s = score_numeric("713,163,000,000", 713_163_000_000, 0.005, alternates=[706_413_000_000])
    assert s.correct and s.matched == 713_163_000_000
    s = score_numeric("700,000,000,000", 713_163_000_000, 0.005, alternates=[706_413_000_000])
    assert not s.correct and s.matched is None


def test_answer_has_single_number() -> None:
    assert answer_has_single_number("$8.7 billion")
    assert not answer_has_single_number("Between 2019 and 2020 it rose 5%")
    assert not answer_has_single_number("Yes, it did.")
