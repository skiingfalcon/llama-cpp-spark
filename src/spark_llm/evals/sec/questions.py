"""Ground-truth questions from XBRL company facts, plus the numeric answer scorer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from spark_llm.evals.config import XbrlTag


@dataclass
class Question:
    id: str
    tag: str
    label: str
    kind: str
    unit: str
    period_start: str | None
    period_end: str
    expected: float
    text: str
    format_hint: str
    source: dict[str, Any] = field(default_factory=dict)


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _entries(facts: dict[str, Any], tag: XbrlTag) -> list[dict[str, Any]]:
    """Fact entries for a tag or its aliases in the requested unit, tagged with concept."""
    out: list[dict[str, Any]] = []
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    for concept in [tag.tag, *tag.aliases]:
        node = gaap.get(concept) or dei.get(concept)
        if not node:
            continue
        for unit, entries in node.get("units", {}).items():
            if unit != tag.unit:
                continue
            for e in entries:
                out.append({**e, "concept": concept, "unit": unit})
    return out


def _duration_ok(kind: str, form: str, e: dict[str, Any]) -> bool:
    if kind == "instant":
        return "start" not in e
    if "start" not in e:
        return False
    d = _days(e["start"], e["end"])
    if kind == "duration":
        return 340 <= d <= 380 if form == "10-K" else 80 <= d <= 100
    if kind == "ytd":
        return 340 <= d <= 380 if form == "10-K" else 80 <= d <= 290
    return False


def select_fact(
    facts: dict[str, Any], tag: XbrlTag, form: str, accession: str, report_date: str
) -> dict[str, Any] | None:
    """Pick the single fact a reader of this filing should find for ``tag``.

    Preference: reported in this very filing (accn) > same form, same period end.
    Within ties: entries carrying a calendar ``frame`` (deduplicated by EDGAR) > latest filed.
    For ``ytd`` on a 10-Q the longest qualifying duration wins.
    """
    cands = [
        e
        for e in _entries(facts, tag)
        if e.get("end") == report_date and _duration_ok(tag.kind, form, e)
    ]
    if not cands:
        return None
    own = [e for e in cands if e.get("accn") == accession]
    pool = own or [e for e in cands if e.get("form") == form] or cands

    def rank(e: dict[str, Any]) -> tuple:
        dur = _days(e["start"], e["end"]) if "start" in e else 0
        return (
            e["concept"] == tag.tag,  # canonical concept over aliases
            dur if tag.kind == "ytd" else 0,
            "frame" in e,
            e.get("filed", ""),
        )

    return max(pool, key=rank)


def _fmt_date(d: str) -> str:
    return date.fromisoformat(d).strftime("%B %d, %Y").replace(" 0", " ")


def _period_phrase(kind: str, form: str, e: dict[str, Any]) -> str:
    end = _fmt_date(e["end"])
    if kind == "instant":
        return f"as of {end}"
    if form == "10-K":
        return f"for the fiscal year ended {end}"
    months = round(_days(e["start"], e["end"]) / 30.4)
    return f"for the {months}-month period ended {end}"


def _hint(unit: str) -> str:
    if unit == "USD/shares":
        return "Give dollars per share as a plain decimal, e.g. 1.23 (negative as -1.23)."
    if unit == "shares":
        return "Give the full number of shares, e.g. 1234567890 (not in thousands or millions)."
    return (
        "Give the full amount in US dollars, e.g. 1234567890 for $1.23 billion "
        "(not in thousands or millions; negative as -1234567890)."
    )


def questions_for_filing(
    facts: dict[str, Any],
    tags: list[XbrlTag],
    *,
    form: str,
    accession: str,
    report_date: str,
    key: str,
) -> list[Question]:
    out: list[Question] = []
    for tag in tags:
        e = select_fact(facts, tag, form, accession, report_date)
        if e is None:
            continue
        phrase = _period_phrase(tag.kind, form, e)
        out.append(
            Question(
                id=f"{key}:{tag.tag}",
                tag=tag.tag,
                label=tag.label,
                kind=tag.kind,
                unit=tag.unit,
                period_start=e.get("start"),
                period_end=e["end"],
                expected=float(e["val"]),
                text=f"What was {tag.label} {phrase}?",
                format_hint=_hint(tag.unit),
                source={k: e.get(k) for k in ("concept", "accn", "form", "fy", "fp", "frame")},
            )
        )
    return out


# -- scoring ---------------------------------------------------------------------------------

_SCALE = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}
_NUM = re.compile(
    r"(?P<neg>-|−|\()?\s*(?:US)?\$?\s*(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)\s*\)?"
    r"(?:\s*(?P<scale>thousands?|millions?|billions?|trillion|mm|mn|bn|tn|[kmb])\b)?",
    re.I,
)


def parse_number(answer: str) -> float | None:
    """First number in a free-text answer, honouring $, commas, parentheses and scale words."""
    text = answer.strip().replace("**", "")
    if re.fullmatch(r"(?i)\W*unknown\W*", text or "x"):
        return None
    for m in _NUM.finditer(text):
        raw = m.group("num").replace(",", "")
        if raw in {".", ""}:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        scale = (m.group("scale") or "").lower()
        value *= _SCALE.get(scale, 1.0)
        if m.group("neg"):
            value = -value
        return value
    return None


@dataclass
class Score:
    correct: bool
    parsed: float | None
    off_by_scale: bool = False
    rel_error: float | None = None


def score_numeric(answer: str, expected: float, tolerance: float) -> Score:
    parsed = parse_number(answer)
    if parsed is None:
        return Score(correct=False, parsed=None)

    def close(a: float, b: float) -> bool:
        if b == 0:
            return abs(a) <= 0.5
        return abs(a - b) <= tolerance * abs(b)

    if close(parsed, expected):
        rel = abs(parsed - expected) / abs(expected) if expected else 0.0
        return Score(correct=True, parsed=parsed, rel_error=rel)
    scaled = any(close(parsed * f, expected) for f in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9))
    rel = abs(parsed - expected) / abs(expected) if expected else None
    return Score(correct=False, parsed=parsed, off_by_scale=scaled, rel_error=rel)


def answer_has_single_number(text: str) -> bool:
    return len([m for m in _NUM.finditer(text) if m.group("num") not in {".", ""}]) == 1
