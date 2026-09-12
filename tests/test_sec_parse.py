"""EDGAR HTML -> text, Item section location, chunking."""

from __future__ import annotations

from spark_llm.evals.sec.parse import (
    FINANCIAL_STATEMENTS,
    MDNA,
    RISK_FACTORS,
    chunk_tokens,
    find_sections,
    html_to_text,
    section_text,
    words,
)

TEN_K = """<?xml version='1.0' encoding='ASCII'?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>x</title>
<style>.a{}</style></head><body>
<div style="display:none"><ix:header>HIDDEN XBRL CONTEXT</ix:header></div>
<p>TABLE OF CONTENTS</p>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
<tr><td>Item 7.</td><td>Management's Discussion</td><td>20</td></tr>
<tr><td>Item 8.</td><td>Financial Statements</td><td>30</td></tr>
</table>
<p>PART I</p>
<p><b>Item 1.</b> Business</p><p>The Company designs widgets. Lots of text here about widgets.</p>
<p><b>Item 1A.</b> Risk Factors</p><p>Widgets may fail. Many risks follow in long prose.</p>
<p><b>Item 7.</b> Management's Discussion and Analysis</p>
<p>Net sales increased 5% driven by widget demand.</p>
<table>
<tr><th></th><th>2025</th><th>2024</th></tr>
<tr><td>Total net sales</td><td>$</td><td>416,161</td><td>$</td><td>391,035</td></tr>
<tr><td>Net loss</td><td>(</td><td>1,234</td><td>)</td><td>5%</td></tr>
</table>
<p><b>Item 8.</b> Financial Statements and Supplementary Data</p><p>Balance sheet follows.</p>
</body></html>
"""


def test_html_to_text_flattens_tables_and_drops_hidden() -> None:
    text = html_to_text(TEN_K)
    assert "HIDDEN XBRL CONTEXT" not in text
    assert ".a{}" not in text
    assert "Total net sales | $416,161 | $391,035" in text
    assert "Net loss | (1,234) | 5%" in text


def test_find_sections_skips_table_of_contents() -> None:
    text = html_to_text(TEN_K)
    secs = find_sections(text, "10-K")
    assert {"1", "1A", "7", "8"} <= set(secs)
    mdna = section_text(text, "10-K", MDNA)
    assert mdna is not None
    assert "Net sales increased 5%" in mdna
    assert "Balance sheet" not in mdna  # ends at Item 8
    risk = section_text(text, "10-K", RISK_FACTORS)
    assert risk is not None and "Widgets may fail" in risk
    fin = section_text(text, "10-K", FINANCIAL_STATEMENTS)
    assert fin is not None and "Balance sheet follows" in fin


def test_10q_sections_are_part_scoped() -> None:
    text = (
        "PART I\nItem 1. Financial Statements\nnumbers numbers numbers\n"
        "Item 2. Management's Discussion and Analysis\nsales rose a lot this quarter\n"
        "PART II\nItem 1. Legal Proceedings\nnone\nItem 1A. Risk Factors\nrisks risks\n"
    )
    secs = find_sections(text, "10-Q")
    assert {"I.1", "I.2", "II.1", "II.1A"} <= set(secs)
    assert "sales rose" in section_text(text, "10-Q", MDNA)
    assert "risks risks" in section_text(text, "10-Q", RISK_FACTORS)


def test_chunk_tokens_and_words() -> None:
    ids = list(range(10))
    assert chunk_tokens(ids, 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert chunk_tokens(ids, 4, overlap=1) == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9]]
    assert words("Net Sales: $416,161 (millions)") == ["net", "sales", "416", "161", "millions"]
