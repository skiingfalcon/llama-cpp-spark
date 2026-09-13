"""EDGAR inline-XBRL HTML -> plain text, section (Item) location, and token chunking."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

_WS = re.compile(r"[ \t\xa0  ]+")
_BLANKS = re.compile(r"\n{3,}")
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)


def _flatten_table(table: Tag) -> str:
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if not cells:
            continue
        line = " | ".join(cells)
        # Re-attach currency / percent / paren fragments split into their own cells.
        line = re.sub(r"\$\s*\|\s*", "$", line)
        line = re.sub(r"\(\s*\|\s*", "(", line)
        line = re.sub(r"\s*\|\s*\)", ")", line)
        line = re.sub(r"\s*\|\s*%", "%", line)
        rows.append(line)
    return "\n".join(rows)


def html_to_text(html: str) -> str:
    """Readable text: hidden XBRL header dropped, tables flattened to pipe rows."""
    with warnings.catch_warnings():
        # EDGAR inline-XBRL declares itself XML; the HTML parser is what we want.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN):
        tag.decompose()
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n" + _flatten_table(table) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    text = soup.get_text("\n")
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text)
    return text.strip() + "\n"


@dataclass
class Section:
    label: str  # e.g. "7" for 10-K Item 7, "I.2" for 10-Q Part I Item 2
    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[abc]?)\s*[.:\-–—]?\s*(?=\S|$)", re.I | re.M)
_PART = re.compile(r"^\s*part\s+(i{1,3}|iv)\b", re.I | re.M)


def find_sections(text: str, form: str) -> dict[str, Section]:
    """Locate Item headings, ignoring the table of contents.

    Strategy: collect every heading occurrence; the body heading for an item is the one
    whose span to the next heading is longest (TOC entries are packed together).
    10-Q items repeat across Part I / Part II, so their labels are prefixed with the part.
    """
    parts = (
        [(m.start(), m.group(1).upper()) for m in _PART.finditer(text)] if form == "10-Q" else []
    )
    heads: list[tuple[int, str]] = []
    for m in _ITEM.finditer(text):
        label = m.group(1).upper()
        if form == "10-Q":
            part = "I"
            for pos, name in parts:
                if pos <= m.start():
                    part = name
            label = f"{part}.{label}"
        heads.append((m.start(), label))
    heads.append((len(text), "__end__"))
    best: dict[str, Section] = {}
    for (pos, label), (nxt, _) in zip(heads, heads[1:], strict=False):
        if label == "__end__":
            continue
        span = nxt - pos
        if label not in best or span > (best[label].end - best[label].start):
            best[label] = Section(label=label, start=pos, end=nxt)
    return best


MDNA = {"10-K": "7", "10-Q": "I.2"}
RISK_FACTORS = {"10-K": "1A", "10-Q": "II.1A"}
FINANCIAL_STATEMENTS = {"10-K": "8", "10-Q": "I.1"}


def section_text(text: str, form: str, which: dict[str, str]) -> str | None:
    label = which.get(form)
    if label is None:
        return None
    sec = find_sections(text, form).get(label)
    return sec.slice(text) if sec else None


def chunk_tokens(ids: list[int], size: int, overlap: int = 0) -> list[list[int]]:
    """Fixed-size token windows; ``overlap`` tokens repeated between neighbours."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(size - overlap, 1)
    return [ids[i : i + size] for i in range(0, max(len(ids), 1), step) if ids[i : i + size]]


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
