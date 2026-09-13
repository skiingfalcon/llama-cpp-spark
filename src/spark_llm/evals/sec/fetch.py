"""Fetch a pinned corpus of 10-K / 10-Q filings and XBRL company facts from EDGAR."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from spark_llm.config import Settings
from spark_llm.console import err as console
from spark_llm.evals.config import SecSuite
from spark_llm.evals.sec.parse import html_to_text

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
MIN_INTERVAL_S = 0.15  # EDGAR fair-access limit is 10 req/s


class Filing(BaseModel):
    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    primary_document: str
    filing_date: str
    report_date: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    html_path: str
    text_path: str
    n_chars: int = 0

    @property
    def key(self) -> str:
        return f"{self.ticker}:{self.form}:{self.report_date}"


class Manifest(BaseModel):
    created: str
    filings: list[Filing] = Field(default_factory=list)
    facts_paths: dict[str, str] = Field(default_factory=dict)  # ticker -> companyfacts json


class Edgar:
    def __init__(self, user_agent: str) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "EDGAR requires a descriptive User-Agent with a contact email; set "
                "LOCAL_LLM_EDGAR_USER_AGENT='local-llm you@example.com'"
            )
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._last = 0.0

    def _get(self, url: str) -> httpx.Response:
        for attempt in range(5):
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self._client.get(url)
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    def submissions(self, cik: int) -> dict[str, Any]:
        return self._get(SUBMISSIONS.format(cik=cik)).json()

    def companyfacts(self, cik: int) -> dict[str, Any]:
        return self._get(COMPANYFACTS.format(cik=cik)).json()

    def document(self, cik: int, accession: str, doc: str) -> str:
        url = ARCHIVE.format(cik=cik, acc_nodash=accession.replace("-", ""), doc=doc)
        return self._get(url).text


def pick_recent(sub: dict[str, Any], forms: dict[str, int]) -> list[dict[str, Any]]:
    """Latest N filings per form from the submissions 'recent' block, newest first."""
    recent = sub.get("filings", {}).get("recent", {})
    keys = [
        "form",
        "accessionNumber",
        "primaryDocument",
        "filingDate",
        "reportDate",
    ]
    rows = [
        dict(zip(keys, vals, strict=False))
        for vals in zip(*(recent.get(k, []) for k in keys), strict=False)
    ]
    picked: list[dict[str, Any]] = []
    for form, n in forms.items():
        picked.extend([r for r in rows if r["form"] == form and r["primaryDocument"]][:n])
    return picked


def corpus_dir(settings: Settings) -> Path:
    return settings.evals_data_dir / "sec"


def manifest_path(settings: Settings) -> Path:
    return corpus_dir(settings) / "manifest.json"


def load_manifest(settings: Settings) -> Manifest:
    path = manifest_path(settings)
    if not path.is_file():
        raise FileNotFoundError(
            f"SEC corpus not fetched yet ({path}); run: local-llm eval sec fetch"
        )
    return Manifest.model_validate_json(path.read_text())


def fiscal_period(form: str, report_date: str, fy_end_month: int | None) -> str | None:
    if form == "10-K":
        return "FY"
    return None  # 10-Q fiscal quarter is taken from XBRL facts where needed


def build_corpus(settings: Settings, cfg: SecSuite, *, force: bool = False) -> Manifest:
    ua = settings.edgar_user_agent or ""
    edgar = Edgar(ua)
    root = corpus_dir(settings)
    root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for company in cfg.companies:
        sub = edgar.submissions(company.cik)
        name = sub.get("name", company.ticker)
        cdir = root / company.ticker
        cdir.mkdir(exist_ok=True)
        facts_path = cdir / "companyfacts.json"
        if force or not facts_path.is_file():
            facts_path.write_text(json.dumps(edgar.companyfacts(company.cik)))
        manifest.facts_paths[company.ticker] = str(facts_path)
        for row in pick_recent(sub, cfg.forms):
            acc = row["accessionNumber"]
            html_path = cdir / f"{acc}.htm"
            text_path = cdir / f"{acc}.txt"
            if force or not html_path.is_file():
                console.print(
                    f"[cyan]fetch[/cyan] {company.ticker} {row['form']} {row['reportDate']}"
                )
                html_path.write_text(edgar.document(company.cik, acc, row["primaryDocument"]))
            if force or not text_path.is_file():
                text_path.write_text(html_to_text(html_path.read_text()))
            manifest.filings.append(
                Filing(
                    ticker=company.ticker,
                    cik=company.cik,
                    company=name,
                    form=row["form"],
                    accession=acc,
                    primary_document=row["primaryDocument"],
                    filing_date=row["filingDate"],
                    report_date=row["reportDate"],
                    fiscal_period=fiscal_period(row["form"], row["reportDate"], None),
                    html_path=str(html_path),
                    text_path=str(text_path),
                    n_chars=text_path.stat().st_size,
                )
            )
    manifest_path(settings).write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest
