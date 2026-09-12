"""Per-filing context fallback: full document -> financial statements Item -> BM25 chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from spark_llm.evals.config import load_eval_config
from spark_llm.evals.sec.fetch import Filing
from spark_llm.evals.sec.questions import Question
from spark_llm.evals.sec.run import ContextBuilder, _budget, summarise


class FakeEndpoint:
    """Whitespace tokenizer standing in for llama-server; no network."""

    def __init__(self, n_ctx: int) -> None:
        self._n_ctx = n_ctx

    def n_ctx(self) -> int:
        return self._n_ctx

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) for w in text.split()]

    def detokenize(self, tokens: list[int]) -> str:
        return " ".join(self._vocab[t] for t in tokens)

    def prime(self, text: str) -> None:
        self._vocab = {hash(w): w for w in text.split()}


def _filing(tmp_path: Path, text: str) -> Filing:
    p = tmp_path / "f.txt"
    p.write_text(text)
    return Filing(
        ticker="GS",
        cik=886982,
        company="Goldman Sachs",
        form="10-K",
        accession="0000886982-26-000001",
        primary_document="gs.htm",
        filing_date="2026-02-20",
        report_date="2025-12-31",
        html_path=str(p),
        text_path=str(p),
    )


def _question() -> Question:
    return Question(
        id="GS:10-K:2025-12-31:Assets",
        tag="Assets",
        label="total assets",
        kind="instant",
        unit="USD",
        period_start=None,
        period_end="2025-12-31",
        expected=1.0e12,
        text="What was total assets as of December 31, 2025?",
        format_hint="",
    )


# A filing whose Item 8 is a small fraction of the whole (as in real 10-Ks). Headings sit on
# their own lines, which is what the section finder keys on.
ITEM8 = "\nItem 8. Financial Statements\n" + "total assets 1,000,000 " * 40
FILING = (
    "\nItem 1. Business\n"
    + "we make things " * 400
    + "\nItem 7. MD&A\n"
    + "results were fine " * 400
    + ITEM8
    + "\nItem 9. Controls\n"
    + "controls are good " * 100
)


@pytest.fixture
def cfg():
    c = load_eval_config()
    return c.model_copy(update={"sec": c.sec.model_copy(update={"chunk_tokens": 50, "top_k": 2})})


def _build(tmp_path: Path, cfg, n_ctx: int, *, chunked: bool = False):
    ep = FakeEndpoint(n_ctx)
    ep.prime(FILING)
    f = _filing(tmp_path, FILING)
    q = _question()
    budget = _budget(n_ctx, cfg, 10)
    return ContextBuilder(ep, cfg).build(f, q, budget, chunked=chunked), len(ep.tokenize(FILING))


def test_full_document_when_it_fits(tmp_path: Path, cfg) -> None:
    ctx, doc_tokens = _build(tmp_path, cfg, n_ctx=doc_tokens_plus(cfg, 100_000))
    assert ctx.mode == "full" and not ctx.fallback and ctx.tokens == doc_tokens
    assert ctx.text is not None and ctx.text.lstrip().startswith("Item 1.")


def doc_tokens_plus(cfg, n: int) -> int:
    return n + cfg.quality.max_tokens + 256 + 10


def test_oversized_filing_falls_back_to_financial_statements(tmp_path: Path, cfg) -> None:
    # Budget too small for the ~3.4K-token filing, big enough for the ~200-token Item 8.
    ctx, doc_tokens = _build(tmp_path, cfg, n_ctx=doc_tokens_plus(cfg, 400))
    assert doc_tokens > 400
    assert ctx.mode == "section" and ctx.fallback
    assert ctx.text is not None and ctx.text.lstrip().startswith("Item 8.")
    assert "Item 9." not in ctx.text
    assert 0 < ctx.tokens <= 400


def test_oversized_section_falls_back_to_chunks(tmp_path: Path, cfg) -> None:
    # Budget below Item 8 (~200 tokens) but above two 50-token chunks.
    ctx, _ = _build(tmp_path, cfg, n_ctx=doc_tokens_plus(cfg, 120))
    assert ctx.mode == "chunked" and ctx.fallback
    assert ctx.text is not None and "total assets" in ctx.text
    assert ctx.tokens == 100  # top_k=2 chunks of 50


def test_nothing_fits_is_skipped_with_reason(tmp_path: Path, cfg) -> None:
    ctx, _ = _build(tmp_path, cfg, n_ctx=doc_tokens_plus(cfg, 20))
    assert ctx.text is None and ctx.mode == "chunked" and ctx.fallback
    assert ctx.reason and "filing is" in ctx.reason and "no chunk fits" in ctx.reason


def test_chunked_task_is_not_a_fallback(tmp_path: Path, cfg) -> None:
    ctx, _ = _build(tmp_path, cfg, n_ctx=doc_tokens_plus(cfg, 100_000), chunked=True)
    assert ctx.mode == "chunked" and not ctx.fallback


def test_summary_breaks_down_by_mode_and_counts_fallbacks() -> None:
    s = summarise(
        [
            {"correct": True, "skipped": False, "mode": "full", "fallback": False},
            {"correct": False, "skipped": False, "mode": "section", "fallback": True},
            {"correct": True, "skipped": False, "mode": "section", "fallback": True},
            {"correct": None, "skipped": True, "mode": "chunked", "fallback": True},
        ]
    )
    assert s["by_mode"] == {"full": {"n": 1, "correct": 1}, "section": {"n": 2, "correct": 1}}
    assert s["fallback"] == 3
