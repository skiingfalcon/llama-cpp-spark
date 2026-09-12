"""SEC suite tasks: numeric extraction (full / chunked context), FinanceBench QA, MD&A summary."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from rich.console import Console

from spark_llm.config import Settings
from spark_llm.evals.config import EvalConfig, load_prompt
from spark_llm.evals.endpoint import ChatResult, Endpoint, OpenAIEndpoint, percentile, run_parallel
from spark_llm.evals.runs import RunRecord, RunWriter
from spark_llm.evals.sec.fetch import Filing, load_manifest
from spark_llm.evals.sec.parse import MDNA, chunk_tokens, section_text, words
from spark_llm.evals.sec.questions import (
    Question,
    answer_has_single_number,
    parse_number,
    questions_for_filing,
    score_numeric,
)
from spark_llm.provenance import collect
from spark_llm.registry import Registry, load_registry
from spark_llm.server import merge_runtime

console = Console(stderr=True)

TASKS = ("extract-full", "extract-chunked", "qa-financebench", "summarise-mdna")
PROMPT_OVERHEAD_TOKENS = 256  # system prompt + template scaffolding, conservative


@dataclass
class SecRunOptions:
    task: str
    host: str = "127.0.0.1"
    port: int | None = None
    limit: int | None = None
    parallel: int = 1
    tickers: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    judge_port: int | None = None
    provider: str = "local"
    context_window: int | None = None


def endpoint_for(
    settings: Settings, registry: Registry, model: str, host: str, port: int | None
) -> Endpoint:
    spec = registry.get(model)
    listen = port or spec.port or settings.base_port
    ep = Endpoint(host, listen, model=model)
    if not ep.healthy():
        raise RuntimeError(
            f"{model} is not serving on {host}:{listen}; run: spark-llm serve {model}"
        )
    return ep


def openai_endpoint_for(settings: Settings, model: str, context_window: int | None) -> Endpoint:
    api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or ""
    base_url = os.environ.get("OPENAI_BASE_URL", settings.openai_base_url)
    ep = OpenAIEndpoint(
        api_key,
        model,
        base_url=base_url,
        context_window=context_window or settings.openai_context_window,
    )
    if not ep.healthy():
        ep.close()
        raise RuntimeError(
            "OpenAI API preflight failed; check OPENAI_API_KEY, OPENAI_BASE_URL, and model access"
        )
    return ep


class Judge:
    def __init__(self, endpoint: Endpoint, max_tokens: int) -> None:
        self.ep = endpoint
        self.max_tokens = max_tokens
        self.prompt = load_prompt("sec_judge")
        self.summary_prompt = load_prompt("sec_summary_judge")

    def _json(self, text: str) -> dict[str, Any]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    def verdict(self, question: str, expected: str, answer: str) -> bool | None:
        msg = self.prompt.format(question=question, expected=expected, answer=answer)
        r = self.ep.chat(
            [{"role": "user", "content": msg}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            stream=False,
        )
        if not r.ok:
            return None
        v = self._json(r.content).get("verdict")
        return None if v is None else str(v).lower().startswith("correct")

    def summary_scores(self, document: str, answer: str) -> dict[str, Any] | None:
        msg = self.summary_prompt.format(document=document, answer=answer)
        r = self.ep.chat(
            [{"role": "user", "content": msg}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return self._json(r.content) if r.ok else None


class DocCache:
    """Per-filing text and token ids, tokenised once through the served model."""

    def __init__(self, ep: Endpoint) -> None:
        self.ep = ep
        self._text: dict[str, str] = {}
        self._ids: dict[str, list[int]] = {}

    def text(self, f: Filing) -> str:
        if f.key not in self._text:
            self._text[f.key] = Path(f.text_path).read_text()
        return self._text[f.key]

    def ids(self, f: Filing) -> list[int]:
        if f.key not in self._ids:
            self._ids[f.key] = self.ep.tokenize(self.text(f))
        return self._ids[f.key]


def _filter(filings: list[Filing], opts: SecRunOptions) -> list[Filing]:
    out = filings
    if opts.tickers:
        out = [f for f in out if f.ticker in opts.tickers]
    if opts.forms:
        out = [f for f in out if f.form in opts.forms]
    return out


def _timing(r: ChatResult) -> dict[str, Any]:
    return r.as_dict()


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if not r.get("skipped") and r.get("correct") is not None]
    ttft = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]
    total = [r["total_s"] for r in results if r.get("total_s") is not None]
    ptps = [r["prompt_tps"] for r in results if r.get("prompt_tps")]
    dtps = [r["decode_tps"] for r in results if r.get("decode_tps")]
    tokens = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in results)
    cached_tokens = sum(r.get("cached_prompt_tokens") or 0 for r in results)
    reasoning_tokens = sum(r.get("reasoning_tokens") or 0 for r in results)
    by_form: dict[str, dict[str, int]] = {}
    for r in scored:
        b = by_form.setdefault(r.get("form", "?"), {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(bool(r["correct"]))
    return {
        "n": len(results),
        "scored": len(scored),
        "correct": sum(int(bool(r["correct"])) for r in scored),
        "score": (sum(int(bool(r["correct"])) for r in scored) / len(scored)) if scored else None,
        "skipped": sum(int(bool(r.get("skipped"))) for r in results),
        "errors": sum(1 for r in results if r.get("error")),
        "off_by_scale": sum(int(bool(r.get("off_by_scale"))) for r in results),
        "ttft_p50_s": percentile(ttft, 0.5),
        "ttft_p95_s": percentile(ttft, 0.95),
        "total_p50_s": percentile(total, 0.5),
        "total_p95_s": percentile(total, 0.95),
        "prompt_tps_p50": percentile(ptps, 0.5),
        "decode_tps_p50": percentile(dtps, 0.5),
        "total_tokens": tokens,
        "cached_prompt_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "by_form": by_form,
    }


def _ask(ep: Endpoint, cfg: EvalConfig, system: str, user: str) -> ChatResult:
    q = cfg.quality
    return ep.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=q.temperature,
        seed=q.seed,
        max_tokens=q.max_tokens,
        stream=True,
        cache_prompt=True,
    )


def _budget(n_ctx: int | None, cfg: EvalConfig, question_tokens: int) -> int | None:
    if not n_ctx:
        return None
    cap = cfg.sec.max_input_tokens or n_ctx
    return min(cap, n_ctx) - cfg.quality.max_tokens - PROMPT_OVERHEAD_TOKENS - question_tokens


# -- tasks ------------------------------------------------------------------------------------


def _extract_items(
    manifest_filings: list[Filing], facts_paths: dict[str, str], cfg: EvalConfig
) -> list[tuple[Filing, Question]]:
    items: list[tuple[Filing, Question]] = []
    facts_cache: dict[str, dict[str, Any]] = {}
    for f in manifest_filings:
        if f.ticker not in facts_cache:
            facts_cache[f.ticker] = json.loads(Path(facts_paths[f.ticker]).read_text())
        qs = questions_for_filing(
            facts_cache[f.ticker],
            cfg.sec.xbrl_tags,
            form=f.form,
            accession=f.accession,
            report_date=f.report_date,
            key=f.key,
        )
        items.extend((f, q) for q in qs)
    return items


def run_extract(
    ep: Endpoint,
    cfg: EvalConfig,
    writer: RunWriter,
    filings: list[Filing],
    facts_paths: dict[str, str],
    opts: SecRunOptions,
    *,
    chunked: bool,
) -> list[dict[str, Any]]:
    system = load_prompt("sec_extract_system")
    user_tpl = load_prompt("sec_extract_user")
    items = _extract_items(filings, facts_paths, cfg)
    if opts.limit:
        items = items[: opts.limit]
    docs = DocCache(ep)
    n_ctx = ep.n_ctx()
    seen_filings: set[str] = set()
    bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}

    def context_for(
        f: Filing, q: Question, budget: int | None
    ) -> tuple[str | None, str | None, int]:
        ids = docs.ids(f)
        if not chunked:
            if budget is not None and len(ids) > budget:
                return (
                    None,
                    f"filing is {len(ids)} tokens; budget {budget} at ctx {n_ctx}",
                    len(ids),
                )
            return docs.text(f), None, len(ids)
        if f.key not in bm25_cache:
            chunks = [ep.detokenize(c) for c in chunk_tokens(ids, cfg.sec.chunk_tokens)]
            bm25_cache[f.key] = (BM25Okapi([words(c) for c in chunks]), chunks)
        bm25, chunks = bm25_cache[f.key]
        scores = bm25.get_scores(words(f"{q.text} {q.label} {q.tag}"))
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[: cfg.sec.top_k]
        keep: list[int] = []
        used = 0
        for i in ranked:
            n = cfg.sec.chunk_tokens
            if budget is not None and used + n > budget:
                continue
            keep.append(i)
            used += n
        if not keep:
            return None, f"no chunk fits budget {budget}", len(ids)
        ctx = "\n\n[...]\n\n".join(chunks[i] for i in sorted(keep))
        return ctx, None, used

    def one(item: tuple[Filing, Question]) -> dict[str, Any]:
        f, q = item
        warm = f.key in seen_filings
        seen_filings.add(f.key)
        base = {
            "id": q.id,
            "ticker": f.ticker,
            "form": f.form,
            "report_date": f.report_date,
            "tag": q.tag,
            "kind": q.kind,
            "unit": q.unit,
            "expected": q.expected,
            "question": q.text,
            "warm": warm,
            "mode": "chunked" if chunked else "full",
        }
        budget = _budget(n_ctx, cfg, len(words(q.text)) * 2)
        context, reason, ctx_tokens = context_for(f, q, budget)
        base["doc_tokens"] = len(docs.ids(f))
        base["context_tokens"] = ctx_tokens
        if context is None:
            rec = {**base, "skipped": True, "reason": reason, "correct": None}
            writer.write(rec)
            return rec
        user = user_tpl.format(
            form=f.form,
            company=f.company,
            period_end=f.report_date,
            document=context,
            question=q.text,
            format_hint=q.format_hint,
        )
        r = _ask(ep, cfg, system, user)
        if not r.ok:
            rec = {
                **base,
                "skipped": True,
                "reason": f"http {r.status}",
                "error": r.error,
                "correct": None,
                **_timing(r),
            }
            writer.write(rec)
            return rec
        s = score_numeric(r.content, q.expected, cfg.sec.tolerance)
        rec = {
            **base,
            "answer": r.content.strip()[:300],
            "parsed": s.parsed,
            "correct": s.correct,
            "off_by_scale": s.off_by_scale,
            "rel_error": s.rel_error,
            "skipped": False,
            **_timing(r),
        }
        writer.write(rec)
        mark = "ok" if s.correct else ("scale" if s.off_by_scale else "miss")
        console.print(f"[dim]{mark:5s}[/dim] {q.id}: {rec['answer'][:40]!r} vs {q.expected:,.2f}")
        return rec

    return run_parallel(items, one, opts.parallel)


def load_financebench(cfg: EvalConfig) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    fb = cfg.sec.financebench
    path = hf_hub_download(repo_id=fb.repo, filename=fb.file, repo_type="dataset")
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def run_financebench(
    ep: Endpoint, cfg: EvalConfig, writer: RunWriter, judge: Judge | None, opts: SecRunOptions
) -> list[dict[str, Any]]:
    """Evidence-grounded QA: the human-selected evidence passages are the context.

    This isolates reading/reasoning from retrieval; full-document mode needs the PDFs,
    which are out of scope here.
    """
    system = load_prompt("sec_extract_system")
    user_tpl = load_prompt("sec_qa_user")
    items = load_financebench(cfg)
    if opts.limit:
        items = items[: opts.limit]

    def one(it: dict[str, Any]) -> dict[str, Any]:
        evidence = "\n\n".join(e.get("evidence_text", "") for e in it.get("evidence", []))
        user = user_tpl.format(
            doc_name=it.get("doc_name", ""), document=evidence, question=it["question"]
        )
        r = _ask(ep, cfg, system, user)
        base = {
            "id": it.get("financebench_id"),
            "ticker": it.get("company"),
            "form": it.get("doc_type", "10-K"),
            "doc_name": it.get("doc_name"),
            "question_type": it.get("question_type"),
            "question": it["question"],
            "expected": it["answer"],
        }
        if not r.ok:
            rec = {
                **base,
                "skipped": True,
                "reason": f"http {r.status}",
                "error": r.error,
                "correct": None,
                **_timing(r),
            }
            writer.write(rec)
            return rec
        answer = r.content.strip()
        method = "numeric"
        expected_num = (
            parse_number(str(it["answer"])) if answer_has_single_number(str(it["answer"])) else None
        )
        if expected_num is not None:
            correct: bool | None = score_numeric(answer, expected_num, 0.01).correct
        elif judge is not None:
            method = "judge"
            correct = judge.verdict(it["question"], str(it["answer"]), answer)
        else:
            method = "unscored"
            correct = None
        rec = {
            **base,
            "answer": answer[:500],
            "correct": correct,
            "method": method,
            "skipped": False,
            **_timing(r),
        }
        writer.write(rec)
        return rec

    return run_parallel(items, one, opts.parallel)


def run_summary(
    ep: Endpoint,
    cfg: EvalConfig,
    writer: RunWriter,
    judge: Judge | None,
    filings: list[Filing],
    opts: SecRunOptions,
) -> list[dict[str, Any]]:
    system = load_prompt("sec_extract_system")
    user_tpl = load_prompt("sec_summary_user")
    docs = DocCache(ep)
    n_ctx = ep.n_ctx()
    items = filings[: opts.limit] if opts.limit else filings

    def one(f: Filing) -> dict[str, Any]:
        base = {
            "id": f"{f.key}:mdna",
            "ticker": f.ticker,
            "form": f.form,
            "report_date": f.report_date,
        }
        mdna = section_text(docs.text(f), f.form, MDNA)
        if not mdna:
            rec = {**base, "skipped": True, "reason": "MD&A section not found", "correct": None}
            writer.write(rec)
            return rec
        budget = _budget(n_ctx, cfg, 0)
        ids = ep.tokenize(mdna)
        if budget is not None and len(ids) > budget:
            mdna = ep.detokenize(ids[:budget])
        user = user_tpl.format(
            company=f.company, form=f.form, period_end=f.report_date, document=mdna
        )
        r = _ask(ep, cfg, system, user)
        if not r.ok:
            rec = {
                **base,
                "skipped": True,
                "reason": f"http {r.status}",
                "error": r.error,
                "correct": None,
                **_timing(r),
            }
            writer.write(rec)
            return rec
        scores = judge.summary_scores(mdna, r.content) if judge else None
        mean = None
        if scores:
            vals = [
                float(scores[k]) for k in ("faithfulness", "coverage", "concision") if k in scores
            ]
            mean = sum(vals) / len(vals) / 5.0 if vals else None
        rec = {
            **base,
            "answer": r.content.strip()[:1500],
            "judge": scores,
            "score": mean,
            "correct": (mean is not None and mean >= 0.8) if mean is not None else None,
            "skipped": False,
            **_timing(r),
        }
        writer.write(rec)
        return rec

    return run_parallel(items, one, opts.parallel)


def run_sec_task(settings: Settings, cfg: EvalConfig, model: str, opts: SecRunOptions) -> RunRecord:
    if opts.task not in TASKS:
        raise ValueError(f"unknown task {opts.task!r}; choose from {', '.join(TASKS)}")
    if opts.provider not in {"local", "openai"}:
        raise ValueError("provider must be 'local' or 'openai'")
    registry = load_registry(settings=settings)
    if opts.provider == "openai":
        ep = openai_endpoint_for(settings, model, opts.context_window)
        run_model = f"openai:{model}"
        runtime = {
            "provider": "openai",
            "model": model,
            "context_window": ep.n_ctx(),
        }
        quality = {
            **cfg.quality.model_dump(),
            "temperature": None,
            "seed": None,
            "note": "provider defaults; frontier reasoning models reject fixed temperature/seed",
        }
        provenance = collect(settings, with_gpu=False)
    else:
        ep = endpoint_for(settings, registry, model, opts.host, opts.port)
        run_model = model
        runtime = merge_runtime(registry.get(model), registry.defaults, settings).as_dict()
        quality = cfg.quality.model_dump()
        provenance = None
    manifest = load_manifest(settings)
    filings = _filter(manifest.filings, opts)
    judge: Judge | None = None
    if opts.task in {"qa-financebench", "summarise-mdna"}:
        try:
            jep = endpoint_for(settings, registry, cfg.judge.model, opts.host, opts.judge_port)
            judge = Judge(jep, cfg.judge.max_tokens)
        except (RuntimeError, KeyError) as exc:
            console.print(f"[yellow]judge unavailable[/yellow] ({exc}); free-text items unscored")

    writer = RunWriter(
        settings,
        "sec",
        opts.task,
        run_model,
        server=ep.server_summary(),
        runtime=runtime,
        quality=quality,
        task_config={
            "chunk_tokens": cfg.sec.chunk_tokens,
            "top_k": cfg.sec.top_k,
            "tolerance": cfg.sec.tolerance,
            "tags": [t.tag for t in cfg.sec.xbrl_tags],
            "companies": [c.ticker for c in cfg.sec.companies],
            "forms": cfg.sec.forms,
            "judge": cfg.judge.model if judge else None,
            "parallel": opts.parallel,
            "limit": opts.limit,
            "filters": {"tickers": opts.tickers, "forms": opts.forms},
        },
        provenance=provenance,
    )
    if opts.task == "extract-full":
        results = run_extract(ep, cfg, writer, filings, manifest.facts_paths, opts, chunked=False)
    elif opts.task == "extract-chunked":
        results = run_extract(ep, cfg, writer, filings, manifest.facts_paths, opts, chunked=True)
    elif opts.task == "qa-financebench":
        results = run_financebench(ep, cfg, writer, judge, opts)
    else:
        results = run_summary(ep, cfg, writer, judge, filings, opts)
    summary = summarise(results)
    if opts.task == "summarise-mdna":
        vals = [r["score"] for r in results if r.get("score") is not None]
        summary["score"] = sum(vals) / len(vals) if vals else None
    rec = writer.finish(summary)
    console.print(f"[green]done[/green] {writer.dir}")
    return rec
