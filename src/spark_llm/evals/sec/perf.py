"""Long-context serving performance on real filing text: TTFT / prefill / decode by length,
cold vs warm prefix cache, and aggregate throughput under concurrency."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from spark_llm.config import Settings
from spark_llm.evals.config import EvalConfig, load_prompt
from spark_llm.evals.endpoint import ChatResult, percentile, run_parallel
from spark_llm.evals.runs import RunRecord, RunWriter
from spark_llm.evals.sec.fetch import load_manifest
from spark_llm.evals.sec.run import endpoint_for
from spark_llm.gpu import gpu_memory_used_mib
from spark_llm.registry import load_registry
from spark_llm.server import merge_runtime

console = Console(stderr=True)

Q_COLD = "In one sentence, what is the company's principal business?"
Q_WARM = "In one sentence, what fiscal period does this filing cover?"


@dataclass
class PerfOptions:
    host: str = "127.0.0.1"
    port: int | None = None
    lengths: list[int] = field(default_factory=list)  # empty = evals.toml perf_lengths
    concurrency: list[int] = field(default_factory=lambda: [1, 4])
    max_tokens: int = 64
    concurrency_prompt_tokens: int = 8192


def _record(kind: str, length: int, r: ChatResult, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "length": length, "fits": r.ok, **r.as_dict(), **extra}


def run_sec_perf(settings: Settings, cfg: EvalConfig, model: str, opts: PerfOptions) -> RunRecord:
    registry = load_registry(settings=settings)
    ep = endpoint_for(settings, registry, model, opts.host, opts.port)
    manifest = load_manifest(settings)
    lengths = opts.lengths or cfg.sec.perf_lengths
    system = load_prompt("sec_extract_system")

    # Build one long token stream from the longest filings; slices of it are our prompts.
    filings = sorted(manifest.filings, key=lambda f: -f.n_chars)
    ids: list[int] = []
    need = max([*lengths, opts.concurrency_prompt_tokens * max(opts.concurrency or [1])])
    for f in filings:
        ids.extend(ep.tokenize(Path(f.text_path).read_text()))
        if len(ids) >= need:
            break
    n_ctx = ep.n_ctx()
    server = ep.server_summary()
    rt = merge_runtime(registry.get(model), registry.defaults, settings)
    writer = RunWriter(
        settings,
        "sec",
        "perf",
        model,
        server=server,
        runtime=rt.as_dict(),
        quality={"max_tokens": opts.max_tokens},
        task_config={
            "lengths": lengths,
            "concurrency": opts.concurrency,
            "concurrency_prompt_tokens": opts.concurrency_prompt_tokens,
            "source_filings": [f.key for f in filings[:3]],
        },
    )

    def prompt(prefix_ids: list[int], question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"<filing>\n{ep.detokenize(prefix_ids)}\n</filing>\n\n{question}",
            },
        ]

    results: list[dict[str, Any]] = []
    for length in lengths:
        if length > len(ids):
            rec = {
                "kind": "cold",
                "length": length,
                "fits": False,
                "reason": "corpus shorter than length",
            }
            writer.write(rec)
            results.append(rec)
            continue
        if n_ctx and length + opts.max_tokens + 128 > n_ctx:
            rec = {
                "kind": "cold",
                "length": length,
                "fits": False,
                "reason": f"exceeds served ctx {n_ctx}",
            }
            writer.write(rec)
            results.append(rec)
            console.print(f"[yellow]skip[/yellow] {length} tokens > ctx {n_ctx}")
            continue
        prefix = ids[:length]
        mem0 = gpu_memory_used_mib()
        cold = ep.chat(
            prompt(prefix, Q_COLD), temperature=0.0, max_tokens=opts.max_tokens, cache_prompt=False
        )
        mem1 = gpu_memory_used_mib()
        warm = ep.chat(
            prompt(prefix, Q_WARM), temperature=0.0, max_tokens=opts.max_tokens, cache_prompt=True
        )
        for kind, r in (("cold", cold), ("warm", warm)):
            rec = _record(kind, length, r, gpu_mem_before_mib=mem0, gpu_mem_after_mib=mem1)
            writer.write(rec)
            results.append(rec)
        console.print(
            f"len={length:>7} cold ttft={cold.ttft_s or 0:.2f}s pp={cold.prompt_tps or 0:.0f} t/s "
            f"tg={cold.decode_tps or 0:.1f} t/s | warm ttft={warm.ttft_s or 0:.2f}s"
            + ("" if cold.ok else f"  [red]{cold.status} {cold.error or ''}[/red]")
        )
        if not cold.ok:
            break  # larger lengths will not fit either

    for c in opts.concurrency:
        n = opts.concurrency_prompt_tokens
        if n * c > len(ids) or (n_ctx and n + opts.max_tokens + 128 > n_ctx):
            rec = {
                "kind": "concurrency",
                "concurrency": c,
                "fits": False,
                "reason": "corpus/ctx too small",
            }
            writer.write(rec)
            results.append(rec)
            continue
        prompts = [prompt(ids[i * n : (i + 1) * n], Q_COLD) for i in range(c)]
        t0 = time.perf_counter()
        rs = run_parallel(
            prompts,
            lambda m: ep.chat(m, temperature=0.0, max_tokens=opts.max_tokens, cache_prompt=False),
            parallel=c,
        )
        wall = time.perf_counter() - t0
        completion = sum(r.completion_tokens or 0 for r in rs)
        rec = {
            "kind": "concurrency",
            "concurrency": c,
            "fits": all(r.ok for r in rs),
            "wall_s": wall,
            "aggregate_decode_tps": completion / wall if wall else None,
            "ttft_p50_s": percentile([r.ttft_s for r in rs if r.ttft_s is not None], 0.5),
            "ttft_p95_s": percentile([r.ttft_s for r in rs if r.ttft_s is not None], 0.95),
            "per_request_decode_tps_p50": percentile(
                [r.decode_tps for r in rs if r.decode_tps], 0.5
            ),
            "errors": [r.error for r in rs if not r.ok],
        }
        writer.write(rec)
        results.append(rec)
        console.print(
            f"conc={c}: wall={wall:.1f}s agg tg={rec['aggregate_decode_tps'] or 0:.1f} t/s "
            f"ttft p95={rec['ttft_p95_s'] or 0:.2f}s"
        )

    cold_ok = [r for r in results if r.get("kind") == "cold" and r.get("fits")]
    warm_ok = [r for r in results if r.get("kind") == "warm" and r.get("fits")]
    summary = {
        "n": len(results),
        "score": None,
        "skipped": sum(1 for r in results if not r.get("fits")),
        "max_fit_tokens": max((r["length"] for r in cold_ok), default=0),
        "ttft_p50_s": percentile(
            [r["ttft_s"] for r in cold_ok if r.get("ttft_s") is not None], 0.5
        ),
        "prompt_tps_p50": percentile(
            [r["prompt_tps"] for r in cold_ok if r.get("prompt_tps")], 0.5
        ),
        "decode_tps_p50": percentile(
            [r["decode_tps"] for r in cold_ok if r.get("decode_tps")], 0.5
        ),
        "warm_ttft_p50_s": percentile(
            [r["ttft_s"] for r in warm_ok if r.get("ttft_s") is not None], 0.5
        ),
        "by_length": {
            str(r["length"]): {
                "ttft_s": r.get("ttft_s"),
                "prompt_tps": r.get("prompt_tps"),
                "decode_tps": r.get("decode_tps"),
            }
            for r in cold_ok
        },
        "concurrency": {
            str(r["concurrency"]): {
                k: r.get(k) for k in ("aggregate_decode_tps", "ttft_p95_s", "fits")
            }
            for r in results
            if r.get("kind") == "concurrency"
        },
        "note": f"fits ≤ {max((r['length'] for r in cold_ok), default=0)} tok",
        "total_tokens": sum(
            (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in results
        ),
    }
    rec = writer.finish(summary)
    console.print(f"[green]done[/green] {writer.dir}")
    return rec
