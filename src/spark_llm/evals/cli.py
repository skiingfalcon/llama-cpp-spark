"""`local-llm eval …` sub-commands."""

from __future__ import annotations

from pathlib import Path

import typer

from spark_llm.config import get_settings
from spark_llm.console import err
from spark_llm.console import out as console
from spark_llm.evals.config import load_eval_config
from spark_llm.evals.report import build_report, latest_runs, write_report
from spark_llm.evals.sec.perf import PerfOptions, run_sec_perf
from spark_llm.evals.sec.run import TASKS, SecRunOptions, run_sec_task
from spark_llm.evals.serving import endpoint_for
from spark_llm.evals.swe.check import tool_call_check
from spark_llm.evals.swe.run import SweRunOptions, run_swe
from spark_llm.registry import load_registry

eval_app = typer.Typer(
    help="Workload evals against a running llama-server (SEC filings, SWE benchmarks).",
    no_args_is_help=True,
)
sec_app = typer.Typer(help="SEC 10-K / 10-Q suite.", no_args_is_help=True)
swe_app = typer.Typer(help="Software-engineering benchmark tiers.", no_args_is_help=True)
eval_app.add_typer(sec_app, name="sec")
eval_app.add_typer(swe_app, name="swe")


def _csv(raw: str | None) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else []


def _ints(raw: str | None) -> list[int]:
    return [int(x) for x in _csv(raw)]


HOST_OPT = typer.Option(
    None, "--host", help="Server host (default LOCAL_LLM_EVAL_HOST / 127.0.0.1)"
)
PORT_OPT = typer.Option(None, "--port", help="Override the model's registered port")


@sec_app.command("fetch")
def sec_fetch(
    force: bool = typer.Option(False, "--force", help="Re-download and re-parse"),
) -> None:
    """Download the pinned 10-K / 10-Q corpus and XBRL company facts from EDGAR."""
    from spark_llm.evals.sec.fetch import build_corpus

    settings = get_settings()
    cfg = load_eval_config(settings=settings)
    try:
        manifest = build_corpus(settings, cfg.sec, force=force)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]ok[/green] {len(manifest.filings)} filings for "
        f"{len(manifest.facts_paths)} companies under {settings.evals_data_dir / 'sec'}"
    )


@sec_app.command("run")
def sec_run(
    model: str = typer.Argument(
        ..., help="Registered local model, or OpenAI model ID with --provider openai"
    ),
    task: str = typer.Option("extract-full", "--task", help=f"One of {', '.join(TASKS)} or 'all'"),
    limit: int | None = typer.Option(None, "--limit", help="Max items"),
    parallel: int = typer.Option(1, "--parallel", help="Concurrent requests"),
    tickers: str | None = typer.Option(None, "--tickers", help="Comma-separated ticker filter"),
    forms: str | None = typer.Option(None, "--forms", help="Comma-separated form filter"),
    host: str | None = HOST_OPT,
    port: int | None = PORT_OPT,
    judge_port: int | None = typer.Option(None, "--judge-port"),
    provider: str = typer.Option("local", "--provider", help="local or openai"),
    context_window: int | None = typer.Option(
        None,
        "--context-window",
        help="OpenAI model context limit (default LOCAL_LLM_OPENAI_CONTEXT_WINDOW)",
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Completion budget incl. hidden reasoning (evals.toml default)"
    ),
    reasoning_effort: str | None = typer.Option(
        None, "--reasoning-effort", help="low | medium | high; '' to force the model default"
    ),
) -> None:
    """Run a SEC task against the served model; results under state/evals/sec/."""
    settings = get_settings()
    cfg = load_eval_config(settings=settings)
    tasks = list(TASKS) if task == "all" else [task]
    for t in tasks:
        opts = SecRunOptions(
            task=t,
            host=host or settings.eval_host,
            port=port,
            limit=limit,
            parallel=parallel,
            tickers=_csv(tickers),
            forms=_csv(forms),
            judge_port=judge_port,
            provider=provider,
            context_window=context_window,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        try:
            rec = run_sec_task(settings, cfg, model, opts)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            err.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        s = rec.summary
        console.print(
            f"[bold]{rec.model}[/bold] {t}: score={s.get('score')} n={s.get('n')} "
            f"skipped={s.get('skipped')} truncated={s.get('truncated')} "
            f"ttft_p50={s.get('ttft_p50_s')}"
        )


@sec_app.command("perf")
def sec_perf(
    model: str = typer.Argument(...),
    lengths: str | None = typer.Option(None, "--lengths", help="Input lengths in tokens"),
    concurrency: str = typer.Option("1,4", "--concurrency"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    host: str | None = HOST_OPT,
    port: int | None = PORT_OPT,
) -> None:
    """TTFT / prefill / decode by input length (cold vs warm cache) and under concurrency."""
    settings = get_settings()
    cfg = load_eval_config(settings=settings)
    opts = PerfOptions(
        host=host or settings.eval_host,
        port=port,
        lengths=_ints(lengths),
        concurrency=_ints(concurrency),
        max_tokens=max_tokens,
    )
    try:
        rec = run_sec_perf(settings, cfg, model, opts)
    except (RuntimeError, FileNotFoundError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(rec.summary.get("by_length"))
    console.print(rec.summary.get("concurrency"))


@swe_app.command("check")
def swe_check(
    model: str = typer.Argument(...),
    host: str | None = HOST_OPT,
    port: int | None = PORT_OPT,
) -> None:
    """Tool-calling smoke test through the served chat template (run before tiers 2/3)."""
    settings = get_settings()
    registry = load_registry(settings=settings)
    try:
        ep = endpoint_for(settings, registry, model, host or settings.eval_host, port)
    except RuntimeError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    result = tool_call_check(ep)
    for label, probe in result["probes"].items():
        mark = "[green]ok[/green]" if probe["ok"] else "[red]FAIL[/red]"
        why = probe["reason"] or "parseable tool call"
        console.print(f"{mark} {label}: {why} ({probe['latency_s']:.2f}s)")
    raise typer.Exit(0 if result["ok"] else 1)


@swe_app.command("run")
def swe_run(
    model: str = typer.Argument(...),
    tier: int = typer.Option(
        1, "--tier", min=1, max=3, help="1=evalplus 2=aider polyglot 3=SWE-bench"
    ),
    limit: int | None = typer.Option(None, "--limit"),
    workers: int | None = typer.Option(None, "--workers"),
    datasets: str | None = typer.Option(None, "--datasets", help="Tier 1: humaneval,mbpp"),
    instances: str | None = typer.Option(None, "--instances", help="Tier 3: instance ids"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print harness commands only"),
    skip_check: bool = typer.Option(False, "--skip-check", help="Skip the tool-call smoke test"),
    host: str | None = HOST_OPT,
    port: int | None = PORT_OPT,
) -> None:
    """Run a SWE tier against the served model; results under state/evals/swe/."""
    settings = get_settings()
    cfg = load_eval_config(settings=settings)
    opts = SweRunOptions(
        tier=tier,
        host=host or settings.eval_host,
        port=port,
        limit=limit,
        workers=workers,
        dry_run=dry_run,
        skip_check=skip_check,
        instances=_csv(instances),
        datasets=_csv(datasets),
    )
    try:
        rec = run_swe(settings, cfg, model, opts)
    except (RuntimeError, FileNotFoundError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]{model}[/bold] tier{tier}: {rec.summary}")


@eval_app.command("report")
def report(
    suite: str = typer.Option(..., "--suite", help="sec or swe"),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model filter"),
    runs: list[Path] | None = typer.Option(None, "--run", help="Specific run directories"),
) -> None:
    """Latest run per model, plus a gpt-oss vs Terra comparison when those runs exist."""
    settings = get_settings()
    run_dirs = runs or latest_runs(settings, suite, _csv(models) or None)
    if not run_dirs:
        err.print(f"[yellow]no finished runs for suite {suite!r}[/yellow]")
        raise typer.Exit(1)
    table, md = build_report(run_dirs)
    console.print(table)
    path = write_report(settings, suite, md)
    console.print(f"[dim]wrote {path}[/dim]")
