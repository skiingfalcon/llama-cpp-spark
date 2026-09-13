"""SWE tiers: evalplus (1), aider polyglot (2), SWE-bench Verified subset via mini-swe-agent (3).

External harnesses are invoked through ``uvx`` with pinned versions and never vendored. Each
tier records the harness version and exact commands in run.json. Tiers 2 and 3 need Docker;
Tier 3's scoring harness is x86_64-first, so run it from the driver box pointed at the Spark.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spark_llm.config import Settings
from spark_llm.console import err as console
from spark_llm.evals.config import EvalConfig, SweTier
from spark_llm.evals.endpoint import Endpoint
from spark_llm.evals.runs import RunRecord, RunWriter, evals_root
from spark_llm.evals.serving import endpoint_for
from spark_llm.evals.swe.check import tool_call_check
from spark_llm.evals.swe.normalize import normalize_aider, normalize_evalplus, normalize_swebench
from spark_llm.registry import load_registry
from spark_llm.server import merge_runtime


@dataclass
class SweRunOptions:
    tier: int
    host: str = "127.0.0.1"
    port: int | None = None
    limit: int | None = None
    workers: int | None = None
    dry_run: bool = False
    skip_check: bool = False
    instances: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)


def _run(
    cmd: list[str], *, cwd: Path | None, env: dict[str, str], dry: bool, log: Path | None
) -> str:
    console.print("[cyan]$[/cyan] " + " ".join(cmd))
    if dry:
        return ""
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log:
        log.write_text(out)
    if proc.returncode != 0:
        console.print(out[-4000:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:3])} …")
    return out


def _tool_version(spec: str, cmd: list[str]) -> str:
    try:
        return (
            subprocess.check_output(
                ["uvx", "--from", spec, *cmd], text=True, stderr=subprocess.STDOUT, timeout=300
            )
            .strip()
            .splitlines()[-1]
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, IndexError) as exc:
        return f"unknown ({type(exc).__name__})"


def _endpoint_env(host: str, port: int) -> dict[str, str]:
    env = os.environ.copy()
    base = f"http://{host}:{port}/v1"
    env.update({"OPENAI_API_BASE": base, "OPENAI_BASE_URL": base, "OPENAI_API_KEY": "not-needed"})
    return env


def _workdir(settings: Settings, model: str, tier: int) -> Path:
    d = evals_root(settings) / "swe" / model / f"work-tier{tier}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _require_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError("docker is required for this tier and was not found on PATH")


# -- tier 1: evalplus ---------------------------------------------------------------------------


def run_tier1(
    settings: Settings,
    cfg: EvalConfig,
    model: str,
    ep: Endpoint,
    opts: SweRunOptions,
    writer: RunWriter,
) -> dict[str, Any]:
    tier = cfg.swe.tier1
    spec = f"evalplus=={tier.version}" if tier.version else "evalplus"
    work = _workdir(settings, model, 1)
    port = int(ep.base.rsplit(":", 1)[1])
    env = _endpoint_env(opts.host, port)
    writer.record.tools["evalplus"] = spec
    writer.flush_record()
    all_results: list[dict[str, Any]] = []
    per_dataset: dict[str, Any] = {}
    for ds in opts.datasets or tier.datasets:
        cmd = [
            "uvx",
            "--from",
            spec,
            "evalplus.evaluate",
            "--model",
            model,
            "--dataset",
            ds,
            "--backend",
            "openai",
            "--base-url",
            env["OPENAI_API_BASE"],
            "--greedy",
            "--root",
            str(work),
        ]
        out = _run(cmd, cwd=work, env=env, dry=opts.dry_run, log=work / f"{ds}.log")
        results, summary = normalize_evalplus(work, ds, out)
        for r in results:
            writer.write(r)
        all_results.extend(results)
        per_dataset[ds] = summary
    n = len(all_results)
    return {
        "n": n,
        "score": (sum(int(bool(r["correct"])) for r in all_results) / n) if n else None,
        "by_dataset": per_dataset,
        "skipped": 0,
        "commands": "see run.json tools; logs under work-tier1/",
    }


# -- tier 2: aider polyglot ----------------------------------------------------------------------


def _clone(repo: str, ref: str | None, dest: Path, dry: bool) -> str:
    if not dest.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                *(["--branch", ref] if ref and ref != "main" else []),
                repo,
                str(dest),
            ],
            cwd=None,
            env=os.environ.copy(),
            dry=dry,
            log=None,
        )
    if dry or not dest.exists():
        return ref or "?"
    return subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "--short=12", "HEAD"], text=True
    ).strip()


def run_tier2(
    settings: Settings,
    cfg: EvalConfig,
    model: str,
    ep: Endpoint,
    opts: SweRunOptions,
    writer: RunWriter,
) -> dict[str, Any]:
    tier: SweTier = cfg.swe.tier2
    if not opts.dry_run:
        _require_docker()
    tools = settings.evals_data_dir.parent / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    aider_dir = tools / "aider"
    ex_dir = aider_dir / "polyglot-benchmark"
    writer.record.tools["aider"] = _clone(tier.repo or "", tier.ref, aider_dir, opts.dry_run)
    writer.record.tools["polyglot-benchmark"] = _clone(
        tier.exercises_repo or "", tier.exercises_ref, ex_dir, opts.dry_run
    )
    writer.flush_record()
    port = int(ep.base.rsplit(":", 1)[1])
    env = _endpoint_env(opts.host, port)
    bench_root = aider_dir / "tmp.benchmarks"
    bench_root.mkdir(exist_ok=True)
    run_name = f"local-llm-{model}-{writer.dir.name}"
    workers = opts.workers or tier.workers
    # Built by aider's benchmark/docker_build.sh; host networking so 127.0.0.1 is reachable.
    if not (opts.dry_run or _image_exists("aider-benchmark")):
        _run(
            ["bash", "benchmark/docker_build.sh"],
            cwd=aider_dir,
            env=env,
            dry=False,
            log=writer.dir / "docker_build.log",
        )
    fmt = tier.edit_format or "diff"
    inner = (
        f"./benchmark/benchmark.py {run_name} --model openai/{model} --edit-format {fmt} "
        f"--threads {workers} --exercises-dir polyglot-benchmark --new"
        + (f" --num-tests {opts.limit}" if opts.limit else "")
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{aider_dir}:/aider",
        "-v",
        f"{bench_root}:/benchmarks",
        "-e",
        f"OPENAI_API_BASE={env['OPENAI_API_BASE']}",
        "-e",
        "OPENAI_API_KEY=not-needed",
        "-e",
        "AIDER_DOCKER=1",
        "-e",
        "AIDER_BENCHMARK_DIR=/benchmarks",
        "aider-benchmark",
        "bash",
        "-c",
        inner,
    ]
    _run(cmd, cwd=aider_dir, env=env, dry=opts.dry_run, log=writer.dir / "aider.log")
    if opts.dry_run:
        return {"n": 0, "score": None, "skipped": 0, "dry_run": True}
    run_dirs = sorted(bench_root.glob(f"*--{run_name}"))
    if not run_dirs:
        raise RuntimeError(f"no benchmark output under {bench_root} for {run_name}")
    results, summary = normalize_aider(run_dirs[-1])
    for r in results:
        writer.write(r)
    summary["output_dir"] = str(run_dirs[-1])
    return summary


def _image_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(["docker", "images", "-q", name], text=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return bool(out.strip())


# -- tier 3: SWE-bench Verified subset -------------------------------------------------------------


def run_tier3(
    settings: Settings,
    cfg: EvalConfig,
    model: str,
    ep: Endpoint,
    opts: SweRunOptions,
    writer: RunWriter,
) -> dict[str, Any]:
    tier = cfg.swe.tier3
    if not opts.dry_run:
        _require_docker()
    ids = opts.instances or tier.instances
    if opts.limit:
        ids = ids[: opts.limit]
    work = _workdir(settings, model, 3)
    port = int(ep.base.rsplit(":", 1)[1])
    env = _endpoint_env(opts.host, port)
    workers = opts.workers or tier.workers
    agent_spec = "mini-swe-agent"
    harness_spec = "swebench"
    if opts.dry_run:
        writer.record.tools["mini-swe-agent"] = writer.record.tools["swebench"] = "dry-run"
    else:
        writer.record.tools["mini-swe-agent"] = _tool_version(agent_spec, ["mini", "--version"])
        writer.record.tools["swebench"] = _tool_version(
            harness_spec, ["python", "-c", "import swebench;print(swebench.__version__)"]
        )
    writer.flush_record()

    override = work / "model_override.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_name": f"openai/{model}",
                    "model_kwargs": {
                        "api_base": env["OPENAI_API_BASE"],
                        "api_key": "not-needed",
                        "temperature": 0.0,
                    },
                },
                "agent": {"step_limit": tier.step_limit or 75, "cost_limit": 0.0},
            }
        )
    )
    out_dir = work / "agent_out"
    pattern = "^(" + "|".join(i.replace(".", r"\.") for i in ids) + ")$"
    agent_cmd = [
        "uvx",
        "--from",
        agent_spec,
        "mini-extra",
        "swebench",
        "--subset",
        "verified",
        "--split",
        "test",
        "--filter",
        pattern,
        "-m",
        f"openai/{model}",
        "-c",
        str(override),
        "-o",
        str(out_dir),
        "-w",
        str(workers),
    ]
    _run(agent_cmd, cwd=work, env=env, dry=opts.dry_run, log=work / "agent.log")

    preds_path = out_dir / "preds.json"
    run_id = writer.dir.name
    harness_cmd = [
        "uvx",
        "--from",
        harness_spec,
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        tier.dataset or "princeton-nlp/SWE-bench_Verified",
        "--split",
        "test",
        "--predictions_path",
        str(preds_path),
        "--max_workers",
        str(workers),
        "--run_id",
        run_id,
        "--instance_ids",
        *ids,
    ]
    _run(harness_cmd, cwd=work, env=env, dry=opts.dry_run, log=work / "harness.log")
    if opts.dry_run:
        return {"n": len(ids), "score": None, "skipped": len(ids), "dry_run": True}

    preds = json.loads(preds_path.read_text()) if preds_path.is_file() else {}
    if isinstance(preds, list):  # some versions emit a list of records
        preds = {p["instance_id"]: p for p in preds}
    reports = sorted(work.glob(f"*.{run_id}.json"))
    report = json.loads(reports[-1].read_text()) if reports else None
    trajs: dict[str, dict[str, Any]] = {}
    for tpath in out_dir.rglob("*.traj.json"):
        try:
            trajs[tpath.name.removesuffix(".traj.json")] = json.loads(tpath.read_text())
        except json.JSONDecodeError:
            continue
    results, summary = normalize_swebench(preds, report, trajs, ids)
    for r in results:
        writer.write(r)
    summary["report_path"] = str(reports[-1]) if reports else None
    return summary


# -- entry point -------------------------------------------------------------------------------


def run_swe(settings: Settings, cfg: EvalConfig, model: str, opts: SweRunOptions) -> RunRecord:
    registry = load_registry(settings=settings)
    ep = endpoint_for(settings, registry, model, opts.host, opts.port)
    rt = merge_runtime(registry.get(model), registry.defaults, settings)
    tier_cfg = {1: cfg.swe.tier1, 2: cfg.swe.tier2, 3: cfg.swe.tier3}[opts.tier]
    writer = RunWriter(
        settings,
        "swe",
        f"tier{opts.tier}",
        model,
        server=ep.server_summary(),
        runtime=rt.as_dict(),
        quality={"temperature": 0.0},
        task_config={**tier_cfg.model_dump(), "limit": opts.limit, "workers": opts.workers},
    )
    check: dict[str, Any] | None = None
    if opts.tier >= 2 and not opts.skip_check and not opts.dry_run:
        check = tool_call_check(ep)
        writer.record.task_config["tool_call_check"] = check
        writer.flush_record()
        if not check["ok"]:
            summary = {
                "n": 0,
                "score": None,
                "skipped": 0,
                "aborted": "tool_call_check failed",
                "tool_call_check": check,
            }
            writer.finish(summary)
            raise RuntimeError(
                "tool-call smoke test failed; fix the chat template / --jinja setup before running "
                f"tier {opts.tier} (details in {writer.dir}/run.json)"
            )
    runner = {1: run_tier1, 2: run_tier2, 3: run_tier3}[opts.tier]
    summary = runner(settings, cfg, model, ep, opts, writer)
    if check is not None:
        summary["tool_call_check_ok"] = check["ok"]
    rec = writer.finish(summary)
    console.print(f"[green]done[/green] {writer.dir}")
    return rec
