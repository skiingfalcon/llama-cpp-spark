"""Fail-fast tool-calling smoke test.

Agentic harnesses (aider, mini-swe-agent) depend on the served model emitting parseable
tool calls through llama-server's ``--jinja`` chat template. Run this before spending hours
on Tier 2/3; a model that cannot pass it will score near zero for reasons unrelated to its
coding ability.
"""

from __future__ import annotations

import json
from typing import Any

from spark_llm.evals.config import load_prompt
from spark_llm.evals.endpoint import Endpoint

READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the repository and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Repository-relative path"}},
            "required": ["path"],
        },
    },
}


def _judge_call(tool_calls: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not tool_calls:
        return False, "no tool_calls in response"
    fn = tool_calls[0].get("function") or {}
    if fn.get("name") != "read_file":
        return False, f"wrong tool name {fn.get('name')!r}"
    try:
        args = json.loads(fn.get("arguments") or "")
    except json.JSONDecodeError as exc:
        return False, f"arguments not JSON: {exc}"
    if args.get("path") != "src/app/main.py":
        return False, f"unexpected arguments {args!r}"
    return True, None


def tool_call_check(ep: Endpoint) -> dict[str, Any]:
    """Probe both non-streaming and streaming tool calls; returns a summary dict."""
    prompt = load_prompt("swe_tool_check")
    probes: dict[str, dict[str, Any]] = {}
    for label, stream in (("non_stream", False), ("stream", True)):
        r = ep.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            stream=stream,
            tools=[READ_FILE_TOOL],
            tool_choice="auto",
        )
        ok, why = (False, f"http {r.status}: {r.error}") if not r.ok else _judge_call(r.tool_calls)
        probes[label] = {
            "ok": ok,
            "reason": why,
            "latency_s": r.total_s,
            "finish_reason": r.finish_reason,
            "tool_calls": r.tool_calls,
            "content": r.content[:200],
        }
    return {"ok": all(p["ok"] for p in probes.values()), "probes": probes}
