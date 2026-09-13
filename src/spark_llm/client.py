"""Minimal chat client for ``local-llm chat`` against a running OpenAI-compatible server.

httpx only (same as the eval Endpoint); the ``openai`` SDK is not a dependency of this project.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx


def _messages(message: str, system: str | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": message})
    return msgs


def chat_once(
    port: int,
    message: str,
    *,
    system: str | None = None,
    model: str = "local",
    host: str = "127.0.0.1",
    timeout_s: float = 600.0,
) -> str:
    """One non-streaming completion; returns the assistant text."""
    url = f"http://{host}:{port}/v1/chat/completions"
    body = {"model": model, "messages": _messages(message, system), "stream": False}
    r = httpx.post(url, json=body, timeout=timeout_s)
    r.raise_for_status()
    choice = (r.json().get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""


def chat_stream(
    port: int,
    message: str,
    *,
    system: str | None = None,
    model: str = "local",
    host: str = "127.0.0.1",
    timeout_s: float = 600.0,
) -> Iterator[str]:
    """Streaming completion; yields visible content deltas as they arrive."""
    url = f"http://{host}:{port}/v1/chat/completions"
    body = {"model": model, "messages": _messages(message, system), "stream": True}
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0)) as client:
        with client.stream("POST", url, json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text
