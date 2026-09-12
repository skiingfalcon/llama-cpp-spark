"""OpenAI-compatible client for a running llama-server with per-request timing.

Uses httpx directly (not the openai SDK) so we can read llama-server's extra fields
(``timings``, ``usage`` on streams) and time the first token ourselves.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ChatResult:
    content: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ttft_s: float | None = None
    total_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    status: int = 200
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200

    @property
    def prompt_tps(self) -> float | None:
        v = self.timings.get("prompt_per_second")
        return float(v) if v is not None else None

    @property
    def decode_tps(self) -> float | None:
        v = self.timings.get("predicted_per_second")
        if v is not None:
            return float(v)
        if self.completion_tokens and self.ttft_s is not None and self.total_s > self.ttft_s:
            return self.completion_tokens / (self.total_s - self.ttft_s)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_tps": self.prompt_tps,
            "decode_tps": self.decode_tps,
            "status": self.status,
            "error": self.error,
        }


class Endpoint:
    def __init__(
        self,
        host: str,
        port: int,
        model: str = "local",
        *,
        timeout_s: float = 1800.0,
        retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_delay_s: float = 1.0,
    ) -> None:
        self.base = f"http://{host}:{port}"
        self.model = model
        self.timeout = httpx.Timeout(timeout_s, connect=10.0)
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self._client = httpx.Client(timeout=self.timeout, transport=transport)

    # -- introspection -------------------------------------------------------------------
    def healthy(self) -> bool:
        try:
            return self._client.get(f"{self.base}/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def props(self) -> dict[str, Any]:
        r = self._client.get(f"{self.base}/props")
        r.raise_for_status()
        return r.json()

    def server_summary(self) -> dict[str, Any]:
        """Compact /props subset recorded in every run."""
        try:
            p = self.props()
        except httpx.HTTPError as exc:
            return {"error": str(exc)}
        gen = p.get("default_generation_settings", {})
        return {
            "n_ctx_per_slot": gen.get("n_ctx"),
            "total_slots": p.get("total_slots"),
            "model_path": p.get("model_path"),
            "chat_template_sha": _short_sha(p.get("chat_template", "")),
            "build_info": p.get("build_info"),
        }

    def n_ctx(self) -> int | None:
        return self.server_summary().get("n_ctx_per_slot")

    def tokenize(self, text: str) -> list[int]:
        r = self._client.post(f"{self.base}/tokenize", json={"content": text})
        r.raise_for_status()
        return list(r.json().get("tokens", []))

    def detokenize(self, tokens: list[int]) -> str:
        r = self._client.post(f"{self.base}/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        return str(r.json().get("content", ""))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # -- chat ------------------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        stream: bool = True,
        cache_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        # llama-server extensions
        body["cache_prompt"] = cache_prompt
        if stream:
            body["stream_options"] = {"include_usage": True}
            body["timings_per_token"] = True
        if extra:
            body.update(extra)

        delay = self.retry_delay_s
        for attempt in range(self.retries + 1):
            result = self._stream(body) if stream else self._once(body)
            if result.status == 503 and attempt < self.retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            return result
        return result

    def _once(self, body: dict[str, Any]) -> ChatResult:
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{self.base}/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return ChatResult(status=0, error=str(exc), total_s=time.perf_counter() - t0)
        total = time.perf_counter() - t0
        if r.status_code != 200:
            return ChatResult(status=r.status_code, error=r.text[:500], total_s=total)
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            content=msg.get("content") or "",
            finish_reason=choice.get("finish_reason"),
            tool_calls=list(msg.get("tool_calls") or []),
            ttft_s=None,
            total_s=total,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            timings=data.get("timings") or {},
        )

    def _stream(self, body: dict[str, Any]) -> ChatResult:
        t0 = time.perf_counter()
        res = ChatResult()
        parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        try:
            with self._client.stream("POST", f"{self.base}/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    return ChatResult(
                        status=r.status_code,
                        error=r.text[:500],
                        total_s=time.perf_counter() - t0,
                    )
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("timings"):
                        res.timings = chunk["timings"]
                    if chunk.get("usage"):
                        res.prompt_tokens = chunk["usage"].get("prompt_tokens")
                        res.completion_tokens = chunk["usage"].get("completion_tokens")
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            if res.ttft_s is None:
                                res.ttft_s = time.perf_counter() - t0
                            parts.append(text)
                        for tc in delta.get("tool_calls") or []:
                            if res.ttft_s is None:
                                res.ttft_s = time.perf_counter() - t0
                            idx = int(tc.get("index", 0))
                            slot = tool_calls.setdefault(idx, _empty_tool_call())
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
                        if choice.get("finish_reason"):
                            res.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as exc:
            res.status = 0
            res.error = str(exc)
        res.total_s = time.perf_counter() - t0
        res.content = "".join(parts)
        res.tool_calls = [tool_calls[i] for i in sorted(tool_calls)]
        if res.timings.get("prompt_n") and res.prompt_tokens is None:
            res.prompt_tokens = int(res.timings["prompt_n"])
        if res.timings.get("predicted_n") and res.completion_tokens is None:
            res.completion_tokens = int(res.timings["predicted_n"])
        return res

    def close(self) -> None:
        self._client.close()


def _empty_tool_call() -> dict[str, Any]:
    return {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}


def _short_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:12] if text else ""


def run_parallel[T, R](items: Iterable[T], fn: Callable[[T], R], parallel: int = 1) -> list[R]:
    """Apply fn to items with bounded concurrency, preserving order."""
    seq = list(items)
    if parallel <= 1:
        return [fn(x) for x in seq]
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(fn, seq))


def percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
