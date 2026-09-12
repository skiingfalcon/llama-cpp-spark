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
from urllib.parse import quote

import httpx


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""  # hidden chain-of-thought (``reasoning_content``), when the server emits it
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ttft_s: float | None = None
    total_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    status: int = 200
    error: str | None = None
    retry_after_s: float | None = None  # from a 429/503 ``Retry-After`` header
    derive_decode_tps: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200

    @property
    def truncated(self) -> bool:
        """Hit ``max_tokens`` before finishing; the answer is missing or cut short."""
        return self.finish_reason == "length"

    @property
    def prompt_tps(self) -> float | None:
        v = self.timings.get("prompt_per_second")
        return float(v) if v is not None else None

    @property
    def decode_tps(self) -> float | None:
        v = self.timings.get("predicted_per_second")
        if v is not None:
            return float(v)
        if (
            self.derive_decode_tps
            and self.completion_tokens
            and self.ttft_s is not None
            and self.total_s > self.ttft_s
        ):
            return self.completion_tokens / (self.total_s - self.ttft_s)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "reasoning_chars": len(self.reasoning),
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
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
        self.chat_url = f"{self.base}/v1/chat/completions"

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
                time.sleep(result.retry_after_s or delay)
                delay = min(delay * 2, 30.0)
                continue
            self._fill_reasoning_tokens(result)
            return result
        return result

    def _fill_reasoning_tokens(self, result: ChatResult) -> None:
        """llama-server reports no reasoning token count; measure the text it streamed."""
        if result.ok and result.reasoning and result.reasoning_tokens is None:
            try:
                result.reasoning_tokens = self.count_tokens(result.reasoning)
            except httpx.HTTPError:
                pass

    def _once(self, body: dict[str, Any]) -> ChatResult:
        t0 = time.perf_counter()
        try:
            r = self._client.post(self.chat_url, json=body)
        except httpx.HTTPError as exc:
            return ChatResult(status=0, error=str(exc), total_s=time.perf_counter() - t0)
        total = time.perf_counter() - t0
        if r.status_code != 200:
            return ChatResult(
                status=r.status_code,
                error=r.text[:500],
                total_s=total,
                retry_after_s=_retry_after(r.headers),
            )
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        result = ChatResult(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason"),
            tool_calls=list(msg.get("tool_calls") or []),
            ttft_s=None,
            total_s=total,
            timings=data.get("timings") or {},
        )
        _apply_usage(result, usage)
        return result

    def _stream(self, body: dict[str, Any]) -> ChatResult:
        t0 = time.perf_counter()
        res = ChatResult()
        parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        try:
            with self._client.stream("POST", self.chat_url, json=body) as r:
                if r.status_code != 200:
                    r.read()
                    return ChatResult(
                        status=r.status_code,
                        error=r.text[:500],
                        total_s=time.perf_counter() - t0,
                        retry_after_s=_retry_after(r.headers),
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
                        _apply_usage(res, chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        thought = delta.get("reasoning_content")
                        if thought:
                            if res.ttft_s is None:
                                res.ttft_s = time.perf_counter() - t0
                            reasoning_parts.append(thought)
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
        res.reasoning = "".join(reasoning_parts)
        res.tool_calls = [tool_calls[i] for i in sorted(tool_calls)]
        if res.timings.get("prompt_n") and res.prompt_tokens is None:
            res.prompt_tokens = int(res.timings["prompt_n"])
        if res.timings.get("predicted_n") and res.completion_tokens is None:
            res.completion_tokens = int(res.timings["predicted_n"])
        return res

    def close(self) -> None:
        self._client.close()


class OpenAIEndpoint(Endpoint):
    """OpenAI Chat Completions endpoint with local tokenization for context budgeting."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        context_window: int = 128000,
        timeout_s: float = 1800.0,
        retries: int = 8,
        transport: httpx.BaseTransport | None = None,
        retry_delay_s: float = 1.0,
        max_retry_delay_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for --provider openai")
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_s, connect=10.0)
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.max_retry_delay_s = max_retry_delay_s
        self._context_window = context_window
        self._client = httpx.Client(
            timeout=self.timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.chat_url = f"{self.base}/chat/completions"
        self._encoding = _openai_encoding(model)

    def healthy(self) -> bool:
        try:
            model = quote(self.model, safe="")
            return self._client.get(f"{self.base}/models/{model}", timeout=10.0).status_code == 200
        except httpx.HTTPError:
            return False

    def server_summary(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "api": "chat_completions",
            "model": self.model,
            "base_url": self.base,
            "n_ctx_per_slot": self._context_window,
        }

    def n_ctx(self) -> int:
        return self._context_window

    def tokenize(self, text: str) -> list[int]:
        return list(self._encoding.encode(text))

    def detokenize(self, tokens: list[int]) -> str:
        return str(self._encoding.decode(tokens))

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
        # Frontier reasoning models do not consistently accept temperature or seed.
        # Omitting both uses the provider default and is recorded in run metadata.
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stream:
            body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)

        delay = self.retry_delay_s
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(self.retries + 1):
            result = self._stream(body) if stream else self._once(body)
            result.derive_decode_tps = False
            if result.status in retryable and attempt < self.retries:
                # Providers say how long to back off on 429; honour it, else exponential.
                wait = result.retry_after_s if result.retry_after_s is not None else delay
                time.sleep(min(wait, self.max_retry_delay_s))
                delay = min(delay * 2, self.max_retry_delay_s)
                continue
            return result
        return result


def _empty_tool_call() -> dict[str, Any]:
    return {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}


def _retry_after(headers: httpx.Headers) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds form only)."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _apply_usage(result: ChatResult, usage: dict[str, Any]) -> None:
    result.prompt_tokens = usage.get("prompt_tokens")
    result.completion_tokens = usage.get("completion_tokens")
    result.cached_prompt_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")


def _openai_encoding(model: str) -> Any:
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # o200k_base is the tokenizer family used by current OpenAI frontier models.
        return tiktoken.get_encoding("o200k_base")


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
