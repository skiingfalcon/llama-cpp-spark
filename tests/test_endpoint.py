"""Streaming/non-streaming parsing and retry behaviour of the eval Endpoint (mock transport)."""

from __future__ import annotations

import json

import httpx

from spark_llm.evals.endpoint import Endpoint, OpenAIEndpoint, percentile, run_parallel


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def test_stream_parses_content_usage_timings_and_ttft() -> None:
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}, "index": 0}]},
        {"choices": [{"delta": {"content": "109,"}, "index": 0}]},
        {"choices": [{"delta": {"content": "417"}, "index": 0, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 21000, "completion_tokens": 3},
            "timings": {
                "prompt_n": 21000,
                "prompt_per_second": 4200.0,
                "predicted_per_second": 55.0,
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True and body["cache_prompt"] is True
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200, content=_sse(chunks), headers={"content-type": "text/event-stream"}
        )

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler))
    r = ep.chat([{"role": "user", "content": "q"}], temperature=0.0, seed=1, max_tokens=8)
    assert r.ok and r.content == "109,417"
    assert r.finish_reason == "stop"
    assert r.ttft_s is not None and r.ttft_s >= 0
    assert r.prompt_tokens == 21000 and r.completion_tokens == 3
    assert r.prompt_tps == 4200.0 and r.decode_tps == 55.0


def test_stream_collects_reasoning_and_flags_truncation() -> None:
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Let me find"}, "index": 0}]},
        {"choices": [{"delta": {"reasoning_content": " the table"}, "index": 0}]},
        {"choices": [{"delta": {"content": "42"}, "index": 0, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            n = len(json.loads(request.content)["content"].split())
            return httpx.Response(200, json={"tokens": list(range(n))})
        return httpx.Response(200, content=_sse(chunks))

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler))
    r = ep.chat([{"role": "user", "content": "q"}], max_tokens=8)
    assert r.content == "42" and r.reasoning == "Let me find the table"
    assert r.ttft_s is not None  # first token counted from the first reasoning delta
    assert r.truncated and r.finish_reason == "length"
    assert r.reasoning_tokens == 5  # measured via /tokenize since llama-server gives no count
    d = r.as_dict()
    assert d["truncated"] is True and d["reasoning_chars"] == len("Let me find the table")


def test_stream_without_reasoning_leaves_reasoning_tokens_unset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/tokenize", "no reasoning text, no tokenize call"
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": "1"}, "finish_reason": "stop"}]}]),
        )

    r = Endpoint("h", 1, transport=httpx.MockTransport(handler)).chat(
        [{"role": "user", "content": "q"}]
    )
    assert r.content == "1" and r.reasoning == "" and r.reasoning_tokens is None
    assert not r.truncated and r.as_dict()["reasoning_chars"] == 0


def test_local_chat_forwards_reasoning_effort_via_chat_template_kwargs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": "1"}, "finish_reason": "stop"}]}]),
        )

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler))
    r = ep.chat(
        [{"role": "user", "content": "q"}],
        extra={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    assert r.ok


def test_stream_assembles_tool_calls() -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "function": {"name": "read_", "arguments": ""}}
                        ]
                    },
                    "index": 0,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "file", "arguments": '{"path": '}}
                        ]
                    },
                    "index": 0,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]},
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    ep = Endpoint(
        "h", 1, transport=httpx.MockTransport(lambda req: httpx.Response(200, content=_sse(chunks)))
    )
    r = ep.chat(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert r.finish_reason == "tool_calls"
    assert r.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }
    ]


def test_non_stream_and_error_paths() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="no slot available")
        if json.loads(request.content).get("max_tokens") == 999:
            return httpx.Response(400, text="the request exceeds the available context size")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                "timings": {"prompt_per_second": 100.0, "predicted_per_second": 10.0},
            },
        )

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler), retry_delay_s=0.0)
    r = ep.chat([{"role": "user", "content": "q"}], stream=False)
    assert r.ok and r.content == "42" and calls["n"] == 2  # one 503 retry
    assert r.prompt_tokens == 5 and r.decode_tps == 10.0
    bad = ep.chat([{"role": "user", "content": "q"}], stream=False, max_tokens=999)
    assert not bad.ok and bad.status == 400 and "context" in (bad.error or "")


def test_props_tokenize_detokenize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 65536},
                    "total_slots": 2,
                    "model_path": "/m.gguf",
                    "chat_template": "x",
                },
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": "abc"})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler))
    assert ep.healthy()
    assert ep.n_ctx() == 65536
    assert ep.server_summary()["total_slots"] == 2
    assert ep.count_tokens("abc") == 3
    assert ep.detokenize([1, 2, 3]) == "abc"


def test_openai_endpoint_uses_portable_request_and_tracks_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path == "/v1/models/frontier-test":
            return httpx.Response(200, json={"data": []})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "frontier-test"
        assert body["max_completion_tokens"] == 32
        assert "temperature" not in body and "seed" not in body
        assert "cache_prompt" not in body and "timings_per_token" not in body
        return httpx.Response(
            200,
            content=_sse(
                [
                    {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 7,
                            "prompt_tokens_details": {"cached_tokens": 80},
                            "completion_tokens_details": {"reasoning_tokens": 5},
                        },
                    },
                ]
            ),
        )

    ep = OpenAIEndpoint(
        "test-key",
        "frontier-test",
        context_window=200000,
        transport=httpx.MockTransport(handler),
    )
    assert ep.healthy()
    assert ep.n_ctx() == 200000
    assert ep.server_summary()["provider"] == "openai"
    assert ep.detokenize(ep.tokenize("annual report")) == "annual report"
    r = ep.chat(
        [{"role": "user", "content": "q"}],
        temperature=0.0,
        seed=42,
        max_tokens=32,
    )
    assert r.ok and r.content == "42"
    assert r.prompt_tokens == 100 and r.completion_tokens == 7
    assert r.cached_prompt_tokens == 80 and r.reasoning_tokens == 5
    assert r.decode_tps is None


def test_openai_endpoint_passes_reasoning_effort_and_honours_retry_after() -> None:
    calls: list[dict] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/models/"):
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": "7"}, "finish_reason": "stop"}]}]),
        )

    ep = OpenAIEndpoint(
        "k", "frontier-test", transport=httpx.MockTransport(handler), retry_delay_s=0.0
    )
    r = ep.chat(
        [{"role": "user", "content": "q"}], max_tokens=16, extra={"reasoning_effort": "low"}
    )
    assert r.ok and r.content == "7"
    assert len(calls) == 2 and calls[0]["reasoning_effort"] == "low"
    assert calls[1]["reasoning_effort"] == "low" and "chat_template_kwargs" not in calls[1]
    assert ep.retries == 8 and ep.max_retry_delay_s == 60.0
    del slept


def test_openai_endpoint_omits_reasoning_effort_when_not_requested() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/models/"):
            return httpx.Response(200, json={"data": []})
        assert "reasoning_effort" not in json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse([{"choices": [{"delta": {"content": "7"}, "finish_reason": "stop"}]}]),
        )

    ep = OpenAIEndpoint("k", "frontier-test", transport=httpx.MockTransport(handler))
    assert ep.chat([{"role": "user", "content": "q"}], max_tokens=16).ok


def test_retry_after_error_result_carries_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="nope", headers={"Retry-After": "2.5"})

    ep = Endpoint("h", 1, transport=httpx.MockTransport(handler), retries=0)
    r = ep.chat([{"role": "user", "content": "q"}])
    assert r.status == 429 and r.retry_after_s == 2.5


def test_helpers() -> None:
    assert run_parallel([1, 2, 3], lambda x: x * 2, parallel=2) == [2, 4, 6]
    assert run_parallel([1, 2, 3], lambda x: x * 2, parallel=1) == [2, 4, 6]
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([], 0.5) is None
