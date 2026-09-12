"""OpenAI-compatible client pointed at a local llama-server."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI


def make_client(port: int, host: str = "127.0.0.1") -> OpenAI:
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


def chat_once(
    port: int,
    message: str,
    *,
    system: str | None = None,
    model: str = "local",
    stream: bool = False,
) -> str | Iterator[str]:
    client = make_client(port)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    if not stream:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    def _gen() -> Iterator[str]:
        stream_resp = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _gen()
