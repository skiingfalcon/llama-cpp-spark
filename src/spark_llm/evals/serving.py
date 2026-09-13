"""Endpoint factories shared by every eval suite (SEC, perf, SWE).

Kept out of the SEC task module so SWE and perf do not import a task file to reach them.
"""

from __future__ import annotations

import os

from spark_llm.config import Settings
from spark_llm.evals.endpoint import Endpoint, OpenAIEndpoint
from spark_llm.registry import Registry


def endpoint_for(
    settings: Settings, registry: Registry, model: str, host: str, port: int | None
) -> Endpoint:
    """Client for a served local model; fails fast if nothing answers on the port."""
    spec = registry.get(model)
    listen = port or spec.port or settings.base_port
    ep = Endpoint(host, listen, model=model, timeout_s=settings.eval_timeout_s)
    if not ep.healthy():
        raise RuntimeError(
            f"{model} is not serving on {host}:{listen}; run: local-llm serve {model}"
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
        timeout_s=settings.eval_timeout_s,
    )
    if not ep.healthy():
        ep.close()
        raise RuntimeError(
            "OpenAI API preflight failed; check OPENAI_API_KEY, OPENAI_BASE_URL, and model access"
        )
    return ep
