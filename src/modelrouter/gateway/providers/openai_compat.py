"""Any OpenAI-compatible /v1/chat/completions server (vLLM, llama.cpp server, Ollama, ...)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from modelrouter.gateway.config import Deployment
from modelrouter.gateway.providers.base import Completion, Provider, ProviderError, ProviderRequest


def _status_error(resp: httpx.Response) -> ProviderError:
    s = resp.status_code
    body = resp.text[:300]
    if s == 429:
        return ProviderError("rate_limited", body, s)
    if s == 503:
        return ProviderError("overloaded", body, s)
    if s >= 500 or s in (408, 409):
        return ProviderError("server", body, s)
    return ProviderError("client", body, s)


class OpenAICompatProvider(Provider):
    def __init__(self, deployment: Deployment, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(deployment)
        if not deployment.base_url:
            raise ValueError(f"deployment {deployment.name} needs base_url")
        headers = {}
        if deployment.api_key_env and os.environ.get(deployment.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[deployment.api_key_env]}"
        self.client = client or httpx.AsyncClient(
            base_url=deployment.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(deployment.timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    def _body(self, req: ProviderRequest, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.deployment.upstream_model,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        return body

    async def complete(self, req: ProviderRequest) -> Completion:
        try:
            resp = await self.client.post("/chat/completions", json=self._body(req, False))
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", str(exc)) from exc
        except httpx.TransportError as exc:
            raise ProviderError("network", str(exc)) from exc
        if resp.status_code != 200:
            raise _status_error(resp)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        return Completion(
            text=choice["message"].get("content") or "",
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=self.price(tin, tout),
            finish_reason=choice.get("finish_reason") or "stop",
            served_model=data.get("model", self.deployment.upstream_model),
        )

    async def stream(self, req: ProviderRequest) -> AsyncIterator[str | Completion]:
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=self._body(req, True)
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _status_error(resp)
                usage: dict[str, Any] = {}
                finish = "stop"
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        finish = choice.get("finish_reason") or finish
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield content
        except httpx.TimeoutException as exc:
            raise ProviderError("timeout", str(exc)) from exc
        except httpx.TransportError as exc:
            raise ProviderError("network", str(exc)) from exc
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        yield Completion(
            "", tin, tout, self.price(tin, tout), finish, self.deployment.upstream_model
        )

    async def aclose(self) -> None:
        await self.client.aclose()
