"""Claude via the official Anthropic Python SDK.

- OpenAI-style messages are mapped to the Messages API: system messages become `system`,
  consecutive same-role turns are merged, and the conversation must start with a user turn.
- `temperature` is not forwarded: current API versions reject sampling parameters.
- SDK retries are disabled (`max_retries=0`); the gateway owns retries, breakers and fallback.
- A `refusal` stop reason is surfaced as a fallback-eligible error so the gateway can try the
  next deployment.  Deployments may also opt into the API's own server-side fallback.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from modelrouter.gateway.config import Deployment
from modelrouter.gateway.providers.base import Completion, Provider, ProviderError, ProviderRequest
from modelrouter.gateway.schemas import ChatMessage

SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def to_messages_api(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, str]]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    turns: list[dict[str, str]] = []
    for m in messages:
        if m.role == "system":
            continue
        if turns and turns[-1]["role"] == m.role:
            turns[-1]["content"] += "\n\n" + m.content
        else:
            turns.append({"role": m.role, "content": m.content})
    if not turns or turns[0]["role"] != "user":
        raise ProviderError("client", "conversation must start with a user message", 400)
    return system, turns


def classify(exc: Exception) -> ProviderError:
    """Map SDK exceptions (most specific first) to gateway error kinds."""
    if isinstance(exc, anthropic.RateLimitError):
        return ProviderError("rate_limited", str(exc), 429)
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderError("timeout", str(exc))
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderError("network", str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        if status == 529 or getattr(exc, "type", None) == "overloaded_error":
            return ProviderError("overloaded", exc.message, status)
        if status >= 500 or status in (408, 409):
            return ProviderError("server", exc.message, status)
        return ProviderError("client", exc.message, status)
    return ProviderError("server", repr(exc))


class AnthropicProvider(Provider):
    def __init__(self, deployment: Deployment, client: anthropic.AsyncAnthropic | None = None):
        super().__init__(deployment)
        if client is None:
            kwargs: dict[str, Any] = {"max_retries": 0, "timeout": deployment.timeout_s}
            if deployment.api_key_env:
                kwargs["api_key"] = os.environ.get(deployment.api_key_env)
            if deployment.base_url:
                kwargs["base_url"] = deployment.base_url
            client = anthropic.AsyncAnthropic(**kwargs)
        self.client = client
        self.server_side_fallback = (
            deployment.upstream_model
            in {
                "claude-opus-5-5",
                "claude-sonnet-5-5",
                "claude-fable-5-2",
            }
            and deployment.server_side_fallback
        )

    def _params(self, req: ProviderRequest) -> dict[str, Any]:
        system, turns = to_messages_api(req.messages)
        params: dict[str, Any] = {
            "model": self.deployment.upstream_model,
            "max_tokens": req.max_tokens,
            "messages": turns,
        }
        if system:
            params["system"] = system
        return params

    async def complete(self, req: ProviderRequest) -> Completion:
        params = self._params(req)
        try:
            if self.server_side_fallback:
                msg: Any = await self.client.beta.messages.create(
                    **params, betas=[SERVER_SIDE_FALLBACK_BETA], fallbacks="default"
                )
            else:
                msg = await self.client.messages.create(**params)
        except anthropic.APIError as exc:
            raise classify(exc) from exc
        if msg.stop_reason == "refusal":
            raise ProviderError("refusal", "model declined the request", 200)
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = msg.usage
        return Completion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self.price(usage.input_tokens, usage.output_tokens),
            finish_reason="length" if msg.stop_reason == "max_tokens" else "stop",
            served_model=getattr(msg, "model", self.deployment.upstream_model),
        )

    async def stream(self, req: ProviderRequest) -> AsyncIterator[str | Completion]:
        params = self._params(req)
        try:
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise classify(exc) from exc
        if final.stop_reason == "refusal":
            raise ProviderError("refusal", "model declined the request", 200)
        usage = final.usage
        yield Completion(
            text="",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self.price(usage.input_tokens, usage.output_tokens),
            finish_reason="length" if final.stop_reason == "max_tokens" else "stop",
            served_model=final.model,
        )

    async def aclose(self) -> None:
        await self.client.close()
