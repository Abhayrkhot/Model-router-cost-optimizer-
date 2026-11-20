"""Provider interface: one upstream model behind a uniform async call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from modelrouter.gateway.config import Deployment
from modelrouter.gateway.schemas import ChatMessage

ErrorKind = Literal[
    "rate_limited", "overloaded", "server", "timeout", "network", "client", "refusal"
]

# Errors after which another deployment may succeed.  "client" (a malformed request) would fail
# the same way everywhere, so it is returned to the caller instead.
FALLBACK_KINDS: frozenset[str] = frozenset(
    {"rate_limited", "overloaded", "server", "timeout", "network", "refusal"}
)
RETRY_KINDS: frozenset[str] = frozenset(
    {"rate_limited", "overloaded", "server", "timeout", "network"}
)


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.kind in RETRY_KINDS

    @property
    def fallback(self) -> bool:
        return self.kind in FALLBACK_KINDS


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    finish_reason: str = "stop"
    served_model: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderRequest:
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float | None = None


class Provider(ABC):
    def __init__(self, deployment: Deployment) -> None:
        self.deployment = deployment

    @abstractmethod
    async def complete(self, req: ProviderRequest) -> Completion: ...

    async def stream(self, req: ProviderRequest) -> AsyncIterator[str | Completion]:
        """Yield text deltas, then one final `Completion` carrying usage and cost.

        Default: a single chunk from `complete`; providers with native streaming override it.
        """
        completion = await self.complete(req)
        yield completion.text
        yield completion

    def price(self, input_tokens: int, output_tokens: int) -> float:
        d = self.deployment
        return (
            input_tokens * d.price_input_per_mtok + output_tokens * d.price_output_per_mtok
        ) / 1e6

    async def aclose(self) -> None:
        return None
