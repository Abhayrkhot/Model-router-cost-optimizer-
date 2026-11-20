"""The subset of the OpenAI Chat Completions schema the gateway accepts and returns."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class RouterOptions(BaseModel):
    """Per-request routing controls (an extension field; OpenAI clients can omit it)."""

    # Quality-per-dollar trade-off; higher = cheaper.  Defaults to the tenant's value.
    lambda_: float | None = Field(default=None, alias="lambda", ge=0)
    # Hard cap on the predicted cost of this request.
    max_cost_usd: float | None = Field(default=None, gt=0)
    # Restrict routing to these router models.
    allowed_models: list[str] | None = None

    model_config = {"populate_by_name": True}


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=65536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False
    user: str | None = None
    router: RouterOptions | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class RoutingInfo(BaseModel):
    """Returned alongside the completion so callers can see and audit the decision."""

    router_model: str | None
    deployment: str
    predicted_quality: float | None = None
    predicted_cost_usd: float | None = None
    cost_usd: float
    lambda_: float | None = Field(default=None, serialization_alias="lambda")
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    cache_hit: bool = False
    router_ms: float = 0.0


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    routing: RoutingInfo


def router_text_from_messages(messages: list[ChatMessage]) -> str:
    """Same router text the offline data uses: head of the first turn + the last user turn."""
    from modelrouter.data.routerbench import router_text

    turns = [m.content for m in messages if m.role != "assistant"]
    return router_text(turns) if turns else ""


def prompt_chars(messages: list[ChatMessage]) -> int:
    return sum(len(m.content) for m in messages)
