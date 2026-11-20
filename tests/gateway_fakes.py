"""Test doubles for the gateway: a keyword quality model and scriptable providers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import numpy as np

from modelrouter.gateway.app import Gateway
from modelrouter.gateway.config import BreakerConfig, Deployment, GatewayConfig, Tenant
from modelrouter.gateway.limits import Limits, MemoryLimits
from modelrouter.gateway.providers.base import Completion, Provider, ProviderError, ProviderRequest
from modelrouter.gateway.routing import RouterService
from modelrouter.gateway.store import DecisionLog, ResponseCache
from modelrouter.routers.base import CostModel, FloatArray, Inputs, QualityModel

MODELS = ("cheap", "mid", "strong")
KEY = "sk-test-tenant"


class KeywordQuality(QualityModel):
    """'hard' prompts: only the strong model is good; anything else: all models are good."""

    kind: ClassVar[str] = "keyword-test"

    def __init__(self) -> None:
        self.models = MODELS

    def fit(self, *args: Any) -> KeywordQuality:
        return self

    def predict(self, x: Inputs) -> FloatArray:
        hard = np.array(["hard" in t for t in x.text])
        q = np.where(hard[:, None], [0.1, 0.4, 0.95], [0.92, 0.92, 0.95])
        return q.astype(np.float64)


def cost_model() -> CostModel:
    cm = CostModel(MODELS)
    cm.coef = np.array([[1e-5, 1e-4, 1e-3], [0.0, 0.0, 0.0]])
    cm.floor = np.zeros(3)
    return cm


class FakeProvider(Provider):
    def __init__(self, deployment: Deployment, script: list[str | ProviderError] | None = None):
        super().__init__(deployment)
        self.script = list(script or [])
        self.calls = 0

    async def complete(self, req: ProviderRequest) -> Completion:
        self.calls += 1
        await asyncio.sleep(0)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, ProviderError):
                raise step
        text = f"{self.deployment.name} says hi"
        return Completion(text, 10, 5, self.deployment.price_input_per_mtok / 1e6, "stop")

    async def stream(self, req: ProviderRequest) -> AsyncIterator[str | Completion]:
        c = await self.complete(req)
        for word in c.text.split(" "):
            yield word + " "
        yield c


def make_config(**overrides: Any) -> GatewayConfig:
    deployments = [
        Deployment(
            name=f"{m}-dep",
            provider="replay",
            upstream_model=m,
            max_retries=0,
            price_input_per_mtok=p,
        )
        for m, p in zip(MODELS, (10.0, 100.0, 1000.0), strict=True)
    ]
    tenant = Tenant(
        name="acme",
        key_sha256=Tenant.hash_key(KEY),
        daily_budget_usd=1.0,
        requests_per_minute=10_000,
        default_lambda=100.0,
    )
    base: dict[str, Any] = {
        "router_dir": "unused",
        "model_map": {m: f"{m}-dep" for m in MODELS},
        "deployments": deployments,
        "tenants": [tenant],
        "breaker": BreakerConfig(failure_threshold=2, reset_after_s=30.0),
    }
    base.update(overrides)
    return GatewayConfig.model_validate(base)


def make_gateway(
    cfg: GatewayConfig | None = None,
    scripts: dict[str, list[str | ProviderError]] | None = None,
    limits: Limits | None = None,
    cache_ttl_s: int = 0,
    decision_log: Any = None,
) -> tuple[Gateway, dict[str, FakeProvider]]:
    cfg = cfg or make_config()
    scripts = scripts or {}
    providers = {d.name: FakeProvider(d, scripts.get(d.name)) for d in cfg.deployments}
    router = RouterService(KeywordQuality(), cost_model(), cfg.model_map, embedder=None)
    gw = Gateway(
        cfg,
        router,
        dict(providers),
        limits or MemoryLimits(),
        ResponseCache(cache_ttl_s),
        DecisionLog(decision_log),
    )
    return gw, providers


def chat(content: str, **extra: Any) -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": content}], **extra}


AUTH = {"Authorization": f"Bearer {KEY}"}
