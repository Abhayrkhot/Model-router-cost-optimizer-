"""OpenAI-compatible routing gateway.

Request path for `model: "auto"`:

  auth (bearer key -> tenant) -> rate limit -> cache lookup -> router decision (micro-batched
  embedding + quality/cost prediction, ranked by q̂ − λ·ĉ) -> for each candidate in rank order:
  skip if its breaker is open or its predicted cost doesn't fit the tenant's remaining budget;
  reserve the predicted cost; call with retries; on a fallback-eligible failure move to the next
  candidate -> settle the actual cost -> metrics + decision log.

A named model (a deployment, or a router model name) skips routing and fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from modelrouter.gateway.config import Deployment, GatewayConfig, Tenant
from modelrouter.gateway.limits import Limits, MemoryLimits, RedisLimits
from modelrouter.gateway.providers.base import (
    RETRY_KINDS,
    Completion,
    Provider,
    ProviderError,
    ProviderRequest,
)
from modelrouter.gateway.resilience import CircuitBreaker, State, backoff_s
from modelrouter.gateway.routing import Candidate, Decision, RouterService
from modelrouter.gateway.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    RoutingInfo,
    Usage,
    prompt_chars,
    router_text_from_messages,
)
from modelrouter.gateway.store import DecisionLog, ResponseCache, cache_key
from modelrouter.gateway.telemetry import Metrics

log = logging.getLogger("modelrouter.gateway")


class GatewayError(Exception):
    def __init__(self, status: int, kind: str, message: str, extra: dict[str, Any] | None = None):
        super().__init__(message)
        self.status, self.kind, self.message, self.extra = status, kind, message, extra or {}

    def response(self, request_id: str) -> JSONResponse:
        body = {"error": {"type": self.kind, "message": self.message, **self.extra}}
        return JSONResponse(body, status_code=self.status, headers={"x-request-id": request_id})


@dataclass
class Attempt:
    deployment: str
    outcome: str  # ok | error:<kind> | skipped:breaker | skipped:budget
    ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"deployment": self.deployment, "outcome": self.outcome, "ms": round(self.ms, 1)}


@dataclass
class Served:
    completion: Completion
    candidate: Candidate
    attempts: list[Attempt] = field(default_factory=list)


def build_providers(cfg: GatewayConfig) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    store = None
    for d in cfg.deployments:
        if d.provider == "replay":
            from modelrouter.gateway.providers.replay import ReplayProvider, ReplayStore

            if cfg.replay_fixture is None:
                raise ValueError("replay deployments need replay_fixture")
            store = store or ReplayStore.load(cfg.replay_fixture)
            providers[d.name] = ReplayProvider(d, store)
        elif d.provider == "anthropic":
            from modelrouter.gateway.providers.anthropic_provider import AnthropicProvider

            providers[d.name] = AnthropicProvider(d)
        else:
            from modelrouter.gateway.providers.openai_compat import OpenAICompatProvider

            providers[d.name] = OpenAICompatProvider(d)
    return providers


class Gateway:
    def __init__(
        self,
        cfg: GatewayConfig,
        router: RouterService,
        providers: dict[str, Provider],
        limits: Limits,
        cache: ResponseCache,
        decisions: DecisionLog,
        metrics: Metrics | None = None,
    ) -> None:
        self.cfg = cfg
        self.router = router
        self.providers = providers
        self.limits = limits
        self.cache = cache
        self.decisions = decisions
        self.metrics = metrics or Metrics()
        self.breakers = {d.name: CircuitBreaker(cfg.breaker) for d in cfg.deployments}
        self.tenants = {t.key_sha256: t for t in cfg.tenants}
        self.deployments = {d.name: d for d in cfg.deployments}
        self.reverse_map = {v: k for k, v in cfg.model_map.items()}

    # --- helpers -----------------------------------------------------------------------------

    def authenticate(self, request: Request) -> Tenant:
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        tenant = self.tenants.get(Tenant.hash_key(key)) if key else None
        if tenant is None:
            raise GatewayError(401, "invalid_api_key", "missing or unknown API key")
        return tenant

    def _sync_breaker_gauge(self, name: str) -> None:
        state = self.breakers[name].state
        self.metrics.breaker_state.labels(name).set(1.0 if state is State.OPEN else 0.0)

    async def _candidates(self, tenant: Tenant, req: ChatRequest) -> Decision:
        opts = req.router
        if req.model != "auto":
            name = self.cfg.model_map.get(req.model, req.model)
            if name not in self.deployments:
                raise GatewayError(400, "model_not_found", f"unknown model {req.model!r}")
            # Score it anyway so budgets can use a predicted cost.
            router_model = self.reverse_map.get(name)
            decision = await self.router.decide(
                router_text_from_messages(req.messages),
                prompt_chars(req.messages),
                0.0,
                allowed=[router_model] if router_model else [],
            )
            if decision.candidates:
                return Decision(decision.candidates[:1], 0.0, decision.router_ms)
            return Decision([Candidate(req.model, name, float("nan"), 0.0, 0.0)], 0.0, 0.0)
        lam = tenant.default_lambda if opts is None or opts.lambda_ is None else opts.lambda_
        decision = await self.router.decide(
            router_text_from_messages(req.messages),
            prompt_chars(req.messages),
            lam,
            allowed=opts.allowed_models if opts else None,
            max_cost_usd=opts.max_cost_usd if opts else None,
        )
        if not decision.candidates:
            raise GatewayError(400, "no_candidate", "no model satisfies the routing constraints")
        return decision

    async def _call(
        self, provider: Provider, d: Deployment, preq: ProviderRequest, deadline: float
    ) -> Completion:
        last: ProviderError | None = None
        for attempt in range(1 + d.max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("timeout", "request deadline exceeded")
            try:
                async with asyncio.timeout(min(d.timeout_s, remaining)):
                    return await provider.complete(preq)
            except TimeoutError:
                last = ProviderError("timeout", f"{d.name} timed out")
            except ProviderError as err:
                last = err
            if not last.retryable or attempt == d.max_retries:
                break
            await asyncio.sleep(min(backoff_s(attempt), max(0.0, deadline - time.monotonic())))
        assert last is not None
        raise last

    def _provider_request(self, req: ChatRequest, d: Deployment) -> ProviderRequest:
        return ProviderRequest(
            req.messages, req.max_tokens or d.max_tokens_default, req.temperature
        )

    async def _reserve(self, tenant: Tenant, cand: Candidate) -> bool:
        return await self.limits.reserve(tenant.name, cand.cost_usd, tenant.daily_budget_usd)

    # --- non-streaming -------------------------------------------------------------------

    async def serve(self, tenant: Tenant, req: ChatRequest, decision: Decision) -> Served:
        deadline = time.monotonic() + self.cfg.request_deadline_s
        attempts: list[Attempt] = []
        tried = 0
        last_err: ProviderError | None = None
        budget_blocked = False
        for cand in decision.candidates:
            if tried >= self.cfg.max_attempts:
                break
            breaker = self.breakers[cand.deployment]
            if not breaker.allow():
                attempts.append(Attempt(cand.deployment, "skipped:breaker"))
                continue
            if not await self._reserve(tenant, cand):
                attempts.append(Attempt(cand.deployment, "skipped:budget"))
                budget_blocked = True
                if breaker.state is State.HALF_OPEN:
                    breaker.probes_in_flight = max(0, breaker.probes_in_flight - 1)
                continue
            tried += 1
            d = self.deployments[cand.deployment]
            start = time.perf_counter()
            try:
                completion = await self._call(
                    self.providers[d.name], d, self._provider_request(req, d), deadline
                )
            except ProviderError as err:
                ms = (time.perf_counter() - start) * 1000
                await self.limits.settle(tenant.name, cand.cost_usd, 0.0)
                attempts.append(Attempt(d.name, f"error:{err.kind}", ms))
                self.metrics.attempt_errors.labels(d.name, err.kind).inc()
                if err.kind in RETRY_KINDS:
                    breaker.record_failure()
                elif breaker.state is State.HALF_OPEN:
                    breaker.probes_in_flight = max(0, breaker.probes_in_flight - 1)
                self._sync_breaker_gauge(d.name)
                last_err = err
                if not err.fallback:
                    raise GatewayError(
                        err.status or 400,
                        "upstream_rejected",
                        str(err),
                        {"attempts": [a.as_dict() for a in attempts]},
                    ) from err
                continue
            ms = (time.perf_counter() - start) * 1000
            breaker.record_success()
            self._sync_breaker_gauge(d.name)
            await self.limits.settle(tenant.name, cand.cost_usd, completion.cost_usd)
            self.metrics.upstream_latency.labels(d.name).observe(ms / 1000)
            attempts.append(Attempt(d.name, "ok", ms))
            return Served(completion, cand, attempts)
        detail = {"attempts": [a.as_dict() for a in attempts]}
        if tried == 0 and budget_blocked:
            raise GatewayError(402, "budget_exceeded", "daily budget exhausted", detail)
        msg = f"all candidates failed (last: {last_err})" if last_err else "no available upstream"
        raise GatewayError(503, "upstream_unavailable", msg, detail)

    def _record(
        self,
        request_id: str,
        tenant: Tenant,
        decision: Decision,
        served: Served | None,
        status: str,
        started: float,
        cache_hit: bool = False,
    ) -> None:
        total_s = time.perf_counter() - started
        self.metrics.requests.labels(tenant.name, status).inc()
        self.metrics.request_latency.observe(total_s)
        row: dict[str, Any] = {
            "ts": time.time(),
            "request_id": request_id,
            "tenant": tenant.name,
            "status": status,
            "lambda": decision.lam,
            "router_ms": round(decision.router_ms, 3),
            "total_ms": round(total_s * 1000, 3),
            "cache_hit": cache_hit,
            "ranked": [
                {"model": c.router_model, "q": round(c.quality, 4), "c": c.cost_usd}
                for c in decision.candidates[:5]
            ],
        }
        if served is not None:
            c, cand = served.completion, served.candidate
            self.metrics.routed.labels(cand.deployment).inc()
            self.metrics.cost.labels(tenant.name, cand.deployment).inc(c.cost_usd)
            self.metrics.tokens.labels(cand.deployment, "input").inc(c.input_tokens)
            self.metrics.tokens.labels(cand.deployment, "output").inc(c.output_tokens)
            if served.attempts and served.attempts[0].deployment != cand.deployment:
                self.metrics.fallbacks.inc()
            row |= {
                "router_model": cand.router_model,
                "deployment": cand.deployment,
                "cost_usd": c.cost_usd,
                "attempts": [a.as_dict() for a in served.attempts],
            }
        self.decisions.write(row)

    async def chat(self, request: Request, req: ChatRequest) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            tenant = self.authenticate(request)
        except GatewayError as err:
            return err.response(request_id)
        if not await self.limits.allow_request(tenant.name, tenant.requests_per_minute):
            self.metrics.requests.labels(tenant.name, "rate_limited").inc()
            return GatewayError(429, "rate_limited", "too many requests").response(request_id)

        cacheable = self.cache.enabled and not req.stream and not req.temperature
        key = ""
        if cacheable:
            key = cache_key(
                {
                    # Per tenant: a shared entry would reveal (via $0 cost and latency) that
                    # another tenant sent the same prompt.
                    "tenant": tenant.name,
                    "tenant_lambda": tenant.default_lambda,
                    **req.model_dump(exclude={"user", "stream"}, by_alias=True),
                }
            )
            hit = await self.cache.get(key)
            self.metrics.cache.labels("hit" if hit else "miss").inc()
            if hit is not None:
                hit["routing"]["cache_hit"] = True
                hit["routing"]["cost_usd"] = 0.0
                self._record(request_id, tenant, Decision([], 0.0, 0.0), None, "ok", started, True)
                return JSONResponse(hit, headers={"x-request-id": request_id})

        try:
            decision = await self._candidates(tenant, req)
            self.metrics.router_latency.observe(decision.router_ms / 1000)
            if req.stream:
                return await self.chat_stream(request_id, tenant, req, decision, started)
            served = await self.serve(tenant, req, decision)
        except GatewayError as err:
            self._record(request_id, tenant, Decision([], 0.0, 0.0), None, err.kind, started)
            return err.response(request_id)

        c, cand = served.completion, served.candidate
        body = ChatResponse(
            model=cand.deployment,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=c.text),
                    finish_reason=c.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=c.input_tokens,
                completion_tokens=c.output_tokens,
                total_tokens=c.input_tokens + c.output_tokens,
            ),
            routing=RoutingInfo(
                router_model=cand.router_model,
                deployment=cand.deployment,
                predicted_quality=None if cand.quality != cand.quality else round(cand.quality, 4),
                predicted_cost_usd=cand.cost_usd,
                cost_usd=c.cost_usd,
                lambda_=decision.lam,
                attempts=[a.as_dict() for a in served.attempts],
                router_ms=round(decision.router_ms, 3),
            ),
        ).model_dump(by_alias=True)
        self._record(request_id, tenant, decision, served, "ok", started)
        if cacheable:
            await self.cache.put(key, body)
        headers = {
            "x-request-id": request_id,
            "x-router-model": cand.router_model,
            "x-router-deployment": cand.deployment,
        }
        return JSONResponse(body, headers=headers)

    # --- streaming --------------------------------------------------------------------------

    async def chat_stream(
        self, request_id: str, tenant: Tenant, req: ChatRequest, decision: Decision, started: float
    ) -> Response:
        """Fallback is possible until the first byte reaches the client, not after."""
        deadline = time.monotonic() + self.cfg.request_deadline_s
        attempts: list[Attempt] = []
        tried = 0
        for cand in decision.candidates:
            if tried >= self.cfg.max_attempts:
                break
            breaker = self.breakers[cand.deployment]
            if not breaker.allow():
                attempts.append(Attempt(cand.deployment, "skipped:breaker"))
                continue
            if not await self._reserve(tenant, cand):
                attempts.append(Attempt(cand.deployment, "skipped:budget"))
                continue
            tried += 1
            d = self.deployments[cand.deployment]
            it = self.providers[d.name].stream(self._provider_request(req, d)).__aiter__()
            start = time.perf_counter()
            try:
                async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                    first = await it.__anext__()
            except (ProviderError, TimeoutError, StopAsyncIteration) as exc:
                with contextlib.suppress(Exception):
                    await it.aclose()  # type: ignore[attr-defined]
                kind = exc.kind if isinstance(exc, ProviderError) else "timeout"
                attempts.append(
                    Attempt(d.name, f"error:{kind}", (time.perf_counter() - start) * 1e3)
                )
                await self.limits.settle(tenant.name, cand.cost_usd, 0.0)
                self.metrics.attempt_errors.labels(d.name, kind).inc()
                if kind in RETRY_KINDS:
                    breaker.record_failure()
                self._sync_breaker_gauge(d.name)
                if isinstance(exc, ProviderError) and not exc.fallback:
                    raise GatewayError(exc.status or 400, "upstream_rejected", str(exc)) from exc
                continue
            breaker.record_success()
            attempts.append(Attempt(d.name, "ok", (time.perf_counter() - start) * 1e3))
            return StreamingResponse(
                self._sse(request_id, tenant, decision, cand, attempts, first, it, started),
                media_type="text/event-stream",
                headers={
                    "x-request-id": request_id,
                    "x-router-model": cand.router_model,
                    "x-router-deployment": cand.deployment,
                    "cache-control": "no-cache",
                },
            )
        raise GatewayError(
            503,
            "upstream_unavailable",
            "all candidates failed",
            {"attempts": [a.as_dict() for a in attempts]},
        )

    async def _sse(
        self,
        request_id: str,
        tenant: Tenant,
        decision: Decision,
        cand: Candidate,
        attempts: list[Attempt],
        first: str | Completion,
        it: AsyncIterator[str | Completion],
        started: float,
    ) -> AsyncIterator[bytes]:
        cid = f"chatcmpl-{request_id[:24]}"
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            body = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": cand.deployment,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(body)}\n\n".encode()

        final: Completion | None = None
        yield chunk({"role": "assistant"})
        try:
            item: str | Completion | None = first
            while item is not None:
                if isinstance(item, Completion):
                    final = item
                elif item:
                    yield chunk({"content": item})
                item = await anext(it, None)
            yield chunk({}, final.finish_reason if final else "stop")
            routing = {
                "router_model": cand.router_model,
                "deployment": cand.deployment,
                "cost_usd": final.cost_usd if final else None,
                "attempts": [a.as_dict() for a in attempts],
            }
            yield f"data: {json.dumps({'routing': routing})}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except ProviderError as err:
            error = {"error": {"type": err.kind, "message": str(err)}}
            yield f"data: {json.dumps(error)}\n\n".encode()
        finally:
            actual = final.cost_usd if final else 0.0
            await self.limits.settle(tenant.name, cand.cost_usd, actual)
            served = Served(final, cand, attempts) if final else None
            self._record(request_id, tenant, decision, served, "ok" if final else "error", started)


def create_app(cfg: GatewayConfig, gateway: Gateway | None = None) -> FastAPI:
    """Build the ASGI app.  Pass a prebuilt `gateway` in tests to inject fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        gw = gateway
        if gw is None:
            limits: Limits
            redis_client = None
            if cfg.redis_url:
                redis_limits = RedisLimits.from_url(cfg.redis_url)
                redis_client, limits = redis_limits.r, redis_limits
            else:
                limits = MemoryLimits()
            gw = Gateway(
                cfg,
                RouterService.from_config(cfg),
                build_providers(cfg),
                limits,
                ResponseCache(cfg.cache_ttl_s, redis_client),
                DecisionLog(cfg.decision_log),
            )
        app.state.gateway = gw
        await gw.router.start()
        gw.decisions.start()
        try:
            yield
        finally:
            await gw.router.stop()
            await gw.decisions.stop()
            for p in gw.providers.values():
                await p.aclose()
            await gw.limits.aclose()

    app = FastAPI(title="modelrouter gateway", version="0.1.0", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatRequest) -> Response:
        gw: Gateway = request.app.state.gateway
        return await gw.chat(request, body)

    @app.post("/v1/router/decide")
    async def decide(request: Request, body: ChatRequest) -> Response:
        """Dry run: the ranking the router would use, without calling any upstream."""
        gw: Gateway = request.app.state.gateway
        try:
            tenant = gw.authenticate(request)
            decision = await gw._candidates(tenant, body)
        except GatewayError as err:
            return err.response(uuid.uuid4().hex)
        return JSONResponse(
            {
                "lambda": decision.lam,
                "router_ms": round(decision.router_ms, 3),
                "candidates": [c.__dict__ for c in decision.candidates],
            }
        )

    @app.post("/admin/faults")
    async def faults(request: Request) -> Response:
        """Set a replay deployment's injected failure rate (only with `fault_injection: true`)."""
        gw: Gateway = request.app.state.gateway
        if not gw.cfg.fault_injection:
            return JSONResponse({"error": {"type": "not_found"}}, status_code=404)
        try:
            gw.authenticate(request)
        except GatewayError as err:
            return err.response(uuid.uuid4().hex)
        body = await request.json()
        provider = gw.providers.get(str(body.get("deployment")))
        if provider is None or not hasattr(provider, "failure_rate"):
            return JSONResponse({"error": {"type": "unknown_deployment"}}, status_code=400)
        provider.failure_rate = float(body.get("failure_rate", 0.0))
        return JSONResponse(
            {"deployment": provider.deployment.name, "failure_rate": provider.failure_rate}
        )

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        gw: Gateway = request.app.state.gateway
        data = [{"id": "auto", "object": "model", "owned_by": "router"}]
        data += [{"id": d, "object": "model", "owned_by": "gateway"} for d in gw.deployments]
        return {"object": "list", "data": data}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        gw: Gateway = request.app.state.gateway
        ok = await gw.limits.ping()
        open_breakers = [n for n, b in gw.breakers.items() if b.state is State.OPEN]
        return JSONResponse(
            {"ready": ok, "limits_backend": ok, "open_breakers": open_breakers},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        gw: Gateway = request.app.state.gateway
        return Response(generate_latest(gw.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
