import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gateway_fakes import AUTH, chat, make_config, make_gateway
from modelrouter.gateway.app import create_app
from modelrouter.gateway.batching import MicroBatcher
from modelrouter.gateway.config import BreakerConfig, Tenant
from modelrouter.gateway.limits import MemoryLimits, RedisLimits
from modelrouter.gateway.providers.base import ProviderError
from modelrouter.gateway.resilience import CircuitBreaker, State


def client_for(gw):
    return TestClient(create_app(gw.cfg, gateway=gw))


def test_rejects_missing_or_unknown_keys():
    gw, _ = make_gateway()
    with client_for(gw) as c:
        assert c.post("/v1/chat/completions", json=chat("hi")).status_code == 401
        r = c.post("/v1/chat/completions", json=chat("hi"), headers={"Authorization": "Bearer x"})
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "invalid_api_key"


def test_easy_prompts_go_cheap_and_hard_prompts_go_strong():
    gw, providers = make_gateway()
    with client_for(gw) as c:
        easy = c.post("/v1/chat/completions", json=chat("what is 2+2"), headers=AUTH)
        hard = c.post("/v1/chat/completions", json=chat("a hard proof"), headers=AUTH)
    assert easy.status_code == 200
    assert easy.json()["routing"]["router_model"] == "cheap"
    assert easy.headers["x-router-deployment"] == "cheap-dep"
    assert hard.json()["routing"]["router_model"] == "strong"
    assert hard.json()["choices"][0]["message"]["content"] == "strong-dep says hi"
    assert providers["mid-dep"].calls == 0


def test_lambda_zero_buys_the_best_model_and_allowed_models_restricts():
    gw, _ = make_gateway()
    with client_for(gw) as c:
        best = c.post("/v1/chat/completions", json=chat("easy", router={"lambda": 0}), headers=AUTH)
        only_mid = c.post(
            "/v1/chat/completions",
            json=chat("hard", router={"allowed_models": ["mid"]}),
            headers=AUTH,
        )
    assert best.json()["routing"]["router_model"] == "strong"
    assert only_mid.json()["routing"]["router_model"] == "mid"


def test_explicit_model_bypasses_routing():
    gw, _ = make_gateway()
    with client_for(gw) as c:
        r = c.post("/v1/chat/completions", json=chat("hard", model="mid-dep"), headers=AUTH)
        bad = c.post("/v1/chat/completions", json=chat("hard", model="nope"), headers=AUTH)
    assert r.json()["routing"]["deployment"] == "mid-dep"
    assert bad.status_code == 400


def test_falls_back_to_the_next_candidate_on_overload():
    overloaded = ProviderError("overloaded", "503 from upstream", 503)
    gw, _ = make_gateway(scripts={"strong-dep": [overloaded]})
    with client_for(gw) as c:
        r = c.post("/v1/chat/completions", json=chat("hard question"), headers=AUTH)
        metrics = c.get("/metrics").text
    body = r.json()
    assert r.status_code == 200
    assert body["routing"]["deployment"] == "mid-dep"
    assert [a["outcome"] for a in body["routing"]["attempts"]] == ["error:overloaded", "ok"]
    assert "router_fallbacks_total 1.0" in metrics


def test_client_errors_are_returned_without_fallback():
    bad = ProviderError("client", "context too long", 400)
    gw, providers = make_gateway(scripts={"strong-dep": [bad]})
    with client_for(gw) as c:
        r = c.post("/v1/chat/completions", json=chat("hard question"), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "upstream_rejected"
    assert providers["mid-dep"].calls == 0


def test_breaker_opens_and_traffic_skips_the_dead_deployment():
    err = ProviderError("server", "500", 500)
    gw, providers = make_gateway(scripts={"strong-dep": [err, err, err, err]})
    with client_for(gw) as c:
        for _ in range(4):
            c.post("/v1/chat/completions", json=chat("hard question"), headers=AUTH)
        ready = c.get("/readyz").json()
    # threshold is 2: after two failures the breaker is open and strong-dep isn't called again
    assert providers["strong-dep"].calls == 2
    assert ready["open_breakers"] == ["strong-dep"]


def test_breaker_half_open_probe_closes_on_success():
    now = [0.0]
    b = CircuitBreaker(BreakerConfig(failure_threshold=1, reset_after_s=5), clock=lambda: now[0])
    b.record_failure()
    assert b.state is State.OPEN
    assert not b.allow()
    now[0] = 6.0
    assert b.allow()  # the probe
    assert not b.allow()  # only one probe in flight
    b.record_success()
    assert b.state is State.CLOSED


def test_budget_blocks_spend_and_routes_cheaper_when_the_top_choice_does_not_fit():
    tenant = Tenant(
        name="acme",
        key_sha256=Tenant.hash_key("sk-test-tenant"),
        daily_budget_usd=0.0005,
        default_lambda=0.0,
    )
    gw, _ = make_gateway(make_config(tenants=[tenant]))
    with client_for(gw) as c:
        # λ=0 prefers strong (predicted $0.001) but only $0.0005 remains -> mid ($0.0001).
        r1 = c.post("/v1/chat/completions", json=chat("hard"), headers=AUTH)
        assert r1.json()["routing"]["deployment"] == "mid-dep"
        outcomes = [a["outcome"] for a in r1.json()["routing"]["attempts"]]
        assert outcomes == ["skipped:budget", "ok"]
        codes = [
            c.post("/v1/chat/completions", json=chat("hard"), headers=AUTH).status_code
            for _ in range(10)
        ]
    assert 402 in codes


def test_rate_limit():
    tenant = Tenant(
        name="acme", key_sha256=Tenant.hash_key("sk-test-tenant"), requests_per_minute=3
    )
    gw, _ = make_gateway(make_config(tenants=[tenant]))
    with client_for(gw) as c:
        codes = [
            c.post("/v1/chat/completions", json=chat("x"), headers=AUTH).status_code
            for _ in range(5)
        ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_cache_serves_repeated_deterministic_requests():
    gw, providers = make_gateway(cache_ttl_s=60)
    with client_for(gw) as c:
        a = c.post("/v1/chat/completions", json=chat("easy", temperature=0), headers=AUTH)
        b = c.post("/v1/chat/completions", json=chat("easy", temperature=0), headers=AUTH)
        warm = c.post("/v1/chat/completions", json=chat("easy", temperature=0.7), headers=AUTH)
    assert a.json()["routing"]["cache_hit"] is False
    assert b.json()["routing"]["cache_hit"] is True
    assert b.json()["routing"]["cost_usd"] == 0.0
    assert warm.json()["routing"]["cache_hit"] is False
    assert providers["cheap-dep"].calls == 2


def test_streaming_with_fallback_before_first_byte():
    gw, _ = make_gateway(scripts={"strong-dep": [ProviderError("timeout", "slow")]})
    with (
        client_for(gw) as c,
        c.stream("POST", "/v1/chat/completions", json=chat("hard", stream=True), headers=AUTH) as r,
    ):
        assert r.status_code == 200
        assert r.headers["x-router-deployment"] == "mid-dep"
        events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(
        json.loads(e)["choices"][0]["delta"].get("content", "")
        for e in events[:-1]
        if "choices" in json.loads(e)
    )
    assert text.strip() == "mid-dep says hi"
    routing = next(json.loads(e)["routing"] for e in events[:-1] if "routing" in json.loads(e))
    assert routing["attempts"][0]["outcome"] == "error:timeout"


def test_decide_endpoint_and_models_listing():
    gw, providers = make_gateway()
    with client_for(gw) as c:
        d = c.post("/v1/router/decide", json=chat("hard"), headers=AUTH).json()
        models = c.get("/v1/models").json()
    assert d["candidates"][0]["router_model"] == "strong"
    assert sum(p.calls for p in providers.values()) == 0
    assert {m["id"] for m in models["data"]} >= {"auto", "cheap-dep"}


def test_decision_log_rows(tmp_path):
    path = tmp_path / "decisions.jsonl"
    gw, _ = make_gateway(decision_log=path)
    with client_for(gw) as c:
        c.post("/v1/chat/completions", json=chat("hard"), headers=AUTH)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["router_model"] == "strong"
    assert rows[0]["ranked"][0]["model"] == "strong"


async def test_micro_batcher_groups_concurrent_calls():
    seen: list[int] = []

    def fn(items):
        seen.append(len(items))
        return [x * 2 for x in items]

    b = MicroBatcher(fn, max_size=8, max_wait_ms=20)
    b.start()
    out = await asyncio.gather(*(b.submit(i) for i in range(20)))
    await b.stop()
    assert out == [i * 2 for i in range(20)]
    assert max(seen) == 8
    assert len(seen) <= 4


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_limits_backends(backend):
    if backend == "redis":
        import fakeredis

        limits = RedisLimits(fakeredis.FakeAsyncRedis())
    else:
        limits = MemoryLimits()
    assert await limits.reserve("t", 0.4, 1.0)
    assert await limits.reserve("t", 0.5, 1.0)
    assert not await limits.reserve("t", 0.2, 1.0)  # 0.9 + 0.2 > 1.0
    await limits.settle("t", 0.5, 0.1)  # actual was cheaper: frees 0.4
    assert await limits.spent("t") == pytest.approx(0.5)
    assert await limits.reserve("t", 0.2, 1.0)
    allowed = [await limits.allow_request("t", 3) for _ in range(5)]
    assert allowed == [True, True, True, False, False]


def test_fault_injection_endpoint_is_off_unless_enabled():
    gw, _ = make_gateway()
    with client_for(gw) as c:
        r = c.post(
            "/admin/faults", json={"deployment": "strong-dep", "failure_rate": 1}, headers=AUTH
        )
    assert r.status_code == 404


def test_cache_is_per_tenant():
    other = Tenant(name="other", key_sha256=Tenant.hash_key("sk-other"), daily_budget_usd=1.0)
    cfg = make_config()
    cfg.tenants.append(other)
    gw, providers = make_gateway(cfg, cache_ttl_s=60)
    with client_for(gw) as c:
        c.post("/v1/chat/completions", json=chat("easy", temperature=0), headers=AUTH)
        r = c.post(
            "/v1/chat/completions",
            json=chat("easy", temperature=0),
            headers={"Authorization": "Bearer sk-other"},
        )
    assert r.json()["routing"]["cache_hit"] is False
    assert providers["cheap-dep"].calls == 2
