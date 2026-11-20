"""Prometheus metrics (one registry per app instance, so tests don't share state)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "router_requests", "Chat requests by outcome", ["tenant", "status"], registry=r
        )
        self.routed = Counter(
            "router_routed", "Requests served, by deployment", ["deployment"], registry=r
        )
        self.cost = Counter(
            "router_cost_usd", "Upstream spend in USD", ["tenant", "deployment"], registry=r
        )
        self.tokens = Counter(
            "router_tokens", "Tokens by direction", ["deployment", "direction"], registry=r
        )
        self.attempt_errors = Counter(
            "router_attempt_errors", "Failed upstream attempts", ["deployment", "kind"], registry=r
        )
        self.fallbacks = Counter(
            "router_fallbacks", "Requests served by a non-first candidate", registry=r
        )
        self.cache = Counter("router_cache", "Response cache lookups", ["result"], registry=r)
        self.breaker_state = Gauge(
            "router_breaker_open",
            "1 if the deployment's breaker is open",
            ["deployment"],
            registry=r,
        )
        self.router_latency = Histogram(
            "router_decision_seconds",
            "Embedding + scoring time per request",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.upstream_latency = Histogram(
            "router_upstream_seconds",
            "Upstream call time",
            ["deployment"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.request_latency = Histogram(
            "router_request_seconds",
            "End-to-end gateway time",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.batch_size = Histogram(
            "router_batch_size",
            "Router micro-batch sizes",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=r,
        )
