"""Per-deployment circuit breakers and retry backoff.

Breaker states: closed (normal) -> open after `failure_threshold` consecutive failures -> half-open
after `reset_after_s`, where up to `half_open_probes` requests are let through; a success closes
it, a failure re-opens it.  Open deployments are skipped by the router, so traffic shifts to the
next-best model instead of queueing behind a dead upstream.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from modelrouter.gateway.config import BreakerConfig


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    config: BreakerConfig
    clock: Callable[[], float] = time.monotonic
    state: State = State.CLOSED
    failures: int = 0
    opened_at: float = 0.0
    probes_in_flight: int = 0
    transitions: list[tuple[float, State]] = field(default_factory=list)

    def _move(self, state: State) -> None:
        if state is not self.state:
            self.state = state
            self.transitions.append((self.clock(), state))

    def allow(self) -> bool:
        if self.state is State.OPEN and self.clock() - self.opened_at >= self.config.reset_after_s:
            self._move(State.HALF_OPEN)
            self.probes_in_flight = 0
        if self.state is State.CLOSED:
            return True
        if self.state is State.HALF_OPEN and self.probes_in_flight < self.config.half_open_probes:
            self.probes_in_flight += 1
            return True
        return False

    def available(self) -> bool:
        """Like allow() but without reserving a probe (used when ranking candidates)."""
        if self.state is State.OPEN:
            return self.clock() - self.opened_at >= self.config.reset_after_s
        if self.state is State.HALF_OPEN:
            return self.probes_in_flight < self.config.half_open_probes
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.probes_in_flight = 0
        self._move(State.CLOSED)

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is State.HALF_OPEN or self.failures >= self.config.failure_threshold:
            self.opened_at = self.clock()
            self.probes_in_flight = 0
            self._move(State.OPEN)


def backoff_s(
    attempt: int, base: float = 0.05, cap: float = 1.0, rng: random.Random | None = None
) -> float:
    """Full-jitter exponential backoff: U(0, min(cap, base * 2**attempt))."""
    r = rng or random
    return r.uniform(0.0, min(cap, base * (2**attempt)))
