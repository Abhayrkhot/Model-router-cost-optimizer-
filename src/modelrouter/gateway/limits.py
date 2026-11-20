"""Per-tenant rate limits and daily dollar budgets.

Redis-backed (atomic Lua scripts, shared by every gateway replica) or in-memory for a single
process.  Budgets use a reserve/settle pattern: the predicted cost is reserved before the
upstream call (so concurrent requests can't overshoot together) and replaced by the actual cost
afterwards.  Amounts are stored as integer micro-dollars.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

MICRO = 1_000_000
DAY_S = 86_400


def _day() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def _micro(usd: float) -> int:
    return max(0, round(usd * MICRO))


class Limits(ABC):
    @abstractmethod
    async def allow_request(self, tenant: str, per_minute: int) -> bool: ...

    @abstractmethod
    async def reserve(self, tenant: str, usd: float, limit_usd: float) -> bool: ...

    @abstractmethod
    async def settle(self, tenant: str, reserved_usd: float, actual_usd: float) -> None: ...

    @abstractmethod
    async def spent(self, tenant: str) -> float: ...

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class MemoryLimits(Limits):
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}
        self.budget: dict[tuple[str, str], int] = {}
        self.lock = asyncio.Lock()

    async def allow_request(self, tenant: str, per_minute: int) -> bool:
        rate, cap = per_minute / 60.0, float(per_minute)
        async with self.lock:
            now = self.clock()
            tokens, ts = self.buckets.get(tenant, (cap, now))
            tokens = min(cap, tokens + (now - ts) * rate)
            ok = tokens >= 1.0
            self.buckets[tenant] = (tokens - 1.0 if ok else tokens, now)
            return ok

    async def reserve(self, tenant: str, usd: float, limit_usd: float) -> bool:
        key = (tenant, _day())
        async with self.lock:
            cur = self.budget.get(key, 0)
            if cur + _micro(usd) > _micro(limit_usd):
                return False
            self.budget[key] = cur + _micro(usd)
            return True

    async def settle(self, tenant: str, reserved_usd: float, actual_usd: float) -> None:
        key = (tenant, _day())
        async with self.lock:
            self.budget[key] = max(
                0, self.budget.get(key, 0) + _micro(actual_usd) - _micro(reserved_usd)
            )

    async def spent(self, tenant: str) -> float:
        return self.budget.get((tenant, _day()), 0) / MICRO


_TOKEN_BUCKET = """
local rate = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local s = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(s[1]) or cap
local ts = tonumber(s[2]) or now
tokens = math.min(cap, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(cap / rate) + 1)
return allowed
"""

_RESERVE = """
local v = redis.call('INCRBY', KEYS[1], ARGV[1])
if v > tonumber(ARGV[2]) then
  redis.call('DECRBY', KEYS[1], ARGV[1])
  return 0
end
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
"""


class RedisLimits(Limits):
    def __init__(self, client: Any) -> None:
        self.r = client
        self._bucket = client.register_script(_TOKEN_BUCKET)
        self._reserve = client.register_script(_RESERVE)

    @classmethod
    def from_url(cls, url: str) -> RedisLimits:
        import redis.asyncio as redis

        return cls(redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5))

    async def allow_request(self, tenant: str, per_minute: int) -> bool:
        res = await self._bucket(keys=[f"mr:rl:{tenant}"], args=[per_minute / 60.0, per_minute])
        return bool(int(res))

    async def reserve(self, tenant: str, usd: float, limit_usd: float) -> bool:
        key = f"mr:budget:{tenant}:{_day()}"
        res = await self._reserve(keys=[key], args=[_micro(usd), _micro(limit_usd), 2 * DAY_S])
        return bool(int(res))

    async def settle(self, tenant: str, reserved_usd: float, actual_usd: float) -> None:
        delta = _micro(actual_usd) - _micro(reserved_usd)
        if delta:
            await self.r.incrby(f"mr:budget:{tenant}:{_day()}", delta)

    async def spent(self, tenant: str) -> float:
        v = await self.r.get(f"mr:budget:{tenant}:{_day()}")
        return int(v or 0) / MICRO

    async def ping(self) -> bool:
        try:
            return bool(await self.r.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self.r.aclose()
