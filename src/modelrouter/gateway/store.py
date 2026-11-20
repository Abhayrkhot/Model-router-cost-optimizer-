"""Response cache and decision log."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class ResponseCache:
    """Exact-match cache for deterministic requests.  Redis when configured, else in-process LRU.

    Only requests with temperature unset or 0 are cached; the key covers the messages, the model
    selector and every routing option, so two tenants with different λ never share an entry.
    """

    def __init__(self, ttl_s: int, redis: Any | None = None, max_items: int = 10_000) -> None:
        self.ttl_s = ttl_s
        self.redis = redis
        self.max_items = max_items
        self.lru: OrderedDict[str, tuple[float, str]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    async def get(self, key: str) -> dict[str, Any] | None:
        if self.redis is not None:
            raw = await self.redis.get(f"mr:cache:{key}")
            if raw is None:
                return None
            cached: dict[str, Any] = json.loads(raw)
            return cached
        hit = self.lru.get(key)
        if hit is None or hit[0] < time.monotonic():
            self.lru.pop(key, None)
            return None
        self.lru.move_to_end(key)
        value: dict[str, Any] = json.loads(hit[1])
        return value

    async def put(self, key: str, value: dict[str, Any]) -> None:
        raw = json.dumps(value)
        if self.redis is not None:
            await self.redis.set(f"mr:cache:{key}", raw, ex=self.ttl_s)
            return
        self.lru[key] = (time.monotonic() + self.ttl_s, raw)
        self.lru.move_to_end(key)
        while len(self.lru) > self.max_items:
            self.lru.popitem(last=False)


class DecisionLog:
    """Append-only JSONL of routing decisions, written off the request path.

    These rows are the training data for the next router: what was predicted, what was chosen,
    what it cost.  Join them with quality feedback (thumbs, evals) to retrain.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=50_000)
        self.dropped = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.path is not None and self._task is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._task = asyncio.create_task(self._run(), name="decision-log")

    def write(self, row: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.queue.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self) -> None:
        assert self.path is not None
        fh = await asyncio.to_thread(open, self.path, "a", encoding="utf-8")
        try:
            while True:
                rows = [await self.queue.get()]
                while not self.queue.empty() and len(rows) < 512:
                    rows.append(self.queue.get_nowait())
                text = "\n".join(json.dumps(r) for r in rows) + "\n"
                await asyncio.to_thread(_append, fh, text)
                for _ in rows:
                    self.queue.task_done()
        finally:
            fh.close()

    async def stop(self) -> None:
        if self._task is None:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.queue.join(), timeout=5.0)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None


def _append(fh: Any, text: str) -> None:
    fh.write(text)
    fh.flush()
