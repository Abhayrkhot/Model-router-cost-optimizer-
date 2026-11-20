"""Micro-batching for router inference.

Concurrent requests each need one embedding + one forward pass.  Running them one by one wastes
the ONNX runtime's batch efficiency and, worse, blocks the event loop.  The batcher collects
requests for up to `max_wait_ms` (or `max_size` items), runs the batch function once in a worker
thread, and resolves each caller's future with its row.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class MicroBatcher(Generic[T, R]):
    def __init__(
        self, fn: Callable[[Sequence[T]], Sequence[R]], max_size: int = 32, max_wait_ms: float = 2.0
    ) -> None:
        self.fn = fn
        self.max_size = max_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[tuple[T, asyncio.Future[R]]] = asyncio.Queue()
        self.batch_sizes: list[int] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="router-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, item: T) -> R:
        fut: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = loop.time() + self.max_wait_s
            while len(batch) < self.max_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except TimeoutError:
                    break
            self.batch_sizes.append(len(batch))
            items = [item for item, _ in batch]
            try:
                results = await asyncio.to_thread(self.fn, items)
            except Exception as exc:  # propagate to every waiter in the batch
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), res in zip(batch, results, strict=True):
                if not fut.done():
                    fut.set_result(res)
