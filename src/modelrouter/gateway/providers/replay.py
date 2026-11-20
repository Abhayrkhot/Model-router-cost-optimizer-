"""Replay provider: answers with RouterBench's recorded response for (prompt, model).

This lets the whole gateway path (HTTP, routing, budgets, fallback, metrics) run against real
recorded answers with zero API spend, and makes the end-to-end result directly comparable with
the offline evaluation: same prompts, same responses, same costs.  Latency and failures are
injected from the deployment config.

The fixture is a gzipped JSONL file built by `modelrouter gateway build-replay`: one line per
prompt with its turns, and per model the response text, cost and quality.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from modelrouter.gateway.config import Deployment
from modelrouter.gateway.providers.base import Completion, Provider, ProviderError, ProviderRequest


def turns_key(turns: list[str]) -> str:
    return hashlib.sha256(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Recorded:
    text: str
    cost_usd: float
    quality: float


class ReplayStore:
    def __init__(self, records: dict[str, dict[str, Recorded]], sample_ids: dict[str, str]):
        self.records = records
        self.sample_ids = sample_ids

    @classmethod
    def load(cls, path: str | Path) -> ReplayStore:
        records: dict[str, dict[str, Recorded]] = {}
        sample_ids: dict[str, str] = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                key = turns_key(row["turns"])
                records[key] = {
                    m: Recorded(v["text"], float(v["cost"]), float(v["quality"]))
                    for m, v in row["models"].items()
                }
                sample_ids[key] = row["sample_id"]
        return cls(records, sample_ids)

    def lookup(self, turns: list[str], model: str) -> Recorded | None:
        rec = self.records.get(turns_key(turns))
        return None if rec is None else rec.get(model)


class ReplayProvider(Provider):
    def __init__(self, deployment: Deployment, store: ReplayStore, seed: int | None = None):
        super().__init__(deployment)
        self.store = store
        self.rng = random.Random(seed)
        # Fault injection can be changed at runtime (benchmarks flip it mid-run).
        self.failure_rate = deployment.failure_rate

    async def _delay(self) -> None:
        d = self.deployment
        ms = max(0.0, self.rng.gauss(d.latency_ms, d.latency_jitter_ms)) if d.latency_ms else 0.0
        if ms:
            await asyncio.sleep(ms / 1000.0)

    def _record(self, req: ProviderRequest) -> Recorded:
        turns = [m.content for m in req.messages if m.role != "assistant"]
        rec = self.store.lookup(turns, self.deployment.upstream_model)
        if rec is None:
            raise ProviderError("client", "prompt not in replay fixture", 404)
        return rec

    async def complete(self, req: ProviderRequest) -> Completion:
        await self._delay()
        if self.failure_rate and self.rng.random() < self.failure_rate:
            raise ProviderError("overloaded", "injected failure", 503)
        rec = self._record(req)
        chars_in = sum(len(m.content) for m in req.messages)
        return Completion(
            text=rec.text,
            input_tokens=max(1, chars_in // 4),
            output_tokens=max(1, len(rec.text) // 4),
            cost_usd=rec.cost_usd,
            served_model=self.deployment.upstream_model,
            extra={"recorded_quality": rec.quality},
        )

    async def stream(self, req: ProviderRequest) -> AsyncIterator[str | Completion]:
        completion = await self.complete(req)
        text = completion.text
        for i in range(0, len(text), 64):
            yield text[i : i + 64]
            await asyncio.sleep(0)
        yield completion
