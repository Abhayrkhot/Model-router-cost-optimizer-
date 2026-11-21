"""Online routing: score every candidate model for a request and rank by utility."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modelrouter.gateway.batching import MicroBatcher
from modelrouter.gateway.config import GatewayConfig
from modelrouter.routers.base import CostModel, FloatArray, Inputs, QualityModel, load_router

NEEDS_EMBEDDING = {"knn", "mlp", "ensemble"}


@dataclass(frozen=True)
class Candidate:
    router_model: str
    deployment: str
    quality: float
    cost_usd: float
    utility: float


@dataclass(frozen=True)
class Decision:
    candidates: list[Candidate]
    lam: float
    router_ms: float


class RouterService:
    def __init__(
        self,
        quality_model: QualityModel,
        cost_model: CostModel,
        model_map: dict[str, str],
        embedder: object | None = None,
        batch_max_size: int = 32,
        batch_max_wait_ms: float = 2.0,
    ) -> None:
        self.qm = quality_model
        self.cm = cost_model
        self.embedder = embedder
        self.models = quality_model.models
        self.routable = np.array([m in model_map for m in self.models])
        self.model_map = model_map
        self.batcher: MicroBatcher[tuple[str, int], tuple[FloatArray, FloatArray]] = MicroBatcher(
            self._score_batch, batch_max_size, batch_max_wait_ms
        )

    @classmethod
    def from_config(cls, cfg: GatewayConfig) -> RouterService:
        qm, cm, _meta = load_router(Path(cfg.router_dir))
        embedder = None
        if qm.kind in NEEDS_EMBEDDING:
            from modelrouter.features.embed import Embedder

            embedder = Embedder(threads=cfg.embed_threads)
        return cls(qm, cm, cfg.model_map, embedder, cfg.batch_max_size, cfg.batch_max_wait_ms)

    def _score_batch(self, items: Sequence[tuple[str, int]]) -> list[tuple[FloatArray, FloatArray]]:
        texts = [t for t, _ in items]
        chars = np.array([c for _, c in items], dtype=np.int64)
        emb = self.embedder.embed(texts) if self.embedder is not None else None  # type: ignore[attr-defined]
        q = self.qm.predict(Inputs(texts, chars, emb))
        c = self.cm.predict(chars)
        return [(q[i], c[i]) for i in range(len(items))]

    async def start(self) -> None:
        if self.embedder is not None:
            self.batcher.start()
            await self.batcher.submit(("warm-up", 7))  # warm the ONNX session first

    async def stop(self) -> None:
        await self.batcher.stop()

    async def decide(
        self,
        text: str,
        chars: int,
        lam: float,
        allowed: Sequence[str] | None = None,
        max_cost_usd: float | None = None,
    ) -> Decision:
        start = time.perf_counter()
        if self.embedder is None:
            # Lexical routers score in ~0.1 ms: batching would only add its wait window.
            q, c = self._score_batch([(text, chars)])[0]
        else:
            q, c = await self.batcher.submit((text, chars))
        mask = self.routable.copy()
        if allowed is not None:
            mask &= np.array([m in set(allowed) for m in self.models])
        if max_cost_usd is not None:
            mask &= c <= max_cost_usd
        utility = q - lam * c
        order = [i for i in np.argsort(-utility, kind="stable") if mask[i]]
        cands = [
            Candidate(
                self.models[i],
                self.model_map[self.models[i]],
                float(q[i]),
                float(c[i]),
                float(utility[i]),
            )
            for i in order
        ]
        return Decision(cands, lam, (time.perf_counter() - start) * 1000.0)
