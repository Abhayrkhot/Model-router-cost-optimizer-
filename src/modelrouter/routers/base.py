"""Router building blocks: quality predictors, a cost model and the routing policy.

A router is split into three parts so each can be tested and swapped on its own:

* a `QualityModel` predicts P(model m answers prompt i well), shape (n, m);
* a `CostModel` predicts what each model's answer would cost from the prompt length;
* the policy picks argmax_m  q̂[i, m] − λ · ĉ[i, m].  λ is dollars-to-quality: λ = 0 buys the
  best predicted answer at any price, large λ buys the cheapest.  One λ is one operating point.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt

from modelrouter.data.dataset import RoutingDataset

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]

# λ grid used for sweeps: 0 plus log-spaced values covering "always the best" to "always the
# cheapest" for per-prompt costs between ~$1e-5 and ~$1e-1.
LAMBDAS = np.concatenate([[0.0], np.logspace(-1, 6, 281)])


@dataclass(frozen=True)
class Inputs:
    """What a router sees about each prompt."""

    text: Sequence[str]
    prompt_chars: IntArray
    embedding: F32 | None = None

    def __len__(self) -> int:
        return len(self.text)

    @classmethod
    def from_dataset(cls, ds: RoutingDataset, embedding: F32 | None = None) -> Inputs:
        return cls(text=ds.text, prompt_chars=ds.prompt_chars, embedding=embedding)

    def require_embedding(self) -> F32:
        if self.embedding is None:
            raise ValueError("this router needs embeddings; pass Inputs.embedding")
        return self.embedding


class QualityModel(ABC):
    """Predicts per-model answer quality in [0, 1]."""

    kind: ClassVar[str]
    models: tuple[str, ...]

    @abstractmethod
    def fit(
        self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray
    ) -> QualityModel: ...

    @abstractmethod
    def predict(self, x: Inputs) -> FloatArray: ...

    # Persistence: a JSON config plus a NumPy archive of arrays.
    def config(self) -> dict[str, Any]:
        return {}

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {}

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> QualityModel:
        raise NotImplementedError(f"{cls.__name__} cannot be loaded")


class CostModel:
    """Per-model linear cost in prompt length: ĉ_m = a_m + b_m · chars (least squares)."""

    def __init__(self, models: tuple[str, ...]) -> None:
        self.models = models
        self.coef = np.zeros((2, len(models)))
        self.floor = np.zeros(len(models))

    def fit(self, prompt_chars: IntArray, cost: FloatArray) -> CostModel:
        x = np.column_stack([np.ones(len(prompt_chars)), prompt_chars.astype(np.float64)])
        self.coef, *_ = np.linalg.lstsq(x, cost, rcond=None)
        self.floor = cost.min(axis=0)
        return self

    def predict(self, prompt_chars: IntArray) -> FloatArray:
        x = np.column_stack([np.ones(len(prompt_chars)), prompt_chars.astype(np.float64)])
        return np.maximum(x @ self.coef, self.floor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "coef": self.coef.tolist(),
            "floor": self.floor.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CostModel:
        cm = cls(tuple(d["models"]))
        cm.coef = np.asarray(d["coef"])
        cm.floor = np.asarray(d["floor"])
        return cm


def route(q_hat: FloatArray, c_hat: FloatArray, lam: float) -> IntArray:
    return np.argmax(q_hat - lam * c_hat, axis=1).astype(np.int64)


def sweep(q_hat: FloatArray, c_hat: FloatArray, lambdas: FloatArray = LAMBDAS) -> IntArray:
    """(K, n) decisions, one row per λ."""
    return np.stack([route(q_hat, c_hat, lam) for lam in lambdas])


# --- persistence ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[QualityModel]] = {}


def register(cls: type[QualityModel]) -> type[QualityModel]:
    _REGISTRY[cls.kind] = cls
    return cls


def save_router(
    directory: Path, quality_model: QualityModel, cost_model: CostModel, meta: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    spec = {
        "kind": quality_model.kind,
        "models": list(quality_model.models),
        "config": quality_model.config(),
        "cost_model": cost_model.to_dict(),
        "meta": meta,
    }
    (directory / "router.json").write_text(json.dumps(spec, indent=2) + "\n")
    np.savez_compressed(directory / "arrays.npz", **cast(dict[str, Any], quality_model.arrays()))


def load_router(directory: Path) -> tuple[QualityModel, CostModel, dict[str, Any]]:
    # Router classes register on import; load them here so a fresh process (the gateway) can
    # resolve any saved kind without the caller importing the right module first.
    from modelrouter.routers import baselines, learned  # noqa: F401

    spec = json.loads((directory / "router.json").read_text())
    with np.load(directory / "arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    cls = _REGISTRY[spec["kind"]]
    qm = cls.from_saved(tuple(spec["models"]), spec["config"], arrays)
    return qm, CostModel.from_dict(spec["cost_model"]), spec["meta"]
