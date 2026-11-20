"""Reference policies: they bound what a learned router has to beat."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from modelrouter.routers.base import FloatArray, Inputs, IntArray, QualityModel, register


@register
class PriorQuality(QualityModel):
    """Every prompt gets each model's average training quality.

    With the λ policy this only ever picks one model per λ, so its curve is the set of single
    models; mixing them is RouterBench's "zero router".  A learned router must beat it.
    """

    kind: ClassVar[str] = "prior"

    def __init__(self, models: tuple[str, ...]) -> None:
        self.models = models
        self.mean = np.zeros(len(models))

    def fit(
        self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray
    ) -> PriorQuality:
        self.mean = q_train.mean(axis=0)
        return self

    def predict(self, x: Inputs) -> FloatArray:
        return np.tile(self.mean, (len(x), 1))

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {"mean": self.mean}

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> PriorQuality:
        pm = cls(models)
        pm.mean = arrays["mean"]
        return pm


def oracle_decisions(quality: FloatArray, cost: FloatArray, lambdas: FloatArray) -> IntArray:
    """Routes with the true outcomes: the upper bound for any router on this prompt set."""
    return np.stack([np.argmax(quality - lam * cost, axis=1) for lam in lambdas]).astype(np.int64)


# The original project's rule: word count < 20 -> small, < 50 -> medium, else large.
LENGTH_THRESHOLDS = (20, 50)


def word_counts(texts: Sequence[str]) -> IntArray:
    return np.array([len(t.split()) for t in texts], dtype=np.int64)


def length_tier_decisions(
    texts: Sequence[str],
    tiers: tuple[int, int, int],
    thresholds: tuple[int, int] = LENGTH_THRESHOLDS,
) -> IntArray:
    """(1, n) decisions for the length rule; `tiers` are model indices for small/medium/large."""
    wc = word_counts(texts)
    small, medium, large = tiers
    out = np.where(wc < thresholds[0], small, np.where(wc < thresholds[1], medium, large))
    return out[None, :].astype(np.int64)
