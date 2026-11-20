"""Cost–quality metrics for routing policies.

A policy with a knob (λ, or the share of traffic sent to the strong model) traces a curve of
operating points (mean cost per prompt, mean quality).  Everything here is computed from realised
outcomes: the quality and cost of the answer the chosen model actually gave.

Two families of metrics:

* Multi-model (RouterBench): the curve's upper hull, its normalised area (AIQ), the cheapest
  operating point that matches a reference model's quality, and the best quality within a budget.
  Hull values assume randomised mixing of two operating points; the "point" metrics don't.
* Two-model (RouteLLM): route the top p% of prompts by a score to the strong model.  PGR(p) is
  the share of the weak→strong quality gap recovered; APGR is its mean over p ∈ [0, 1]; CPT(x) is
  the smallest p with PGR ≥ x.  A random router has APGR = 0.5 in expectation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Curve:
    """Operating points of one policy on one prompt set; `params` is the knob per point."""

    name: str
    cost: FloatArray
    quality: FloatArray
    params: FloatArray

    def point_at(self, param: float) -> tuple[float, float]:
        k = int(np.flatnonzero(self.params == param)[0])
        return float(self.cost[k]), float(self.quality[k])


def realised(
    quality: FloatArray, cost: FloatArray, decisions: IntArray
) -> tuple[FloatArray, FloatArray]:
    """Per-row outcomes for a (K, n) decision matrix -> (K, n) cost and (K, n) quality."""
    rows = np.arange(quality.shape[0])[None, :]
    return cost[rows, decisions], quality[rows, decisions]


def curve(
    name: str, quality: FloatArray, cost: FloatArray, decisions: IntArray, params: FloatArray
) -> Curve:
    c, q = realised(quality, cost, decisions)
    return Curve(name, c.mean(axis=1), q.mean(axis=1), np.asarray(params, dtype=np.float64))


def single_model_points(quality: FloatArray, cost: FloatArray) -> tuple[FloatArray, FloatArray]:
    return cost.mean(axis=0), quality.mean(axis=0)


# --- hull --------------------------------------------------------------------------------


def upper_hull(cost: FloatArray, quality: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Non-decreasing concave hull: the best quality reachable at each cost by mixing points."""
    order = np.lexsort((-quality, cost))
    pts = np.column_stack([cost[order], quality[order]])
    hull: list[tuple[float, float]] = []
    for c, q in pts:
        if hull and q <= hull[-1][1]:
            continue  # dominated: costs at least as much, no better
        while len(hull) >= 2:
            (c1, q1), (c2, q2) = hull[-2], hull[-1]
            # drop the middle point if it lies on/below the chord (keeps the hull concave)
            if (q2 - q1) * (c - c1) <= (q - q1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append((float(c), float(q)))
    h = np.asarray(hull)
    return h[:, 0], h[:, 1]


def hull_quality_at(hull_cost: FloatArray, hull_quality: FloatArray, at: FloatArray) -> FloatArray:
    """Hull value at each cost; flat beyond the last hull point, first value before the first."""
    out: FloatArray = np.interp(
        at, hull_cost, hull_quality, left=hull_quality[0], right=hull_quality[-1]
    )
    return out


def aiq(c: Curve, lo: float, hi: float, log: bool = False, grid: int = 2048) -> float:
    """Average hull quality over [lo, hi] (RouterBench's AIQ); log=True averages over log cost."""
    hc, hq = upper_hull(c.cost, c.quality)
    xs = np.geomspace(lo, hi, grid) if log else np.linspace(lo, hi, grid)
    vals = hull_quality_at(hc, hq, xs)
    if log:
        return float(np.trapezoid(vals, np.log(xs)) / (np.log(hi) - np.log(lo)))
    return float(np.trapezoid(vals, xs) / (hi - lo))


# --- point metrics (no mixing) --------------------------------------------------------------


def min_cost_reaching(c: Curve, target: float) -> tuple[float, float]:
    """(cost, param) of the cheapest operating point with quality >= target; (inf, nan) if none."""
    ok = np.flatnonzero(c.quality >= target - 1e-12)
    if len(ok) == 0:
        return float("inf"), float("nan")
    k = ok[np.argmin(c.cost[ok])]
    return float(c.cost[k]), float(c.params[k])


def best_quality_within(c: Curve, budget: float) -> tuple[float, float]:
    """(quality, param) of the best operating point costing <= budget; (nan, nan) if none."""
    ok = np.flatnonzero(c.cost <= budget + 1e-15)
    if len(ok) == 0:
        return float("nan"), float("nan")
    k = ok[np.argmax(c.quality[ok])]
    return float(c.quality[k]), float(c.params[k])


# --- two-model routing ------------------------------------------------------------------------

P_GRID = np.linspace(0.0, 1.0, 101)


def strong_share_curve(
    quality: FloatArray,
    cost: FloatArray,
    strong: int,
    weak: int,
    score: FloatArray,
    p_grid: FloatArray = P_GRID,
    seed: int = 0,
) -> Curve:
    """Send the top-p share of prompts by `score` to `strong`, the rest to `weak`."""
    n = quality.shape[0]
    jitter = np.random.default_rng(seed).random(n) * 1e-9  # break ties without bias
    order = np.argsort(-(score + jitter), kind="stable")
    ks = np.round(p_grid * n).astype(np.int64)
    decisions = np.full((len(ks), n), weak, dtype=np.int64)
    for row, k in enumerate(ks):
        decisions[row, order[:k]] = strong
    return curve("strong-share", quality, cost, decisions, p_grid)


def pgr(c: Curve, q_weak: float, q_strong: float) -> FloatArray:
    return (c.quality - q_weak) / (q_strong - q_weak)


def apgr(c: Curve, q_weak: float, q_strong: float) -> float:
    return float(np.trapezoid(pgr(c, q_weak, q_strong), c.params))


def cpt(c: Curve, q_weak: float, q_strong: float, x: float) -> float:
    """Smallest share of strong-model calls whose PGR reaches x (1.0 if never before the end)."""
    ok = np.flatnonzero(pgr(c, q_weak, q_strong) >= x - 1e-12)
    return float(c.params[ok[0]]) if len(ok) else 1.0


# --- bootstrap --------------------------------------------------------------------------------


def bootstrap(
    per_row_cost: FloatArray,
    per_row_quality: FloatArray,
    stat: Callable[[FloatArray, FloatArray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of `stat(mean_cost[K], mean_quality[K])` over resampled prompts.

    `per_row_*` are (K, n) outcome matrices for a fixed set of K operating points, so the policy
    is held fixed and only the prompt sample varies.
    """
    n = per_row_cost.shape[1]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64) / n
    mean_cost = weights @ per_row_cost.T  # (B, K)
    mean_quality = weights @ per_row_quality.T
    values = np.array([stat(mean_cost[b], mean_quality[b]) for b in range(n_boot)])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
