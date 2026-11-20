"""Train routers on a split and score them on its test set.

Two kinds of numbers come out:

* curve metrics read off the test-set curve (AIQ, cheapest point matching the reference model,
  best quality within a budget).  They describe the router's trade-off but pick the operating
  point with test data, which is optimistic;
* calibrated metrics choose λ on the validation set (e.g. "cheapest λ whose val quality matches
  GPT-4's val quality") and report what that λ does on test.  These are what a deployment
  would see, and they're the headline numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modelrouter.data.dataset import RoutingDataset
from modelrouter.data.splits import Split
from modelrouter.eval import metrics as M
from modelrouter.routers.base import (
    F32,
    LAMBDAS,
    CostModel,
    FloatArray,
    Inputs,
    IntArray,
    QualityModel,
    sweep,
)
from modelrouter.routers.baselines import PriorQuality, length_tier_decisions, oracle_decisions

BUDGETS = (0.05, 0.1, 0.2, 0.5)  # as a fraction of the reference model's cost
QUALITY_TARGETS = (0.9, 0.95, 0.98, 1.0)  # as a fraction of the reference model's quality


@dataclass(frozen=True)
class RouterSpec:
    name: str
    factory: Callable[[tuple[str, ...]], QualityModel]


@dataclass(frozen=True)
class Reference:
    model: str = "gpt-4-1106-preview"
    strong: str = "gpt-4-1106-preview"
    weak: str = "mixtral-8x7b"
    small_medium_large: tuple[str, str, str] = ("mistral-7b", "mixtral-8x7b", "gpt-4-1106-preview")


@dataclass
class RouterResult:
    name: str
    metrics: dict[str, Any]
    test_curve: M.Curve
    two_model_curve: M.Curve | None = None
    fit_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


def _ci(lo_hi: tuple[float, float]) -> list[float]:
    return [round(lo_hi[0], 4), round(lo_hi[1], 4)]


def summarize(
    name: str,
    val: RoutingDataset,
    test: RoutingDataset,
    dec_val: IntArray,
    dec_test: IntArray,
    params: FloatArray,
    ref: Reference,
    two_model_score_test: FloatArray | None,
    n_boot: int,
    seed: int,
) -> RouterResult:
    r = test.model_index(ref.model)
    q_ref_test, c_ref_test = float(test.quality[:, r].mean()), float(test.cost[:, r].mean())
    q_ref_val, c_ref_val = float(val.quality[:, r].mean()), float(val.cost[:, r].mean())
    single_cost, _ = M.single_model_points(test.quality, test.cost)
    lo, hi = float(single_cost.min()), float(single_cost.max())

    cv = M.curve(name, val.quality, val.cost, dec_val, params)
    ct = M.curve(name, test.quality, test.cost, dec_test, params)
    out: dict[str, Any] = {
        "aiq": round(M.aiq(ct, lo, hi), 4),
        "aiq_log": round(M.aiq(ct, lo, hi, log=True), 4),
        "max_quality": round(float(ct.quality.max()), 4),
        "max_quality_cost_frac": round(float(ct.cost[np.argmax(ct.quality)] / c_ref_test), 4),
    }

    # Cost to reach a share of the reference model's quality (1.0 = parity).
    out["targets"] = {}
    for frac in QUALITY_TARGETS:
        cost_t, _ = M.min_cost_reaching(ct, frac * q_ref_test)
        entry_t: dict[str, Any] = {"cost_frac_curve": round(cost_t / c_ref_test, 4)}
        _, lam_t = M.min_cost_reaching(cv, frac * q_ref_val)
        if np.isfinite(lam_t):
            c_at, q_at = ct.point_at(lam_t)
            k = int(np.flatnonzero(params == lam_t)[0])
            shares = np.bincount(dec_test[k], minlength=len(test.models)) / len(test)
            entry_t["calibrated"] = {
                "lambda": float(lam_t),
                "cost_frac": round(c_at / c_ref_test, 4),
                "quality": round(q_at, 4),
                "quality_vs_ref": round(q_at / q_ref_test, 4),
                "model_share": {
                    m: round(float(x), 4) for m, x in zip(test.models, shares, strict=True) if x > 0
                },
            }
        else:
            entry_t["calibrated"] = None  # no λ reaches the target on validation
        out["targets"][str(frac)] = entry_t

    # Fixed budgets.
    out["budgets"] = {}
    for b in BUDGETS:
        q_curve, _ = M.best_quality_within(ct, b * c_ref_test)
        _, lam_b = M.best_quality_within(cv, b * c_ref_val)
        entry: dict[str, Any] = {"quality_curve": round(q_curve, 4)}
        if np.isfinite(lam_b):
            c_at, q_at = ct.point_at(lam_b)
            entry |= {"quality": round(q_at, 4), "cost_frac": round(c_at / c_ref_test, 4)}
        out["budgets"][str(b)] = entry

    # Bootstrap over test prompts (policy fixed).  The reference model is appended as the last
    # operating point so its mean is resampled together with the router's.
    if n_boot > 0:
        c_rows, q_rows = M.realised(test.quality, test.cost, dec_test)
        c_rows = np.vstack([c_rows, test.cost[:, r]])
        q_rows = np.vstack([q_rows, test.quality[:, r]])

        def target_stat(frac: float) -> Callable[[FloatArray, FloatArray], float]:
            def stat(mc: FloatArray, mq: FloatArray) -> float:
                cur = M.Curve(name, mc[:-1], mq[:-1], params)
                cost, _ = M.min_cost_reaching(cur, frac * mq[-1])
                return float(cost / mc[-1])

            return stat

        def budget_stat(mc: FloatArray, mq: FloatArray) -> float:
            cur = M.Curve(name, mc[:-1], mq[:-1], params)
            return float(M.best_quality_within(cur, 0.1 * mc[-1])[0] - mq[-1])

        out["ci95"] = {
            "cost_frac_at_95pct_quality": _ci(
                M.bootstrap(c_rows, q_rows, target_stat(0.95), n_boot, seed)
            ),
            "cost_frac_at_parity": _ci(M.bootstrap(c_rows, q_rows, target_stat(1.0), n_boot, seed)),
            "quality_minus_ref_at_10pct_cost": _ci(
                M.bootstrap(c_rows, q_rows, budget_stat, n_boot, seed)
            ),
        }

    result = RouterResult(name=name, metrics=out, test_curve=ct)

    if two_model_score_test is not None:
        s, w = test.model_index(ref.strong), test.model_index(ref.weak)
        c2 = M.strong_share_curve(test.quality, test.cost, s, w, two_model_score_test)
        q_w, q_s = float(c2.quality[0]), float(c2.quality[-1])
        tm: dict[str, Any] = {
            "strong": ref.strong,
            "weak": ref.weak,
            "apgr": round(M.apgr(c2, q_w, q_s), 4),
            "cpt50": round(M.cpt(c2, q_w, q_s, 0.5), 4),
            "cpt80": round(M.cpt(c2, q_w, q_s, 0.8), 4),
        }
        if n_boot > 0:
            n = len(test)
            order_dec = np.stack(
                [
                    np.where(
                        np.isin(np.arange(n), np.argsort(-two_model_score_test)[: round(p * n)]),
                        s,
                        w,
                    )
                    for p in c2.params
                ]
            )
            c_rows2, q_rows2 = M.realised(test.quality, test.cost, order_dec)

            def apgr_stat(mc: FloatArray, mq: FloatArray) -> float:
                cur = M.Curve(name, mc, mq, c2.params)
                return M.apgr(cur, mq[0], mq[-1])

            tm["apgr_ci95"] = _ci(M.bootstrap(c_rows2, q_rows2, apgr_stat, n_boot, seed))
        out["two_model"] = tm
        result.two_model_curve = c2
    return result


def run_split(
    ds: RoutingDataset,
    emb: F32 | None,
    split: Split,
    specs: list[RouterSpec],
    ref: Reference | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    ref = ref or Reference()
    train, val, test = ds.subset(split.train), ds.subset(split.val), ds.subset(split.test)
    e = (lambda idx: None) if emb is None else (lambda idx: emb[idx])
    x_train = Inputs.from_dataset(train, e(split.train))
    x_val = Inputs.from_dataset(val, e(split.val))
    x_test = Inputs.from_dataset(test, e(split.test))

    cost_model = CostModel(ds.models).fit(train.prompt_chars, train.cost)
    c_hat_val = cost_model.predict(val.prompt_chars)
    c_hat_test = cost_model.predict(test.prompt_chars)
    s, w = ds.model_index(ref.strong), ds.model_index(ref.weak)

    results: list[RouterResult] = []

    # Oracle (true outcomes) and the zero router (single models, mixed).
    oracle = summarize(
        "oracle",
        val,
        test,
        oracle_decisions(val.quality, val.cost, LAMBDAS),
        oracle_decisions(test.quality, test.cost, LAMBDAS),
        LAMBDAS,
        ref,
        test.quality[:, s] - test.quality[:, w],
        0,
        seed,
    )
    results.append(oracle)
    prior = PriorQuality(ds.models).fit(x_train, train.quality, x_val, val.quality)
    results.append(
        summarize(
            "single-model",
            val,
            test,
            sweep(prior.predict(x_val), c_hat_val),
            sweep(prior.predict(x_test), c_hat_test),
            LAMBDAS,
            ref,
            np.random.default_rng(seed).random(len(test)),  # random 2-model split
            n_boot,
            seed,
        )
    )
    # The original project's length rule: one operating point; 2-model score = word count.
    tiers = tuple(ds.model_index(m) for m in ref.small_medium_large)
    length_val = length_tier_decisions(val.text, tiers)  # type: ignore[arg-type]
    length_test = length_tier_decisions(test.text, tiers)  # type: ignore[arg-type]
    results.append(
        summarize(
            "length-rule",
            val,
            test,
            length_val,
            length_test,
            np.array([0.0]),
            ref,
            np.array([len(t.split()) for t in test.text], dtype=np.float64),
            0,
            seed,
        )
    )

    preds: dict[str, tuple[FloatArray, FloatArray]] = {}
    for spec in specs:
        start = time.perf_counter()
        qm = spec.factory(ds.models).fit(x_train, train.quality, x_val, val.quality)
        q_val, q_test = qm.predict(x_val), qm.predict(x_test)
        fit_s = time.perf_counter() - start
        preds[spec.name] = (q_val, q_test)
        res = summarize(
            spec.name,
            val,
            test,
            sweep(q_val, c_hat_val),
            sweep(q_test, c_hat_test),
            LAMBDAS,
            ref,
            q_test[:, s] - q_test[:, w],
            n_boot,
            seed,
        )
        res.fit_seconds = round(fit_s, 1)
        res.extras["config"] = qm.config()
        res.extras["brier_test"] = round(float(np.mean((q_test - test.quality) ** 2)), 5)
        results.append(res)
        par = res.metrics["targets"]["0.95"].get("calibrated") or {}
        log(
            f"[{split.name}] {spec.name:12s} fit {fit_s:6.1f}s"
            f"  aiq_log {res.metrics['aiq_log']:.4f}"
            f"  95%-of-ref (cal) cost {par.get('cost_frac', float('nan')):.3f}"
            f" q/ref {par.get('quality_vs_ref', float('nan')):.4f}"
            f"  apgr {res.metrics['two_model']['apgr']:.3f}"
        )

    single_cost, single_quality = M.single_model_points(test.quality, test.cost)
    return {
        "split": split.name,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "reference": ref.__dict__,
        "single_models": {
            m: {"cost": float(c), "quality": round(float(q), 4)}
            for m, c, q in zip(ds.models, single_cost, single_quality, strict=True)
        },
        "routers": {
            res.name: {
                "metrics": res.metrics,
                "fit_seconds": res.fit_seconds,
                **res.extras,
            }
            for res in results
        },
        "_curves": {res.name: res.test_curve for res in results},
        "_two_model_curves": {
            res.name: res.two_model_curve for res in results if res.two_model_curve is not None
        },
        "_preds": preds,
    }
