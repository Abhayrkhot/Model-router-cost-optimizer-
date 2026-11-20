import numpy as np
import pytest

from modelrouter.eval import metrics as M


def _curve(cost, quality):
    return M.Curve("t", np.asarray(cost, float), np.asarray(quality, float), np.arange(len(cost)))


def test_upper_hull_drops_dominated_and_concave_violations():
    # (2, 0.4) is dominated by (1, 0.5); (3, 0.6) lies under the chord (1,.5)-(4,.9).
    hc, hq = M.upper_hull(np.array([1, 2, 3, 4, 5.0]), np.array([0.5, 0.4, 0.6, 0.9, 0.85]))
    np.testing.assert_allclose(hc, [1, 4])
    np.testing.assert_allclose(hq, [0.5, 0.9])


def test_hull_is_non_decreasing_and_flat_after_the_top():
    c = _curve([1, 2, 10], [0.5, 0.8, 0.7])
    hc, hq = M.upper_hull(c.cost, c.quality)
    assert np.all(np.diff(hq) >= 0)
    np.testing.assert_allclose(M.hull_quality_at(hc, hq, np.array([1.5, 50.0])), [0.65, 0.8])


def test_aiq_of_a_flat_curve_is_its_level():
    c = _curve([1, 10], [0.7, 0.7])
    assert M.aiq(c, 1, 10) == pytest.approx(0.7)
    assert M.aiq(c, 1, 10, log=True) == pytest.approx(0.7)


def test_point_metrics():
    c = _curve([1, 2, 5], [0.5, 0.7, 0.9])
    assert M.min_cost_reaching(c, 0.7) == (2.0, 1.0)
    assert M.min_cost_reaching(c, 0.95)[0] == float("inf")
    assert M.best_quality_within(c, 4.9) == (0.7, 1.0)
    assert np.isnan(M.best_quality_within(c, 0.5)[0])


def test_random_scores_give_apgr_near_half_and_perfect_scores_beat_it():
    rng = np.random.default_rng(0)
    n = 4000
    q = np.column_stack([rng.random(n) < 0.5, rng.random(n) < 0.8]).astype(float)
    c = np.tile([1.0, 10.0], (n, 1))
    rand = M.strong_share_curve(q, c, strong=1, weak=0, score=rng.random(n))
    q_w, q_s = rand.quality[0], rand.quality[-1]
    assert M.apgr(rand, q_w, q_s) == pytest.approx(0.5, abs=0.03)
    oracle = M.strong_share_curve(q, c, 1, 0, score=q[:, 1] - q[:, 0])
    assert M.apgr(oracle, q_w, q_s) > 0.8
    assert M.cpt(oracle, q_w, q_s, 0.8) < M.cpt(rand, q_w, q_s, 0.8)


def test_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    q_rows = (rng.random((1, 2000)) < 0.6).astype(float)
    c_rows = np.ones_like(q_rows)
    lo, hi = M.bootstrap(c_rows, q_rows, lambda mc, mq: float(mq[0]), n_boot=300)
    assert lo < q_rows.mean() < hi
    assert hi - lo < 0.06


def test_hull_cost_reaching_interpolates_between_hull_points():
    c = _curve([1, 3], [0.5, 0.9])
    assert M.hull_cost_reaching(c, 0.7) == pytest.approx(2.0)
    assert M.hull_cost_reaching(c, 0.4) == 1.0
    assert M.hull_cost_reaching(c, 0.95) == float("inf")
