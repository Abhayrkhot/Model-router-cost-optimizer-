import numpy as np
import pytest

from modelrouter.data.splits import random_split
from modelrouter.eval.harness import Reference, RouterSpec, run_split
from modelrouter.routers.base import CostModel, Inputs, load_router, route, save_router, sweep
from modelrouter.routers.baselines import PriorQuality, length_tier_decisions
from modelrouter.routers.learned import EnsembleQuality, KNNQuality, MLPQuality, TfidfQuality


def synthetic_embedding(ds, seed=0):
    """Embeddings that separate easy from hard prompts (plus noise), like a real encoder would."""
    rng = np.random.default_rng(seed)
    hard = np.array([t.startswith("hard") for t in ds.text], dtype=np.float32)
    e = rng.normal(0, 0.3, (len(ds), 16)).astype(np.float32)
    e[:, 0] += 2 * hard - 1
    return e / np.linalg.norm(e, axis=1, keepdims=True)


@pytest.fixture
def split_inputs(synthetic):
    emb = synthetic_embedding(synthetic)
    split = random_split(synthetic, seed=0)
    parts = {}
    for name in ("train", "val", "test"):
        idx = getattr(split, name)
        sub = synthetic.subset(idx)
        parts[name] = (Inputs.from_dataset(sub, emb[idx]), sub.quality, sub)
    return synthetic, emb, split, parts


@pytest.mark.parametrize(
    "factory",
    [
        lambda m: KNNQuality(m),
        lambda m: MLPQuality(m, hidden=32, epochs=40, lr=3e-3, batch_size=32),
        lambda m: MLPQuality(m, hidden=0, lr=1e-2, epochs=60, batch_size=32),
        lambda m: TfidfQuality(m),
        lambda m: EnsembleQuality(
            m, [KNNQuality(m), MLPQuality(m, hidden=32, epochs=40, lr=3e-3, batch_size=32)]
        ),
    ],
)
def test_learned_models_beat_the_prior_on_brier(split_inputs, factory):
    ds, _, _, parts = split_inputs
    (xt, qt, _), (xv, qv, _), (xs, qs, _) = parts["train"], parts["val"], parts["test"]
    prior = PriorQuality(ds.models).fit(xt, qt, xv, qv).predict(xs)
    learned = factory(ds.models).fit(xt, qt, xv, qv).predict(xs)
    assert learned.shape == qs.shape
    assert np.all((learned >= 0) & (learned <= 1))
    assert np.mean((learned - qs) ** 2) < 0.8 * np.mean((prior - qs) ** 2)


def test_mlp_numpy_forward_matches_torch(split_inputs):
    import torch

    ds, _, _, parts = split_inputs
    (xt, qt, _), (xv, qv, _) = parts["train"], parts["val"]
    model = MLPQuality(ds.models, hidden=16, epochs=5).fit(xt, qt, xv, qv)
    net = torch.nn.Sequential(
        torch.nn.Linear(17, 16),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(16, 3),
    ).eval()
    net.load_state_dict({k.replace("_", "."): torch.from_numpy(v) for k, v in model.params.items()})
    with torch.no_grad():
        ref = torch.sigmoid(net(torch.from_numpy(MLPQuality._features(xv)))).numpy()
    np.testing.assert_allclose(model.predict(xv), ref, atol=1e-5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda m: KNNQuality(m),
        lambda m: MLPQuality(m, hidden=8, epochs=3),
        lambda m: TfidfQuality(m),
        lambda m: EnsembleQuality(m, [KNNQuality(m), MLPQuality(m, hidden=8, epochs=3)]),
    ],
)
def test_router_round_trips_through_disk(tmp_path, split_inputs, factory):
    ds, _, _, parts = split_inputs
    (xt, qt, sub), (xv, qv, _) = parts["train"], parts["val"]
    qm = factory(ds.models).fit(xt, qt, xv, qv)
    cm = CostModel(ds.models).fit(sub.prompt_chars, sub.cost)
    save_router(tmp_path / "r", qm, cm, {"note": "test"})
    qm2, cm2, meta = load_router(tmp_path / "r")
    np.testing.assert_allclose(qm2.predict(xv), qm.predict(xv), atol=1e-6)
    np.testing.assert_allclose(cm2.predict(sub.prompt_chars), cm.predict(sub.prompt_chars))
    assert meta == {"note": "test"}


def test_policy_lambda_moves_from_best_to_cheapest():
    q = np.array([[0.2, 0.9], [0.5, 0.6]])
    c = np.array([[1.0, 10.0], [1.0, 10.0]])
    assert route(q, c, 0.0).tolist() == [1, 1]
    assert route(q, c, 1.0).tolist() == [0, 0]
    assert route(q, c, 0.05).tolist() == [1, 0]
    assert sweep(q, c, np.array([0.0, 1.0])).shape == (2, 2)


def test_cost_model_is_linear_in_length_and_floored():
    chars = np.array([100, 200, 300, 400])
    cost = np.column_stack([1e-4 + 1e-6 * chars, 2e-4 + 0 * chars])
    cm = CostModel(("a", "b")).fit(chars, cost)
    np.testing.assert_allclose(cm.predict(np.array([250]))[0], [3.5e-4, 2e-4], rtol=1e-6)
    assert cm.predict(np.array([0]))[0, 0] >= cost[:, 0].min()


def test_length_rule_tiers():
    texts = ["short", "word " * 30, "word " * 80]
    assert length_tier_decisions(texts, (0, 1, 2)).tolist() == [[0, 1, 2]]


def test_run_split_end_to_end(synthetic):
    emb = synthetic_embedding(synthetic)
    split = random_split(synthetic, seed=0)
    specs = [RouterSpec("knn", lambda m: KNNQuality(m))]
    ref = Reference("large", "large", "medium", ("small", "medium", "large"))
    res = run_split(synthetic, emb, split, specs, ref=ref, n_boot=50, log=lambda s: None)
    knn = res["routers"]["knn"]["metrics"]
    single = res["routers"]["single-model"]["metrics"]
    oracle = res["routers"]["oracle"]["metrics"]
    # Difficulty is visible in the text, so routing must beat mixing single models...
    assert knn["aiq_log"] > single["aiq_log"] + 0.02
    assert knn["two_model"]["apgr"] > 0.7
    # ...and can't beat the oracle.
    assert knn["aiq_log"] <= oracle["aiq_log"] + 1e-9
    par = knn["targets"]["0.95"]["calibrated"]
    assert par is not None
    assert par["cost_frac"] < 0.9
    lo, hi = knn["ci95"]["cost_frac_at_95pct_quality"]
    assert lo <= knn["targets"]["0.95"]["cost_frac_curve"] <= hi


def test_saved_router_loads_in_a_fresh_interpreter(tmp_path, split_inputs):
    """The gateway loads artifacts without importing the router modules itself."""
    import subprocess
    import sys

    ds, _, _, parts = split_inputs
    (xt, qt, sub), (xv, qv, _) = parts["train"], parts["val"]
    qm = EnsembleQuality(
        ds.models, [KNNQuality(ds.models), MLPQuality(ds.models, hidden=8, epochs=2)]
    )
    qm.fit(xt, qt, xv, qv)
    save_router(tmp_path / "r", qm, CostModel(ds.models).fit(sub.prompt_chars, sub.cost), {})
    code = (
        "from pathlib import Path; from modelrouter.routers.base import load_router; "
        f"qm, _, _ = load_router(Path({str(tmp_path / 'r')!r})); print(qm.kind)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ensemble"


def test_tfidf_numpy_transform_matches_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer

    from modelrouter.routers.learned import _TfidfTransform

    docs = [
        "Write a Python function to reverse a linked list.",
        "What is the capital of Australia? A) Sydney B) Canberra",
        "Natalia sold clips to 48 of her friends in April, and half as many in May.",
        "Ünïcödé wörds and CamelCase tokens, plus repeated repeated repeated words.",
    ] * 3
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(docs)
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.__getitem__)
    ours = _TfidfTransform(terms, vec.idf_)
    ref = vec.transform(docs).toarray()
    for i, d in enumerate(docs):
        idx, val = ours.row(d)
        dense = np.zeros(len(terms))
        dense[idx] = val
        np.testing.assert_allclose(dense, ref[i], atol=1e-12)
