from pathlib import Path

import numpy as np
import pytest

from modelrouter.data.dataset import RoutingDataset

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "routerbench_0shot"


def make_synthetic(n: int = 600, seed: int = 0) -> RoutingDataset:
    """Three models where difficulty is visible in the text.

    'easy' prompts are solved by every model, 'hard' ones only by the strong model; costs are
    1, 3 and 10 units.  A good router should approach the strong model's quality at well under
    its cost; a length heuristic can't, since length is independent of difficulty here.
    """
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.4
    fam = np.where(rng.random(n) < 0.5, "alpha", "beta")
    text = [
        ("hard question about " if h else "easy question about ") + f"topic {i % 17} " * (i % 5)
        for i, h in enumerate(hard)
    ]
    quality = np.zeros((n, 3))
    quality[:, 2] = (rng.random(n) < np.where(hard, 0.9, 0.95)).astype(float)
    quality[:, 1] = (rng.random(n) < np.where(hard, 0.3, 0.9)).astype(float)
    quality[:, 0] = (rng.random(n) < np.where(hard, 0.05, 0.85)).astype(float)
    cost = np.tile([1.0, 3.0, 10.0], (n, 1)) * rng.uniform(0.8, 1.2, (n, 1))
    return RoutingDataset(
        sample_id=np.array([f"s{i}" for i in range(n)]),
        family=fam,
        eval_name=fam,
        text=text,
        prompt_chars=np.array([len(t) for t in text]),
        models=("small", "medium", "large"),
        quality=quality,
        cost=cost,
    )


@pytest.fixture
def synthetic() -> RoutingDataset:
    return make_synthetic()


@pytest.fixture(scope="session")
def routerbench() -> RoutingDataset:
    if not (PROCESSED / "manifest.json").exists():
        pytest.skip("prepared RouterBench not present (run `modelrouter data prepare`)")
    return RoutingDataset.load(PROCESSED)
