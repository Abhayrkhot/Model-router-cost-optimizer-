"""Offline decisions of a saved router at one λ, for comparison with a gateway replay run.

  python bench/offline_point.py --router artifacts/routers/knn-mlp-random-s0 --split random-s0 \
      --lam 300 --out results/e2e/offline-lam300.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modelrouter.data.dataset import RoutingDataset
from modelrouter.data.splits import random_split
from modelrouter.features.embed import cached_embeddings
from modelrouter.routers.base import Inputs, load_router, route


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--router", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/processed/routerbench_0shot"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/embeddings"))
    p.add_argument("--split", default="random-s0")
    p.add_argument("--lam", type=float, default=300.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    ds = RoutingDataset.load(a.dataset)
    emb = cached_embeddings(ds.text, a.cache)
    split = random_split(ds, int(a.split.removeprefix("random-s")))
    test = ds.subset(split.test)
    qm, cm, _ = load_router(a.router)
    q_hat = qm.predict(Inputs.from_dataset(test, emb[split.test]))
    c_hat = cm.predict(test.prompt_chars)
    choice = route(q_hat, c_hat, a.lam)
    rows = np.arange(len(test))
    out = {
        "lambda": a.lam,
        "n": len(test),
        "quality": float(test.quality[rows, choice].mean()),
        "cost_per_1k_usd": float(1000 * test.cost[rows, choice].mean()),
        "choices": {sid: test.models[c] for sid, c in zip(test.sample_id, choice, strict=True)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "choices"}))


if __name__ == "__main__":
    main()
