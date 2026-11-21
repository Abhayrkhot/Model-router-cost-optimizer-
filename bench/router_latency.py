"""Per-request routing cost of a saved router, outside the gateway (one prompt at a time).

Measures what the router itself adds to a request: embedding (if the router needs one),
quality prediction, cost prediction and ranking.  Single-threaded by default so the number is
per-core CPU time, comparable across machines.

  python bench/router_latency.py --router artifacts/routers/knn-mlp-random-s0 --n 500
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from modelrouter.data.routerbench import router_text
from modelrouter.gateway.routing import NEEDS_EMBEDDING
from modelrouter.routers.base import Inputs, load_router, route


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--router", type=Path, required=True)
    p.add_argument("--fixture", type=Path, default=Path("artifacts/replay/random-s0-test.jsonl.gz"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--lam", type=float, default=31.6)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    with gzip.open(a.fixture, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for _, line in zip(range(a.n), fh, strict=False)]
    qm, cm, _ = load_router(a.router)
    embedder = None
    if qm.kind in NEEDS_EMBEDDING:
        from modelrouter.features.embed import Embedder

        embedder = Embedder(threads=a.threads)

    def decide(turns: list[str]) -> int:
        text = router_text(turns)
        chars = np.array([sum(len(t) for t in turns)], dtype=np.int64)
        emb = embedder.embed([text]) if embedder is not None else None
        q = qm.predict(Inputs([text], chars, emb))
        return int(route(q, cm.predict(chars), a.lam)[0])

    for r in rows[:20]:  # warm-up
        decide(r["turns"])
    ms = []
    for r in rows:
        t = time.perf_counter()
        decide(r["turns"])
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    res = {
        "router": qm.kind,
        "artifact_mb": round(sum(f.stat().st_size for f in a.router.iterdir()) / 1e6, 1),
        "threads": a.threads,
        "n": len(ms),
        "p50_ms": round(ms[len(ms) // 2], 3),
        "p90_ms": round(ms[int(len(ms) * 0.9)], 3),
        "p99_ms": round(ms[int(len(ms) * 0.99)], 3),
        "mean_ms": round(statistics.fmean(ms), 3),
        "loadavg_1m": round(os.getloadavg()[0], 1),
    }
    print(json.dumps(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
