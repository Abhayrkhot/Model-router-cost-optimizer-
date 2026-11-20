"""Compare a gateway replay run (loadgen --dump) with offline decisions (offline_point.py)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offline", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    off = json.loads(a.offline.read_text())
    runs = [json.loads(line) for line in a.dump.read_text().splitlines()]
    ok = [r for r in runs if r["status"] == 200]
    agree = sum(1 for r in ok if off["choices"].get(r["sample_id"]) == r["router_model"])
    res = {
        "requests": len(runs),
        "ok": len(ok),
        "decision_agreement": round(agree / max(1, len(ok)), 5),
        "gateway_quality": round(statistics.fmean(r["quality"] for r in ok), 5),
        "offline_quality": round(off["quality"], 5),
        "gateway_cost_per_1k_usd": round(1000 * statistics.fmean(r["cost"] for r in ok), 5),
        "offline_cost_per_1k_usd": round(off["cost_per_1k_usd"], 5),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
