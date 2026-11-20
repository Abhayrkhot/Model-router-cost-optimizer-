"""Open-loop load generator for the gateway, scored against the replay fixture.

Requests are sent on a fixed schedule (constant or Poisson arrivals) regardless of how fast the
gateway answers, and each latency is measured from the request's *scheduled* start, so queueing
inside the gateway shows up instead of being hidden (no coordinated omission).

Every response names the router model that answered; the fixture holds that model's recorded
quality and cost for the prompt, so a run reports end-to-end quality and spend as well as
latency.  Optional fault windows flip a deployment's injected failure rate mid-run through the
gateway's /admin/faults endpoint.

  python bench/loadgen.py --url http://127.0.0.1:8080 --key sk-bench-local \\
      --fixture artifacts/replay/random-s0-test.jsonl.gz --rate 200 --duration 60 \\
      --fault gpt-4-1106-preview=1.0@20:40 --out results/load/chaos.json
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Result:
    sample_id: str
    t_sched: float
    latency_s: float
    status: int
    router_model: str | None = None
    deployment: str | None = None
    cost: float = 0.0
    quality: float | None = None
    router_ms: float | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class Fault:
    deployment: str
    rate: float
    start: float
    end: float


def parse_fault(spec: str) -> Fault:
    dep, rest = spec.split("=", 1)
    rate, window = rest.split("@", 1)
    start, end = window.split(":", 1)
    return Fault(dep, float(rate), float(start), float(end))


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


@dataclass
class Run:
    args: argparse.Namespace
    rows: list[dict[str, Any]]
    results: list[Result] = field(default_factory=list)

    async def one(self, client: httpx.AsyncClient, row: dict[str, Any], t0: float, t_s: float):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": t} for t in row["turns"]],
            "max_tokens": 512,
            "router": {"lambda": self.args.lam},
        }
        res = Result(sample_id=row["sample_id"], t_sched=t_s, latency_s=0.0, status=0)
        try:
            r = await client.post("/v1/chat/completions", json=body)
            res.status = r.status_code
            if r.status_code == 200:
                data = r.json()
                routing = data["routing"]
                res.router_model = routing["router_model"]
                res.deployment = routing["deployment"]
                res.cost = float(routing["cost_usd"])
                res.router_ms = float(routing.get("router_ms") or 0.0)
                res.attempts = len(routing.get("attempts") or [])
                rec = row["models"].get(res.router_model or "")
                res.quality = None if rec is None else float(rec["quality"])
            else:
                res.error = r.json().get("error", {}).get("type")
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            res.error = type(exc).__name__
        res.latency_s = time.perf_counter() - (t0 + t_s)
        self.results.append(res)

    async def faults(self, client: httpx.AsyncClient, t0: float) -> None:
        events = []
        for f in self.args.fault:
            events += [(f.start, f.deployment, f.rate), (f.end, f.deployment, 0.0)]
        for at, dep, rate in sorted(events):
            await asyncio.sleep(max(0.0, t0 + at - time.perf_counter()))
            await client.post("/admin/faults", json={"deployment": dep, "failure_rate": rate})

    async def main(self) -> None:
        a = self.args
        rng = random.Random(a.seed)
        order = list(range(len(self.rows)))
        rng.shuffle(order)
        n = int(a.rate * a.duration) if not a.concurrency else len(order)
        gaps = [rng.expovariate(a.rate) if a.poisson else 1.0 / a.rate for _ in range(n)]
        limits = httpx.Limits(
            max_connections=a.connections, max_keepalive_connections=a.connections
        )
        headers = {"Authorization": f"Bearer {a.key}"}
        async with httpx.AsyncClient(
            base_url=a.url, headers=headers, timeout=a.timeout, limits=limits
        ) as client:
            # warm-up (not recorded)
            for i in range(min(20, len(order))):
                await Run(a, self.rows).one(client, self.rows[order[i]], time.perf_counter(), 0.0)
            t0 = time.perf_counter() + 0.2
            if a.concurrency:
                # Closed loop: N workers, each sends its next request when the last one returns.
                # For correctness runs (every prompt exactly once); latency is per-request.
                queue = [self.rows[i] for i in order[: (n or len(order))]]

                async def worker() -> None:
                    while queue:
                        row = queue.pop()
                        await self.one(client, row, time.perf_counter(), 0.0)

                await asyncio.gather(*(worker() for _ in range(a.concurrency)))
                self.wall = time.perf_counter() - t0
                return
            fault_task = asyncio.create_task(self.faults(client, t0)) if a.fault else None
            tasks = []
            t_s = 0.0
            for i in range(n):
                t_s += gaps[i]
                await asyncio.sleep(max(0.0, t0 + t_s - time.perf_counter()))
                row = self.rows[order[i % len(order)]]
                tasks.append(asyncio.create_task(self.one(client, row, t0, t_s)))
            await asyncio.gather(*tasks)
            if fault_task:
                await fault_task
            self.wall = time.perf_counter() - t0

    def summary(self) -> dict[str, Any]:
        ok = [r for r in self.results if r.status == 200]
        lat = [r.latency_s * 1000 for r in self.results]
        lat_ok = [r.latency_s * 1000 for r in ok]
        qualities = [r.quality for r in ok if r.quality is not None]
        share = Counter(r.router_model for r in ok)
        timeline: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"sent": 0, "ok": 0, "errors": 0, "lat": [], "by_deployment": Counter()}
        )
        for r in self.results:
            b = timeline[int(r.t_sched)]
            b["sent"] += 1
            b["lat"].append(r.latency_s * 1000)
            if r.status == 200:
                b["ok"] += 1
                b["by_deployment"][r.deployment] += 1
            else:
                b["errors"] += 1
        return {
            "config": {
                k: (v if not isinstance(v, list) else [f.__dict__ for f in v])
                for k, v in vars(self.args).items()
                if k not in {"key"}
            },
            "requests": len(self.results),
            "wall_s": round(self.wall, 2),
            "achieved_rps": round(len(self.results) / self.wall, 1),
            "ok": len(ok),
            "success_rate": round(len(ok) / max(1, len(self.results)), 5),
            "errors": dict(
                Counter(r.error or str(r.status) for r in self.results if r.status != 200)
            ),
            "latency_ms": {
                "p50": round(pct(lat, 50), 2),
                "p90": round(pct(lat, 90), 2),
                "p99": round(pct(lat, 99), 2),
                "p999": round(pct(lat, 99.9), 2),
                "max": round(max(lat), 2) if lat else None,
                "mean_ok": round(statistics.fmean(lat_ok), 2) if lat_ok else None,
            },
            "router_ms": {
                "p50": round(pct([r.router_ms or 0 for r in ok], 50), 3),
                "p99": round(pct([r.router_ms or 0 for r in ok], 99), 3),
            },
            "quality": round(statistics.fmean(qualities), 4) if qualities else None,
            "cost_per_1k_usd": round(1000 * statistics.fmean([r.cost for r in ok]), 4)
            if ok
            else None,
            "fallback_served": sum(1 for r in ok if r.attempts > 1),
            "model_share": {m: round(c / len(ok), 4) for m, c in share.most_common()},
            "timeline": [
                {
                    "t": t,
                    "sent": b["sent"],
                    "ok": b["ok"],
                    "errors": b["errors"],
                    "p99_ms": round(pct(b["lat"], 99), 1),
                    "by_deployment": dict(b["by_deployment"]),
                }
                for t, b in sorted(timeline.items())
            ],
            "loadavg_1m_end": os.getloadavg()[0],
        }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--key", default=os.environ.get("MODELROUTER_KEY", "sk-bench-local"))
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--rate", type=float, default=100.0, help="requests per second")
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    p.add_argument("--poisson", action="store_true", help="exponential inter-arrival gaps")
    p.add_argument("--lam", type=float, default=300.0, help="router lambda sent with each request")
    p.add_argument("--connections", type=int, default=512)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fault",
        type=parse_fault,
        action="append",
        default=[],
        help="deployment=rate@start:end (seconds into the run)",
    )
    p.add_argument("--limit", type=int, default=0, help="use only the first N fixture prompts")
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="closed loop with N workers instead of open-loop arrivals",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--dump", type=Path, help="write one JSON line per request")
    args = p.parse_args()

    with gzip.open(args.fixture, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if args.limit:
        rows = rows[: args.limit]
    run = Run(args, rows)
    asyncio.run(run.main())
    out = run.summary()
    brief = {k: v for k, v in out.items() if k not in {"timeline", "config"}}
    print(json.dumps(brief, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as fh:
            for r in run.results:
                fh.write(json.dumps(r.__dict__) + "\n")


if __name__ == "__main__":
    main()
