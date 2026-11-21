"""`modelrouter` command line: data preparation, training, evaluation and serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare datasets.")
app.add_typer(data_app, name="data")
gateway_app = typer.Typer(no_args_is_help=True, help="Gateway fixtures and serving.")
app.add_typer(gateway_app, name="gateway")

DATA_DIR = Path("data")


@data_app.command("prepare")
def data_prepare(
    raw_dir: Annotated[Path, typer.Option()] = DATA_DIR / "raw",
    out_dir: Annotated[Path, typer.Option()] = DATA_DIR / "processed",
    download: Annotated[bool, typer.Option(help="Fetch the pinned files first.")] = True,
) -> None:
    """Download RouterBench + GSM8K gold, re-grade GSM8K, write data/processed/routerbench_0shot."""
    from modelrouter.data import routerbench

    if download:
        routerbench.download(raw_dir)
    manifest = routerbench.prepare(raw_dir, out_dir)
    typer.echo(
        json.dumps(
            {
                "rows_out": manifest.rows_out,
                "dropped": manifest.dropped,
                "families": manifest.families,
            },
            indent=2,
        )
    )


@app.command("embed")
def embed(
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    cache_dir: Annotated[Path, typer.Option()] = Path("artifacts") / "embeddings",
    threads: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Embed every prompt's router text once (cached by content hash)."""
    import time

    from modelrouter.data.dataset import RoutingDataset
    from modelrouter.features.embed import cached_embeddings

    ds = RoutingDataset.load(dataset)
    start = time.perf_counter()
    vecs = cached_embeddings(ds.text, cache_dir, threads=threads)
    typer.echo(f"{vecs.shape} in {time.perf_counter() - start:.0f}s")


def _router_specs(names: list[str]) -> list[Any]:
    from modelrouter.eval.harness import RouterSpec
    from modelrouter.routers.learned import EnsembleQuality, KNNQuality, MLPQuality, TfidfQuality

    factories: dict[str, Any] = {
        "tfidf": lambda m: TfidfQuality(m),
        "knn": lambda m: KNNQuality(m),
        "linear": lambda m: MLPQuality(m, hidden=0, lr=3e-3),
        "mlp": lambda m: MLPQuality(m),
        "knn+mlp": lambda m: EnsembleQuality(m, [KNNQuality(m), MLPQuality(m)]),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise typer.BadParameter(f"unknown routers {sorted(unknown)}; have {sorted(factories)}")
    return [RouterSpec(n, factories[n]) for n in names]


def _jsonable(obj: Any) -> Any:
    import math

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


FIGURE_ROUTER = ("knn+mlp", "Learned router (kNN + MLP)")


def _figures(res: dict[str, Any], out: Path) -> None:
    from modelrouter.eval import plots

    for mode in ("light", "dark"):
        plots.cost_quality(
            out / "figures" / f"cost_quality_{mode}.png",
            res["_curves"],
            FIGURE_ROUTER,
            res["single_models"],
            "RouterBench, held-out prompts: quality vs cost",
            mode,
        )
        plots.pgr_curve(
            out / "figures" / f"pgr_{mode}.png",
            res["_two_model_curves"],
            FIGURE_ROUTER,
            "GPT-4 Turbo",
            "Mixtral 8x7B",
            mode,
        )


@app.command("evaluate")
def evaluate(
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    cache_dir: Annotated[Path, typer.Option()] = Path("artifacts") / "embeddings",
    out: Annotated[Path, typer.Option()] = Path("results"),
    seeds: Annotated[str, typer.Option(help="Random-split seeds.")] = "0,1,2",
    routers: Annotated[str, typer.Option()] = "tfidf,knn,linear,mlp,knn+mlp",
    lofo: Annotated[bool, typer.Option(help="Also run leave-one-benchmark-out.")] = True,
    n_boot: Annotated[int, typer.Option()] = 1000,
    threads: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Train and score every router on random splits (and held-out benchmarks)."""
    import torch

    from modelrouter.data.dataset import RoutingDataset
    from modelrouter.data.routerbench import FAMILIES
    from modelrouter.data.splits import leave_family_out, random_split
    from modelrouter.eval.harness import run_split
    from modelrouter.features.embed import cached_embeddings

    if threads:
        torch.set_num_threads(threads)
    ds = RoutingDataset.load(dataset)
    emb = cached_embeddings(ds.text, cache_dir, threads=threads)
    specs = _router_specs(routers.split(","))
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, Any]] = [
        (f"random-s{s}", random_split(ds, seed=int(s))) for s in seeds.split(",")
    ]
    if lofo:
        # MT-Bench has 80 prompts: too few to score a held-out benchmark on.
        runs += [(f"lofo-{f}", leave_family_out(ds, f)) for f in FAMILIES if f != "mtbench"]
    for name, split in runs:
        res = run_split(ds, emb, split, specs, n_boot=n_boot if name.startswith("random") else 200)
        (out / f"{name}.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
        if name == "random-s0" and FIGURE_ROUTER[0] in res["_curves"]:
            _figures(res, out)
        typer.echo(f"wrote {out / name}.json")


def _split_by_name(ds: Any, name: str) -> Any:
    from modelrouter.data.splits import leave_family_out, random_split

    if name.startswith("random-s"):
        return random_split(ds, seed=int(name.removeprefix("random-s")))
    if name.startswith("lofo-"):
        return leave_family_out(ds, name.removeprefix("lofo-"))
    raise typer.BadParameter(f"unknown split {name!r} (random-s<N> or lofo-<family>)")


@app.command("figures")
def figures(
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    cache_dir: Annotated[Path, typer.Option()] = Path("artifacts") / "embeddings",
    out: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """Re-draw the README figures from the random-s0 split (fits only the plotted router)."""
    from modelrouter.data.dataset import RoutingDataset
    from modelrouter.data.splits import random_split
    from modelrouter.eval.harness import run_split
    from modelrouter.features.embed import cached_embeddings

    ds = RoutingDataset.load(dataset)
    emb = cached_embeddings(ds.text, cache_dir)
    res = run_split(ds, emb, random_split(ds, 0), _router_specs([FIGURE_ROUTER[0]]), n_boot=0)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _figures(res, out)
    typer.echo(f"wrote {out / 'figures'}")


@app.command("figures-load")
def figures_load(
    results: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """Draw the outage figure from results/load/outage-gpt4*.json (bench workflow output)."""
    from modelrouter.eval import plots

    with_fb = json.loads((results / "load" / "outage-gpt4.json").read_text())
    without = json.loads((results / "load" / "outage-gpt4-nofallback.json").read_text())
    fault = with_fb["config"]["fault"][0]
    for mode in ("light", "dark"):
        plots.outage_timeline(
            results / "figures" / f"outage_{mode}.png",
            [("With fallback", with_fb), ("Without fallback", without)],
            (fault["start"], fault["end"]),
            "GPT-4 Turbo outage at 100 req/s (λ at the 95%-of-GPT-4 point)",
            mode,
        )
    typer.echo(f"wrote {results / 'figures'}/outage_*.png")


@app.command("report")
def report(
    results: Annotated[Path, typer.Option()] = Path("results"),
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    out: Annotated[Path, typer.Option()] = Path("docs") / "RESULTS.md",
) -> None:
    """Write docs/RESULTS.md and results/summary.json from results/*.json."""
    from modelrouter.eval.report import build

    text, summary = build(results, dataset / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (results / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1) + "\n")
    typer.echo(f"wrote {out} and {results / 'summary.json'}")


@app.command("train")
def train(
    router: Annotated[str, typer.Option()] = "knn+mlp",
    split: Annotated[str, typer.Option()] = "random-s0",
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    cache_dir: Annotated[Path, typer.Option()] = Path("artifacts") / "embeddings",
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Fit one router on a split's train set (val for model selection) and save it."""
    from modelrouter.data.dataset import RoutingDataset
    from modelrouter.features.embed import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, cached_embeddings
    from modelrouter.routers.base import CostModel, Inputs, save_router

    ds = RoutingDataset.load(dataset)
    # Lexical routers need no embeddings (and the gateway then loads no embedding model).
    emb = None if router == "tfidf" else cached_embeddings(ds.text, cache_dir)
    sp = _split_by_name(ds, split)
    spec = _router_specs([router])[0]
    tr, va = ds.subset(sp.train), ds.subset(sp.val)
    qm = spec.factory(ds.models).fit(
        Inputs.from_dataset(tr, None if emb is None else emb[sp.train]),
        tr.quality,
        Inputs.from_dataset(va, None if emb is None else emb[sp.val]),
        va.quality,
    )
    cm = CostModel(ds.models).fit(tr.prompt_chars, tr.cost)
    target = out or Path("artifacts") / "routers" / f"{router.replace('+', '-')}-{split}"
    meta = {
        "router": router,
        "split": split,
        "train_rows": len(tr),
        "embedding_model": DEFAULT_MODEL,
        "embedding_max_tokens": DEFAULT_MAX_TOKENS,
        "dataset_manifest": json.loads((dataset / "manifest.json").read_text())["source"],
    }
    save_router(target, qm, cm, meta)
    typer.echo(f"saved {target}")


@gateway_app.command("build-replay")
def build_replay(
    split: Annotated[str, typer.Option()] = "random-s0",
    part: Annotated[str, typer.Option(help="train | val | test")] = "test",
    dataset: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "routerbench_0shot",
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Write the replay fixture (recorded responses) for one part of a split."""
    import gzip

    import pandas as pd

    from modelrouter.data.dataset import RoutingDataset

    ds = RoutingDataset.load(dataset)
    idx = getattr(_split_by_name(ds, split), part)
    routing = pd.read_parquet(dataset / "routing.parquet").iloc[idx]
    responses = pd.read_parquet(dataset / "responses.parquet").iloc[idx]
    target = out or Path("artifacts") / "replay" / f"{split}-{part}.jsonl.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as fh:
        for (_, r), (_, resp) in zip(routing.iterrows(), responses.iterrows(), strict=True):
            row = {
                "sample_id": r["sample_id"],
                "family": r["family"],
                "turns": json.loads(r["turns_json"]),
                "models": {
                    m: {"text": resp[f"r__{m}"], "cost": r[f"c__{m}"], "quality": r[f"q__{m}"]}
                    for m in ds.models
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"wrote {len(idx)} prompts to {target}")


@app.command("serve")
def serve(
    config: Annotated[Path, typer.Option(envvar="MODELROUTER_CONFIG")] = Path(
        "deploy/gateway.yaml"
    ),
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8080,
    log_level: Annotated[str, typer.Option()] = "warning",
    max_attempts: Annotated[
        int | None, typer.Option(help="Override max_attempts (1 disables fallback).")
    ] = None,
    router_dir: Annotated[Path | None, typer.Option(help="Override router_dir.")] = None,
) -> None:
    """Run the gateway (one process; scale out with replicas)."""
    import os

    import uvicorn

    from modelrouter.gateway.app import create_app
    from modelrouter.gateway.config import GatewayConfig

    cfg = GatewayConfig.from_yaml(config)
    if os.environ.get("MODELROUTER_REDIS_URL"):
        cfg.redis_url = os.environ["MODELROUTER_REDIS_URL"]
    if max_attempts is not None:
        cfg.max_attempts = max_attempts
    if router_dir is not None:
        cfg.router_dir = router_dir.resolve()
    uvicorn.run(create_app(cfg), host=host, port=port, log_level=log_level, access_log=False)


if __name__ == "__main__":
    app()
