import json

from modelrouter.cli import _jsonable
from modelrouter.data.splits import leave_family_out, random_split
from modelrouter.eval import report
from modelrouter.eval.harness import Reference, RouterSpec, run_split
from modelrouter.routers.learned import KNNQuality
from test_routers import synthetic_embedding


def test_report_builds_every_section_from_harness_output(tmp_path, synthetic, monkeypatch):
    # The report's labels assume RouterBench's router names; the synthetic run uses the same keys.
    emb = synthetic_embedding(synthetic)
    ref = Reference("large", "large", "medium", ("small", "medium", "large"))
    specs = [RouterSpec("knn+mlp", lambda m: KNNQuality(m))]
    results = tmp_path / "results"
    results.mkdir()
    for seed in (0, 1):
        res = run_split(
            synthetic, emb, random_split(synthetic, seed), specs, ref, n_boot=20, log=lambda s: None
        )
        (results / f"random-s{seed}.json").write_text(json.dumps(_jsonable(res)))
    res = run_split(
        synthetic,
        emb,
        leave_family_out(synthetic, "alpha"),
        specs,
        ref,
        n_boot=0,
        log=lambda s: None,
    )
    (results / "lofo-alpha.json").write_text(json.dumps(_jsonable(res)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows_out": len(synthetic),
                "models": list(synthetic.models),
                "source": {"routerbench_revision": "abcdef0123"},
                "dropped": {},
                "families": {"alpha": 300, "beta": 300},
                "gsm8k_regrade": {
                    "per_model": {"large": {"orig_mean": 0.5, "exact_match_acc": 0.9}}
                },
            }
        )
    )
    monkeypatch.setattr(report, "ROUTERS", ["knn+mlp"])
    text, summary = report.build(results, manifest)
    for heading in (
        "## Data",
        "## Quality at a fixed budget",
        "## Cost to reach",
        "## Two models",
        "## Router comparison",
        "## Unseen benchmarks",
    ):
        assert heading in text
    m, sd = summary["quality_at_budget_hull"]["knn+mlp"]["0.5"]
    assert 0 <= m <= 1
    assert sd >= 0
    assert "alpha" in summary["lofo"]
