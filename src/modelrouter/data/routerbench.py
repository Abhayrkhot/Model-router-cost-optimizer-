"""Download, verify and normalise RouterBench into the routing dataset used everywhere else.

RouterBench (withmartian/routerbench) holds ~36.5k prompts from MMLU, HellaSwag, GSM8K, ARC,
Winogrande, MBPP, MT-Bench and a set of OpenAI-evals tasks, each answered by the same 11 models
with a quality score in [0, 1] and the dollar cost of that answer.  Every prompt has an outcome
for every model, so any routing decision can be scored after the fact.

Output (``data/processed/<name>/``):
  routing.parquet    one row per prompt: ids, benchmark family, router text, prompt length,
                     q__<model> quality, q_orig__<model> RouterBench's own score, c__<model> cost
  responses.parquet  sample_id + r__<model>: the final-turn answer text (used by gateway replay)
  manifest.json      source revisions and checksums, row counts, GSM8K re-grading stats
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from modelrouter.data import gsm8k
from modelrouter.data.safe_pickle import load_dataframe

REPO_ID = "withmartian/routerbench"
REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
FILES = {
    "routerbench_0shot.pkl": "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
}
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILES = ("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet")

# RouterBench column name -> short id used in this project.
MODEL_IDS: dict[str, str] = {
    "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
    "claude-instant-v1": "claude-instant-v1",
    "claude-v1": "claude-v1",
    "claude-v2": "claude-v2",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview": "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat": "codellama-34b",
    "meta/llama-2-70b-chat": "llama-2-70b",
    "mistralai/mistral-7b-chat": "mistral-7b",
    "mistralai/mixtral-8x7b-chat": "mixtral-8x7b",
    "zero-one-ai/Yi-34B-Chat": "yi-34b",
}

FAMILIES = ("mmlu", "hellaswag", "gsm8k", "arc", "winogrande", "mbpp", "mtbench", "misc")
_FAMILY_OF = {
    "hellaswag": "hellaswag",
    "grade-school-math": "gsm8k",
    "arc-challenge": "arc",
    "winogrande": "winogrande",
    "mbpp": "mbpp",
    "mtbench": "mtbench",
    "mtbench-math": "mtbench",
    "mtbench-reference": "mtbench",
}
# RouterBench's own smoke-test eval (3 rows); not a benchmark.
_DROP_EVALS = frozenset({"test-match"})
ROUTER_TEXT_HEAD_CHARS = 200


def family_of(eval_name: str) -> str:
    if eval_name.startswith("mmlu-"):
        return "mmlu"
    return _FAMILY_OF.get(eval_name, "misc")


def parse_turns(raw: str) -> list[str]:
    """RouterBench stores conversations as the repr of a list of strings."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]
    if isinstance(value, list | tuple) and all(isinstance(t, str) for t in value) and value:
        return list(value)
    return [raw]


def router_text(turns: list[str]) -> str:
    """Text the router sees: the last turn, prefixed by the head of the first turn.

    Multi-turn prompts put a shared instruction (GSM8K's few-shot block, consensus_summary's
    rules) or the earlier question (MT-Bench) first; the last turn is what varies.  Embedding
    models truncate from the end, so the varying part must not sit behind a long shared prefix.
    """
    if len(turns) == 1:
        return turns[0]
    return turns[0][:ROUTER_TEXT_HEAD_CHARS] + "\n\n" + turns[-1]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir: Path, name: str = "routerbench_0shot.pkl") -> dict[str, Path]:
    """Fetch the pinned RouterBench pickle and GSM8K gold answers; verify the pickle checksum."""
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    pkl = Path(
        hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION, local_dir=raw_dir)
    )
    actual = sha256_of(pkl)
    if actual != FILES[name]:
        raise RuntimeError(f"{name}: sha256 {actual} != pinned {FILES[name]}")
    paths = {"routerbench": pkl}
    for fname in GSM8K_FILES:
        paths[fname] = Path(
            hf_hub_download(
                GSM8K_REPO,
                fname,
                repo_type="dataset",
                revision=GSM8K_REVISION,
                local_dir=raw_dir / "gsm8k",
            )
        )
    return paths


@dataclass
class Manifest:
    source: dict[str, str]
    rows_in: int
    rows_out: int
    dropped: dict[str, int]
    models: list[str]
    families: dict[str, int]
    gsm8k_regrade: dict[str, Any] = field(default_factory=dict)


def _gsm8k_gold(paths: list[Path]) -> dict[str, Any]:
    import pandas as pd

    gold: dict[str, Any] = {}
    for p in paths:
        for q, a in pd.read_parquet(p, columns=["question", "answer"]).itertuples(index=False):
            gold[gsm8k.normalize_question(q)] = gsm8k.gold_answer(a)
    return gold


def prepare(raw_dir: Path, out_dir: Path, name: str = "routerbench_0shot") -> Manifest:
    """Normalise the raw pickle into ``out_dir`` (see module docstring for the layout)."""
    import pandas as pd

    pkl = raw_dir / f"{name}.pkl"
    df = load_dataframe(pkl)
    rows_in = len(df)
    dropped: dict[str, int] = {}

    keep = ~df["eval_name"].isin(_DROP_EVALS)
    dropped["smoke_test_eval"] = int((~keep).sum())
    df = df[keep]
    dup = df["prompt"].duplicated(keep="first")
    dropped["duplicate_prompt"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    turns = df["prompt"].map(parse_turns)
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "eval_name": df["eval_name"].astype(str),
            "family": df["eval_name"].map(family_of),
            "router_text": turns.map(router_text),
            "turns_json": turns.map(json.dumps),
            "n_turns": turns.map(len).astype(np.int32),
            "prompt_chars": turns.map(lambda t: sum(len(x) for x in t)).astype(np.int32),
        }
    )
    responses = pd.DataFrame({"sample_id": out["sample_id"]})
    for col, mid in MODEL_IDS.items():
        q = pd.to_numeric(df[col], errors="raise").astype(np.float32)
        out[f"q_orig__{mid}"] = q
        out[f"q__{mid}"] = q
        out[f"c__{mid}"] = df[f"{col}|total_cost"].astype(np.float64)
        responses[f"r__{mid}"] = df[f"{col}|model_response"].map(lambda r: parse_turns(r)[-1])

    # --- GSM8K: replace the graded score with exact-match correctness --------------------
    gold = _gsm8k_gold([raw_dir / "gsm8k" / f for f in GSM8K_FILES])
    is_gsm = out["family"] == "gsm8k"
    questions = turns[is_gsm].map(lambda t: t[-1])
    keys = questions.map(gsm8k.normalize_question)
    found = keys.map(lambda k: k in gold)
    no_gold = keys.index[~found]
    stats: dict[str, Any] = {
        "rows": int(is_gsm.sum()),
        "matched_gold": int(found.sum()),
        "dropped_no_gold": len(no_gold),
    }
    flips: dict[str, dict[str, float]] = {}
    for mid in MODEL_IDS.values():
        resp = responses.loc[keys.index[found], f"r__{mid}"]
        correct = [
            gsm8k.is_correct(r, gold[k], q)
            for r, k, q in zip(resp, keys[found], questions[found], strict=True)
        ]
        new = np.asarray(correct, dtype=np.float32)
        out.loc[keys.index[found], f"q__{mid}"] = new
        old = out.loc[keys.index[found], f"q_orig__{mid}"].to_numpy()
        flips[mid] = {
            "orig_mean": round(float(old.mean()), 4),
            "exact_match_acc": round(float(new.mean()), 4),
            "orig_score_ge_0.75_but_wrong": round(float(((old >= 0.75) & (new == 0)).mean()), 4),
            "orig_score_le_0.5_but_right": round(float(((old <= 0.5) & (new == 1)).mean()), 4),
        }
    stats["per_model"] = flips
    if len(no_gold):
        out = out.drop(index=no_gold).reset_index(drop=True)
        responses = responses.drop(index=no_gold).reset_index(drop=True)
    dropped["gsm8k_no_gold"] = len(no_gold)

    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target / "routing.parquet", index=False)
    responses.to_parquet(target / "responses.parquet", index=False)
    manifest = Manifest(
        source={
            "routerbench_repo": REPO_ID,
            "routerbench_revision": REVISION,
            "routerbench_file": pkl.name,
            "routerbench_sha256": sha256_of(pkl),
            "gsm8k_repo": GSM8K_REPO,
            "gsm8k_revision": GSM8K_REVISION,
        },
        rows_in=rows_in,
        rows_out=len(out),
        dropped=dropped,
        models=list(MODEL_IDS.values()),
        families={k: int(v) for k, v in out["family"].value_counts().items()},
        gsm8k_regrade=stats,
    )
    (target / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest
