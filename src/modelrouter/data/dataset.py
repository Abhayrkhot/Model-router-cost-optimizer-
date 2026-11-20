"""In-memory routing dataset: prompts plus the outcome of every model on every prompt."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]


@dataclass(frozen=True)
class RoutingDataset:
    """`quality[i, m]` and `cost[i, m]` are model m's realised outcome on prompt i."""

    sample_id: StrArray
    family: StrArray
    eval_name: StrArray
    text: Sequence[str]
    prompt_chars: IntArray
    models: tuple[str, ...]
    quality: FloatArray
    cost: FloatArray

    def __post_init__(self) -> None:
        n, m = self.quality.shape
        if self.cost.shape != (n, m) or len(self.models) != m:
            raise ValueError("quality/cost/models shapes disagree")
        for name in ("sample_id", "family", "eval_name", "text", "prompt_chars"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.quality.shape[0])

    def subset(self, idx: IntArray | Sequence[int]) -> RoutingDataset:
        idx = np.asarray(idx, dtype=np.int64)
        return RoutingDataset(
            sample_id=self.sample_id[idx],
            family=self.family[idx],
            eval_name=self.eval_name[idx],
            text=[self.text[i] for i in idx],
            prompt_chars=self.prompt_chars[idx],
            models=self.models,
            quality=self.quality[idx],
            cost=self.cost[idx],
        )

    def select_models(self, models: Sequence[str]) -> RoutingDataset:
        cols = [self.models.index(m) for m in models]
        return RoutingDataset(
            sample_id=self.sample_id,
            family=self.family,
            eval_name=self.eval_name,
            text=self.text,
            prompt_chars=self.prompt_chars,
            models=tuple(models),
            quality=self.quality[:, cols],
            cost=self.cost[:, cols],
        )

    def model_index(self, model: str) -> int:
        return self.models.index(model)

    @classmethod
    def load(cls, directory: str | Path) -> RoutingDataset:
        import pandas as pd

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        models: list[str] = manifest["models"]
        df = pd.read_parquet(directory / "routing.parquet")
        return cls(
            sample_id=df["sample_id"].to_numpy(dtype=str),
            family=df["family"].to_numpy(dtype=str),
            eval_name=df["eval_name"].to_numpy(dtype=str),
            text=df["router_text"].tolist(),
            prompt_chars=df["prompt_chars"].to_numpy(dtype=np.int64),
            models=tuple(models),
            quality=df[[f"q__{m}" for m in models]].to_numpy(dtype=np.float64),
            cost=df[[f"c__{m}" for m in models]].to_numpy(dtype=np.float64),
        )
