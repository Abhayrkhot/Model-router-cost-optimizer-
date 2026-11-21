"""Learned quality predictors.

All predict a (n, m) matrix of P(model m answers well).  Labels are soft (most are 0/1, a few
benchmarks are graded), so the torch models train with binary cross-entropy on soft targets.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from modelrouter.routers.base import F32, FloatArray, Inputs, QualityModel, register


def _length_feature(prompt_chars: npt.NDArray[np.int64]) -> F32:
    feat = (np.log1p(prompt_chars.astype(np.float32)) - 6.0)[:, None] / 2.0
    return feat.astype(np.float32)


@register
class KNNQuality(QualityModel):
    """Cosine k-nearest-neighbours over prompt embeddings, shrunk toward each model's prior.

    q̂[i, m] = (Σ_{j ∈ N_k(i)} w_ij q[j, m] + α · prior_m) / (Σ w_ij + α)
    k and α are chosen on the validation set.
    """

    kind: ClassVar[str] = "knn"

    def __init__(self, models: tuple[str, ...], k: int = 32, alpha: float = 4.0) -> None:
        self.models = models
        self.k = k
        self.alpha = alpha
        self.bank = np.zeros((0, 0), dtype=np.float32)
        self.labels = np.zeros((0, len(models)))
        self.prior = np.zeros(len(models))

    def _neighbours(self, emb: F32, k: int) -> tuple[npt.NDArray[np.int64], F32]:
        idx_out = np.empty((len(emb), k), dtype=np.int64)
        sim_out = np.empty((len(emb), k), dtype=np.float32)
        for start in range(0, len(emb), 1024):
            sims = emb[start : start + 1024] @ self.bank.T
            idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            idx_out[start : start + 1024] = idx
            sim_out[start : start + 1024] = np.take_along_axis(sims, idx, axis=1)
        return idx_out, sim_out

    def _predict_from(
        self, idx: npt.NDArray[np.int64], sims: F32, k: int, alpha: float
    ) -> FloatArray:
        order = np.argsort(-sims, axis=1)[:, :k]
        idx = np.take_along_axis(idx, order, axis=1)
        w = np.clip(np.take_along_axis(sims, order, axis=1), 0.0, None).astype(np.float64)
        num = np.einsum("nk,nkm->nm", w, self.labels[idx]) + alpha * self.prior
        out: FloatArray = num / (w.sum(axis=1, keepdims=True) + alpha)
        return out

    def fit(self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray) -> KNNQuality:
        self.bank = train.require_embedding().astype(np.float32)
        self.labels = q_train
        self.prior = q_train.mean(axis=0)
        k_max = 128
        idx, sims = self._neighbours(val.require_embedding(), k_max)
        best = (math.inf, self.k, self.alpha)
        for k in (8, 16, 32, 64, 128):
            for alpha in (0.0, 1.0, 4.0, 16.0):
                pred = np.clip(self._predict_from(idx, sims, k, alpha), 1e-4, 1 - 1e-4)
                loss = float(-np.mean(q_val * np.log(pred) + (1 - q_val) * np.log(1 - pred)))
                if loss < best[0]:
                    best = (loss, k, alpha)
        _, self.k, self.alpha = best
        return self

    def predict(self, x: Inputs) -> FloatArray:
        idx, sims = self._neighbours(x.require_embedding(), self.k)
        return self._predict_from(idx, sims, self.k, self.alpha)

    def config(self) -> dict[str, Any]:
        return {"k": self.k, "alpha": self.alpha}

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {"bank": self.bank, "labels": self.labels, "prior": self.prior}

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> KNNQuality:
        m = cls(models, k=int(config["k"]), alpha=float(config["alpha"]))
        m.bank, m.labels, m.prior = arrays["bank"], arrays["labels"], arrays["prior"]
        return m


@register
class MLPQuality(QualityModel):
    """Multi-output MLP on [embedding, log prompt length] with early stopping on val loss.

    hidden=0 gives per-model logistic regression on the same features.  Serving runs the
    forward pass in NumPy, so the gateway doesn't need torch.
    """

    kind: ClassVar[str] = "mlp"

    def __init__(
        self,
        models: tuple[str, ...],
        hidden: int = 512,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        patience: int = 6,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.models = models
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.params: dict[str, npt.NDArray[np.float32]] = {}
        self.best_epoch = 0
        self.best_val_loss = math.inf

    @staticmethod
    def _features(x: Inputs) -> F32:
        return np.hstack([x.require_embedding(), _length_feature(x.prompt_chars)]).astype(
            np.float32
        )

    def fit(self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray) -> MLPQuality:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        xt = torch.from_numpy(self._features(train))
        yt = torch.from_numpy(q_train.astype(np.float32))
        xv = torch.from_numpy(self._features(val))
        yv = torch.from_numpy(q_val.astype(np.float32))
        d_in, d_out = xt.shape[1], yt.shape[1]
        layers: list[nn.Module]
        if self.hidden > 0:
            layers = [
                nn.Linear(d_in, self.hidden),
                nn.GELU(approximate="tanh"),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, d_out),
            ]
        else:
            layers = [nn.Linear(d_in, d_out)]
        net = nn.Sequential(*layers)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)
        best_state: dict[str, torch.Tensor] = {}
        stale = 0
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for start in range(0, len(xt), self.batch_size):
                b = perm[start : start + self.batch_size]
                opt.zero_grad()
                loss_fn(net(xt[b]), yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(xv), yv))
            if val_loss < self.best_val_loss - 1e-5:
                self.best_val_loss, self.best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.params = {k.replace(".", "_"): v.numpy() for k, v in net.state_dict().items()}
        return self

    def predict(self, x: Inputs) -> FloatArray:
        feats = self._features(x)
        p = self.params
        if self.hidden > 0:
            hidden = _gelu_tanh(feats @ p["0_weight"].T + p["0_bias"])
            logits = hidden @ p["3_weight"].T + p["3_bias"]
        else:
            logits = feats @ p["0_weight"].T + p["0_bias"]
        out: FloatArray = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return out

    def config(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return dict(self.params)

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> MLPQuality:
        m = cls(models, hidden=int(config["hidden"]), dropout=float(config["dropout"]))
        m.params = {k: v.astype(np.float32) for k, v in arrays.items()}
        m.best_epoch = int(config.get("best_epoch", 0))
        return m


def _gelu_tanh(x: F32) -> F32:
    """GELU, tanh form: matches torch's nn.GELU(approximate="tanh") used in training."""
    c = np.float32(math.sqrt(2.0 / math.pi))
    out: F32 = (0.5 * x * (1.0 + np.tanh(c * (x + np.float32(0.044715) * x**3)))).astype(np.float32)
    return out


@register
class TfidfQuality(QualityModel):
    """TF-IDF word/bigram features + multi-output ridge regression, clipped to [0, 1].

    A cheap lexical router: no embedding model, sub-millisecond inference.  Training uses
    scikit-learn; serving uses `_TfidfTransform`, a NumPy re-implementation of the fitted
    vectorizer (same tokenizer, sublinear tf, idf, l2 norm) so the gateway needs neither
    scikit-learn nor an ONNX model.  `tests/test_routers.py` checks the two agree.
    """

    kind: ClassVar[str] = "tfidf"

    def __init__(self, models: tuple[str, ...], max_features: int = 50_000) -> None:
        self.models = models
        self.max_features = max_features
        self.alpha = 1.0
        self.transform: _TfidfTransform | None = None
        self.coef = np.zeros((len(models), 0), dtype=np.float32)
        self.intercept = np.zeros(len(models))

    def fit(
        self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray
    ) -> TfidfQuality:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=self.max_features, sublinear_tf=True, min_df=2
        )
        xt = vec.fit_transform(train.text)
        xv = vec.transform(val.text)
        best = (math.inf, 1.0)
        for alpha in (0.3, 1.0, 3.0, 10.0):
            pred = Ridge(alpha=alpha).fit(xt, q_train).predict(xv)
            mse = float(np.mean((np.clip(pred, 0, 1) - q_val) ** 2))
            if mse < best[0]:
                best = (mse, alpha)
        self.alpha = best[1]
        ridge = Ridge(alpha=self.alpha).fit(xt, q_train)
        terms = sorted(vec.vocabulary_, key=vec.vocabulary_.__getitem__)
        self.transform = _TfidfTransform(terms, vec.idf_.astype(np.float64))
        self.coef = ridge.coef_.astype(np.float32)
        self.intercept = np.asarray(ridge.intercept_, dtype=np.float64)
        return self

    def predict(self, x: Inputs) -> FloatArray:
        assert self.transform is not None, "fit or load the router first"
        out = np.empty((len(x), len(self.models)))
        for i, text in enumerate(x.text):
            idx, val = self.transform.row(text)
            out[i] = self.coef[:, idx] @ val + self.intercept
        clipped: FloatArray = np.clip(out, 0.0, 1.0)
        return clipped

    def config(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "max_features": self.max_features}

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        assert self.transform is not None
        return {
            "terms": np.array(self.transform.terms, dtype=np.str_),
            "idf": self.transform.idf,
            "coef": self.coef,
            "intercept": self.intercept,
        }

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> TfidfQuality:
        m = cls(models, max_features=int(config["max_features"]))
        m.alpha = float(config["alpha"])
        m.transform = _TfidfTransform([str(t) for t in arrays["terms"]], arrays["idf"])
        m.coef = arrays["coef"].astype(np.float32)
        m.intercept = arrays["intercept"].astype(np.float64)
        return m


# scikit-learn's default analyzer: lowercase, tokens of 2+ word characters, then n-grams.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


class _TfidfTransform:
    """NumPy equivalent of a fitted TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)."""

    def __init__(self, terms: list[str], idf: FloatArray) -> None:
        self.terms = terms
        self.index = {t: i for i, t in enumerate(terms)}
        self.idf = idf

    def row(self, text: str) -> tuple[npt.NDArray[np.int64], FloatArray]:
        tokens = _TOKEN.findall(text.lower())
        grams = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
        counts: dict[int, int] = {}
        for g in grams:
            j = self.index.get(g)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        val = (1.0 + np.log(tf)) * self.idf[idx]
        val /= np.linalg.norm(val)
        return idx, val


@register
class EnsembleQuality(QualityModel):
    """Mean of member predictions (members are fitted independently)."""

    kind: ClassVar[str] = "ensemble"

    def __init__(self, models: tuple[str, ...], members: list[QualityModel]) -> None:
        self.models = models
        self.members = members

    def fit(
        self, train: Inputs, q_train: FloatArray, val: Inputs, q_val: FloatArray
    ) -> EnsembleQuality:
        for m in self.members:
            m.fit(train, q_train, val, q_val)
        return self

    def predict(self, x: Inputs) -> FloatArray:
        preds: FloatArray = np.mean([m.predict(x) for m in self.members], axis=0)
        return preds

    def config(self) -> dict[str, Any]:
        return {"members": [{"kind": m.kind, "config": m.config()} for m in self.members]}

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {f"{i}__{k}": v for i, m in enumerate(self.members) for k, v in m.arrays().items()}

    @classmethod
    def from_saved(
        cls, models: tuple[str, ...], config: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]
    ) -> EnsembleQuality:
        from modelrouter.routers.base import _REGISTRY

        members: list[QualityModel] = []
        for i, spec in enumerate(config["members"]):
            prefix = f"{i}__"
            sub = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            members.append(_REGISTRY[spec["kind"]].from_saved(models, spec["config"], sub))
        return cls(models, members)
