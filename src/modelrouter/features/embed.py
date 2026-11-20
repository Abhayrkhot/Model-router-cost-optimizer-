"""Sentence embeddings for routing (ONNX via fastembed, CPU, no torch at serve time)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Enough to tell what kind of question a prompt is; halves the cost of long prompts vs 512.
DEFAULT_MAX_TOKENS = 256

F32 = npt.NDArray[np.float32]


class Embedder:
    """L2-normalised embeddings; batches are length-sorted so short prompts aren't padded."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        threads: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.max_tokens = max_tokens
        self._model = TextEmbedding(model_name, threads=threads, cache_dir=cache_dir)
        tokenizer = self._tokenizer()
        if tokenizer is not None:
            tokenizer.enable_truncation(max_length=max_tokens)
        self.dim = int(self.embed(["dimension probe"]).shape[1])

    def _tokenizer(self) -> Any:
        inner = getattr(self._model, "model", None)
        return getattr(inner, "tokenizer", None)

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> F32:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        sorted_texts = [texts[i] for i in order]
        vecs = np.stack(list(self._model.embed(sorted_texts, batch_size=batch_size)))
        out = np.empty_like(vecs, dtype=np.float32)
        out[order] = vecs
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        normed: F32 = (out / np.maximum(norms, 1e-12)).astype(np.float32)
        return normed

    def embed_one(self, text: str) -> F32:
        one: F32 = self.embed([text])[0]
        return one


def cache_key(texts: Sequence[str], model_name: str, max_tokens: int) -> str:
    digest = hashlib.sha256(f"{model_name}|{max_tokens}|{len(texts)}".encode())
    for t in texts:
        digest.update(t.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def cached_embeddings(
    texts: Sequence[str],
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    threads: int | None = None,
) -> F32:
    """Embed `texts`, reusing artifacts/embeddings/<model>-<key>.npy when the inputs match."""
    key = cache_key(texts, model_name, max_tokens)
    path = cache_dir / f"{model_name.replace('/', '__')}-{max_tokens}-{key}.npy"
    if path.exists():
        cached: F32 = np.load(path)
        return cached
    vecs = Embedder(model_name, max_tokens=max_tokens, threads=threads).embed(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    tmp.rename(path)
    return vecs
