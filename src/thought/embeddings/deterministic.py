"""Deterministic hashed-bag-of-words embedder.

Used for tests and as a zero-dependency fallback. Maps each whitespace token to
a stable hash bucket modulo ``dim``; aggregates token contributions; L2-
normalizes. Two strings with overlapping vocabulary therefore have positive
cosine; two strings with disjoint vocabulary score near zero.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _bucket(token: str, dim: int) -> int:
    h = hashlib.blake2s(token.lower().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def _sign(token: str) -> float:
    h = hashlib.blake2s(token.lower().encode("utf-8"), digest_size=4).digest()
    return 1.0 if (h[0] & 1) == 0 else -1.0


class DeterministicEmbedder:
    """A test/fallback embedder. Not for production retrieval quality."""

    def __init__(self, *, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "deterministic-bow"

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text):
            idx = _bucket(token, self._dim)
            v[idx] += _sign(token)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        else:
            # all-zero vector → place a 1.0 in a deterministic empty-string slot
            v[0] = 1.0
        return v

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])
