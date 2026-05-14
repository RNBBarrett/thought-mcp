"""Embedder protocol.

All embedders return L2-normalized ``np.float32`` vectors of fixed ``dim``.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> np.ndarray: ...


def vector_to_bytes(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def bytes_to_vector(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim).copy()
