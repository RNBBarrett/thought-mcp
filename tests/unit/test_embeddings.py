"""Tests for the embedder ABC and the deterministic test embedder.

The default production embedder is sentence-transformers/all-MiniLM-L6-v2 (lazy-
imported). For tests we use a deterministic hashed-bag-of-words embedder so we
don't depend on a 80MB model download.
"""
from __future__ import annotations

import numpy as np

from thought.embeddings.deterministic import DeterministicEmbedder


def test_deterministic_embedder_returns_unit_vector() -> None:
    emb = DeterministicEmbedder(dim=128)
    v = emb.embed("hello world")
    assert v.shape == (128,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_same_text_same_embedding() -> None:
    emb = DeterministicEmbedder(dim=64)
    a = emb.embed("the quick brown fox")
    b = emb.embed("the quick brown fox")
    np.testing.assert_array_equal(a, b)


def test_similar_text_similar_embedding() -> None:
    emb = DeterministicEmbedder(dim=128)
    a = emb.embed("apples and oranges")
    b = emb.embed("apples and pears")
    c = emb.embed("quantum mechanics tensor networks")
    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac


def test_model_version_and_dim_reported() -> None:
    emb = DeterministicEmbedder(dim=256)
    assert emb.dim == 256
    assert emb.model_name == "deterministic-bow"
    assert emb.model_version
