"""Vector layer tests — GraphRAG-style retrieval with Matryoshka 2-pass.

The vector layer:
1. Embeds a query, runs ANN over stored entity embeddings
2. (Matryoshka) does a coarse ANN at low dim, then reranks top-N at full dim
3. (GraphRAG) optionally expands each seed hit along graph edges, depth K
4. Returns ranked Hits with score, layer="vector", expansion_path
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from thought.embeddings.base import vector_to_bytes
from thought.embeddings.deterministic import DeterministicEmbedder
from thought.layers.vector import VectorLayer
from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteBackend(tmp_path / "v.db")
    b.migrate()
    yield b
    b.close()


@pytest.fixture()
def embedder():
    return DeterministicEmbedder(dim=128)


@pytest.fixture()
def vector_layer(backend, embedder):
    return VectorLayer(backend=backend, embedder=embedder)


def _index(backend, embedder, name: str, type_: str = "CONCEPT") -> str:
    src = backend.upsert_source(f"src::{name}")
    now = datetime.now(UTC)
    eid = backend.upsert_entity(
        type_=type_, name=name, scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    v = embedder.embed(name)
    backend.store_embedding(
        entity_id=eid,
        model_name=embedder.model_name,
        model_version=embedder.model_version,
        dim=embedder.dim,
        vector=vector_to_bytes(v),
    )
    return eid


def test_vector_search_returns_most_similar_first(vector_layer, backend, embedder) -> None:
    _index(backend, embedder, "the quick brown fox jumps")
    target_id = _index(backend, embedder, "apples pears oranges plums")
    _index(backend, embedder, "quantum mechanics tensor networks")
    hits = vector_layer.search(
        query="fruit apples pears", k=3, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=False,
    )
    assert hits[0].entity.id == target_id


def test_vector_search_respects_limit(vector_layer, backend, embedder) -> None:
    for i in range(10):
        _index(backend, embedder, f"item number {i}")
    hits = vector_layer.search(
        query="item", k=3, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=False,
    )
    assert len(hits) <= 3


def test_vector_search_with_graph_expansion(vector_layer, backend, embedder) -> None:
    """Graph expansion boosts the score of a graph-connected neighbor relative
    to its plain ANN score, even when the neighbor's text is semantically far
    from the query.
    """
    seed_id = _index(backend, embedder, "apples and pears fruit basket")
    neighbor_id = _index(backend, embedder, "completely unrelated topic")
    src = backend.upsert_source("edge")
    now = datetime.now(UTC)
    backend.upsert_edge(
        source_id=seed_id, target_id=neighbor_id,
        relation_type="RELATED_TO", source_ref=src, confidence_score=0.9,
        valid_from=now, learned_at=now,
    )

    def neighbor_score(hits):
        for h in hits:
            if h.entity.id == neighbor_id:
                return h.score
        return None

    no_expand = vector_layer.search(
        query="apples", k=10, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=False,
    )
    with_expand = vector_layer.search(
        query="apples", k=10, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=True, expansion_depth=1,
    )
    raw = neighbor_score(no_expand)
    boosted = neighbor_score(with_expand)
    assert boosted is not None
    # If the neighbor wasn't returnable at all without expansion, raw is None;
    # graph expansion at minimum makes it returnable.
    if raw is not None:
        assert boosted > raw, f"graph expansion should boost neighbor score: raw={raw}, boosted={boosted}"


def test_matryoshka_two_pass_returns_same_top_hit(vector_layer, backend, embedder) -> None:
    target_id = _index(backend, embedder, "apples pears bananas")
    _index(backend, embedder, "trains and railways")
    _index(backend, embedder, "ocean currents pacific")
    # The full search is the 2-pass: low-dim first, full-dim rerank.
    hits = vector_layer.search(
        query="apples pears", k=1, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=False,
    )
    assert hits[0].entity.id == target_id


def test_vector_hits_carry_layer_and_confidence_class(
    vector_layer, backend, embedder
) -> None:
    _index(backend, embedder, "apples and pears")
    hits = vector_layer.search(
        query="apples", k=1, scope_filter=ScopeFilter(scope="all"),
        expand_via_graph=False,
    )
    assert hits[0].layer == "vector"
    assert hits[0].confidence_class in {"source_grounded", "inferred", "hallucination_risk"}
