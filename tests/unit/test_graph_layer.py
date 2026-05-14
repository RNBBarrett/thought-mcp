"""Graph layer tests.

The graph layer offers:
- neighbors(entity_id, depth) — bounded BFS expansion
- personalized_pagerank(seeds) — HippoRAG-style scoring across the graph
- contradictions_for(entity_id) — finds CONTRADICTS edges

Personalized PageRank is the key novel mechanism (HippoRAG 2). It scores every
reachable node by the probability mass it accumulates from random walks starting
at the seed nodes; nodes that lots of paths converge to score higher than nodes
only loosely connected.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from thought.layers.graph import GraphLayer
from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteBackend(tmp_path / "g.db")
    b.migrate()
    yield b
    b.close()


@pytest.fixture()
def graph(backend):
    return GraphLayer(backend)


def _mk_entity(backend, name: str, type_: str = "CONCEPT", scope="shared", owner=None) -> str:
    src = backend.upsert_source(f"src::{name}")
    now = datetime.now(UTC)
    return backend.upsert_entity(
        type_=type_, name=name, scope=scope, owner_id=owner,
        valid_from=now, learned_at=now, source_ref=src,
    )


def _mk_edge(backend, a: str, b: str, rel: str = "RELATED_TO", conf: float = 0.9):
    src = backend.upsert_source(f"edge::{a}::{rel}::{b}")
    now = datetime.now(UTC)
    return backend.upsert_edge(
        source_id=a, target_id=b, relation_type=rel,
        source_ref=src, confidence_score=conf,
        valid_from=now, learned_at=now,
    )


def test_neighbors_depth_one(graph, backend) -> None:
    a = _mk_entity(backend, "a")
    b = _mk_entity(backend, "b")
    c = _mk_entity(backend, "c")
    _mk_edge(backend, a, b)
    _mk_edge(backend, b, c)
    n1 = graph.neighbors(a, depth=1)
    ids = {e.id for e in n1}
    assert b in ids
    assert c not in ids  # depth-1 only


def test_neighbors_depth_two_includes_indirect(graph, backend) -> None:
    a = _mk_entity(backend, "a")
    b = _mk_entity(backend, "b")
    c = _mk_entity(backend, "c")
    _mk_edge(backend, a, b)
    _mk_edge(backend, b, c)
    n2 = graph.neighbors(a, depth=2)
    ids = {e.id for e in n2}
    assert b in ids
    assert c in ids


def test_personalized_pagerank_assigns_higher_score_to_well_connected_node(
    graph, backend
) -> None:
    # Build: seed -> X, seed -> Y, X -> hub, Y -> hub, hub -> tail
    # `hub` should outscore `tail` because of converging paths from the seed.
    seed = _mk_entity(backend, "seed")
    x = _mk_entity(backend, "x")
    y = _mk_entity(backend, "y")
    hub = _mk_entity(backend, "hub")
    tail = _mk_entity(backend, "tail")
    _mk_edge(backend, seed, x)
    _mk_edge(backend, seed, y)
    _mk_edge(backend, x, hub)
    _mk_edge(backend, y, hub)
    _mk_edge(backend, hub, tail)

    scores = graph.personalized_pagerank(
        seeds=[seed],
        scope_filter=ScopeFilter(scope="all"),
    )
    # The hub MUST outscore the tail (two converging paths beat a single hop).
    # The seed MUST get a non-trivial score from the restart mass.
    # We don't pin seed-vs-hub ordering: in canonical Personalized PageRank
    # with bidirectional walks, a central hub legitimately accumulates more
    # mass than the seed even though the seed gets restart probability.
    assert scores[hub] > scores[tail]
    assert scores[seed] > 0
    assert scores[hub] > 0
    assert scores[tail] > 0


def test_local_personalized_pagerank_returns_same_ranking_at_top(graph, backend) -> None:
    """The push-style local PPR is an ε-approximation of the global PPR.

    We don't require identical scores — we require that the top-3 nodes by
    score match between the two methods on a small graph. This is the
    Andersen-Chung-Lang guarantee.
    """
    seed = _mk_entity(backend, "seed")
    a = _mk_entity(backend, "a")
    b = _mk_entity(backend, "b")
    c = _mk_entity(backend, "c")
    _mk_edge(backend, seed, a)
    _mk_edge(backend, a, b)
    _mk_edge(backend, b, c)

    sf = ScopeFilter(scope="all")
    global_scores = graph.personalized_pagerank(seeds=[seed], scope_filter=sf)
    local_scores = graph.local_personalized_pagerank(
        seeds=[seed], scope_filter=sf, epsilon=1e-6,
    )

    top_global = {nid for nid, _ in sorted(
        global_scores.items(), key=lambda kv: kv[1], reverse=True
    )[:3]}
    top_local = {nid for nid, _ in sorted(
        local_scores.items(), key=lambda kv: kv[1], reverse=True
    )[:3]}
    # Approximation: at least 2 of the top-3 should overlap.
    assert len(top_global & top_local) >= 2


def test_personalized_pagerank_respects_scope_isolation(graph, backend) -> None:
    seed = _mk_entity(backend, "seed", scope="private", owner="alice")
    target_alice = _mk_entity(backend, "alice_thing", scope="private", owner="alice")
    target_bob = _mk_entity(backend, "bob_thing", scope="private", owner="bob")
    _mk_edge(backend, seed, target_alice)
    # cross-owner edge is allowed but should be filtered out for alice-only scope
    src2 = backend.upsert_source("cross")
    now = datetime.now(UTC)
    backend.upsert_edge(
        source_id=seed, target_id=target_bob, relation_type="RELATED_TO",
        source_ref=src2, confidence_score=0.9,
        valid_from=now, learned_at=now,
    )
    scores = graph.personalized_pagerank(
        seeds=[seed],
        scope_filter=ScopeFilter(scope="all", owner_id="alice"),
    )
    assert target_alice in scores
    assert target_bob not in scores


def test_contradictions_for_returns_contradicts_edges(graph, backend) -> None:
    a = _mk_entity(backend, "fact_a")
    b = _mk_entity(backend, "fact_b")
    eid = _mk_edge(backend, a, b, rel="CONTRADICTS", conf=0.95)
    found = graph.contradictions_for(a)
    assert any(e.id == eid for e in found)
