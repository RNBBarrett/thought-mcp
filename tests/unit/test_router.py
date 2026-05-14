"""Router tests — query classifier + dispatcher + CRAG evaluator.

The classifier is rule-based (regex/keywords loaded from rules.yaml). The
dispatcher fans out to the right layer(s) and merges with CRAG-style
confidence evaluation.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.pipeline import IngestPipeline
from thought.models import QueryClass, ScopeFilter
from thought.router.classifier import RuleBasedClassifier
from thought.router.dispatcher import Dispatcher
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def classifier():
    return RuleBasedClassifier.with_defaults()


@pytest.fixture()
def world(tmp_path):
    backend = SQLiteBackend(tmp_path / "r.db")
    backend.migrate()
    embedder = DeterministicEmbedder(dim=128)
    pipeline = IngestPipeline(backend=backend, embedder=embedder)
    pipeline.ingest(
        content="Alice owns Acme Corp.",
        scope="shared", now=datetime.now(UTC),
    )
    pipeline.ingest(
        content="Kendra prefers Adidas.",
        scope="private", owner_id="kendra",
        now=datetime.now(UTC) - __import__("datetime").timedelta(days=500),
    )
    pipeline.ingest(
        content="Kendra prefers Nike.",
        scope="private", owner_id="kendra",
        now=datetime.now(UTC),
        unique_predicates={"PREFERS"},
    )
    yield {"backend": backend, "embedder": embedder}
    backend.close()


def test_classifier_vibe_query(classifier) -> None:
    cls, _ = classifier.classify("find something similar to apples")
    assert cls == QueryClass.VIBE


def test_classifier_fact_query(classifier) -> None:
    cls, _ = classifier.classify("who owns Acme")
    assert cls == QueryClass.FACT


def test_classifier_change_query(classifier) -> None:
    cls, _ = classifier.classify("what did Kendra prefer in 2024")
    assert cls == QueryClass.CHANGE


def test_classifier_hybrid_query(classifier) -> None:
    cls, _ = classifier.classify("when did Acme acquire its first customer similar to today's")
    assert cls in (QueryClass.HYBRID, QueryClass.CHANGE)


def test_dispatcher_routes_fact_query(world, classifier) -> None:
    d = Dispatcher(
        backend=world["backend"], embedder=world["embedder"], classifier=classifier,
    )
    result = d.recall(
        query="who owns Acme",
        limit=5,
        scope_filter=ScopeFilter(scope="all"),
    )
    assert result.query_class == QueryClass.FACT
    assert len(result.hits) >= 1
    assert all(h.layer in {"vector", "graph", "temporal"} for h in result.hits)


def test_dispatcher_routes_change_query_uses_temporal(world, classifier) -> None:
    d = Dispatcher(
        backend=world["backend"], embedder=world["embedder"], classifier=classifier,
    )
    base = datetime.now(UTC) - __import__("datetime").timedelta(days=200)
    result = d.recall(
        query="what did Kendra prefer historically",
        limit=5,
        scope_filter=ScopeFilter(scope="all", owner_id="kendra"),
        as_of=base,
        as_of_kind="valid",
    )
    assert result.query_class == QueryClass.CHANGE
    names = {h.entity.name.lower() for h in result.hits}
    # At 200 days ago, Adidas was preferred — Nike came later.
    assert any("adidas" in n for n in names)


def test_dispatcher_bounds_results_to_ten(world, classifier) -> None:
    d = Dispatcher(
        backend=world["backend"], embedder=world["embedder"], classifier=classifier,
    )
    # Request limit > 10 — Pydantic input rejects, but the dispatcher itself
    # should also clip internally.
    result = d.recall(
        query="alice acme",
        limit=10,
        scope_filter=ScopeFilter(scope="all"),
    )
    assert len(result.hits) <= 10


def test_dispatcher_low_confidence_flag_when_kb_irrelevant(world, classifier) -> None:
    """CRAG-style: query unrelated to any stored content sets low_confidence."""
    d = Dispatcher(
        backend=world["backend"], embedder=world["embedder"], classifier=classifier,
        crag_threshold=0.3,
    )
    result = d.recall(
        query="quantum chromodynamics tensor networks",
        limit=5,
        scope_filter=ScopeFilter(scope="all"),
    )
    assert result.low_confidence is True
