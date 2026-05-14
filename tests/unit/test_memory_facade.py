"""Memory facade tests.

The Memory class is the public Python entry point. It composes the storage
backend, embedder, ingest pipeline, dispatcher, and consolidation engine into
the two operations that matter: `remember(...)` and `recall(...)`.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from thought.memory import Memory


@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "m.db"),
        embedder_choice="deterministic",
        embedder_dim=128,
    )
    yield m
    m.close()


def test_remember_returns_source_and_entities(mem) -> None:
    r = mem.remember(content="Alice owns Acme Corp.", scope="shared")
    assert r.source_id
    assert r.entity_ids


def test_remember_dedups_identical_content(mem) -> None:
    r1 = mem.remember(content="Hello world.", scope="shared")
    r2 = mem.remember(content="Hello world.", scope="shared")
    assert r2.duplicate_of_source == r1.source_id


def test_recall_returns_results_for_known_content(mem) -> None:
    mem.remember(content="Alice owns Acme Corp.", scope="shared")
    res = mem.recall(query="alice", limit=5)
    assert res.hits
    assert res.query_class.value in {"VIBE", "FACT", "HYBRID", "CHANGE"}


def test_recall_returns_empty_for_empty_kb(mem) -> None:
    res = mem.recall(query="anything", limit=5)
    assert res.hits == []
    assert res.low_confidence is True


def test_recall_as_of_filter(mem) -> None:
    base = datetime.now(UTC)
    mem.remember(
        content="Kendra prefers Adidas.",
        scope="private", owner_id="kendra", now=base - timedelta(days=400),
    )
    mem.remember(
        content="Kendra prefers Nike.",
        scope="private", owner_id="kendra", now=base,
        unique_predicates={"PREFERS"},
    )
    past = mem.recall(
        query="prefer", limit=5, scope="all", owner_id="kendra",
        as_of=base - timedelta(days=200),
    )
    names = {h.entity.name.lower() for h in past.hits}
    # 200 days ago: Adidas was current, Nike not yet learned.
    assert any("adidas" in n for n in names)


def test_remember_with_unique_predicate_detects_contradiction(mem) -> None:
    base = datetime.now(UTC)
    mem.remember(
        content="Kendra prefers Adidas.", scope="private", owner_id="kendra",
        now=base - timedelta(days=10),
    )
    r = mem.remember(
        content="Kendra prefers Nike.", scope="private", owner_id="kendra",
        now=base, unique_predicates={"PREFERS"},
    )
    assert r.contradictions_detected, "expected at least one contradiction"
