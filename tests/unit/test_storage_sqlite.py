"""Tests for the SQLite storage backend.

The backend is the lowest layer — everything else builds on it. We test:
- migration creates the expected schema
- source upsert dedupes by content hash
- entity upsert respects the bi-temporal model
- edge upsert enforces source_ref
- scope filtering returns the right rows
- access tracking increments correctly
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def backend(tmp_path):
    db = SQLiteBackend(tmp_path / "test.db")
    db.migrate()
    yield db
    db.close()


def test_migrate_creates_all_tables(backend: SQLiteBackend) -> None:
    tables = backend.list_tables()
    expected = {
        "sources",
        "entities",
        "edges",
        "triples",
        "embeddings",
        "strength_cache",
        "consolidation_log",
        "schema_version",
    }
    assert expected.issubset(set(tables))


def test_schema_version_is_recorded(backend: SQLiteBackend) -> None:
    # v0.2 migration bumps the version to 2.
    assert backend.schema_version() == 2


def test_upsert_source_dedupes_by_hash(backend: SQLiteBackend) -> None:
    sid_a = backend.upsert_source("alpha content")
    sid_b = backend.upsert_source("alpha content")
    sid_c = backend.upsert_source("different content")
    assert sid_a == sid_b
    assert sid_a != sid_c


def test_upsert_entity_creates_row_with_bitemporal(backend: SQLiteBackend) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    eid = backend.upsert_entity(
        type_="PERSON",
        name="Kendra",
        scope="private",
        owner_id="alice",
        valid_from=now,
        learned_at=now,
        source_ref=src,
    )
    row = backend.get_entity(eid)
    assert row is not None
    assert row.type == "PERSON"
    assert row.canonical_name == "kendra"
    assert row.scope == "private"
    assert row.owner_id == "alice"
    assert row.valid_until is None
    assert row.unlearned_at is None
    assert row.tier == "hot"


def test_upsert_edge_requires_source_ref(backend: SQLiteBackend) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    a = backend.upsert_entity(
        type_="PERSON", name="A", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    b = backend.upsert_entity(
        type_="ORG", name="B", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    eid = backend.upsert_edge(
        source_id=a,
        target_id=b,
        relation_type="OWNS",
        source_ref=src,
        confidence_score=0.9,
        valid_from=now,
        learned_at=now,
    )
    edges = backend.edges_from(a)
    assert len(edges) == 1
    assert edges[0].id == eid
    assert edges[0].relation_type == "OWNS"
    assert edges[0].source_ref == src


def test_scope_filter_isolates_private(backend: SQLiteBackend) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    backend.upsert_entity(
        type_="CONCEPT", name="shared_fact", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    backend.upsert_entity(
        type_="CONCEPT", name="alice_secret", scope="private", owner_id="alice",
        valid_from=now, learned_at=now, source_ref=src,
    )
    backend.upsert_entity(
        type_="CONCEPT", name="bob_secret", scope="private", owner_id="bob",
        valid_from=now, learned_at=now, source_ref=src,
    )

    alice_view = backend.list_entities(ScopeFilter(scope="all", owner_id="alice"))
    names = {e.name for e in alice_view}
    assert names == {"shared_fact", "alice_secret"}

    bob_view = backend.list_entities(ScopeFilter(scope="all", owner_id="bob"))
    names = {e.name for e in bob_view}
    assert names == {"shared_fact", "bob_secret"}

    shared_only = backend.list_entities(ScopeFilter(scope="shared"))
    assert {e.name for e in shared_only} == {"shared_fact"}


def test_access_tracking_increments(backend: SQLiteBackend) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    eid = backend.upsert_entity(
        type_="PERSON", name="Z", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    first = backend.get_entity(eid)
    assert first is not None
    assert first.access_count == 0

    backend.touch_access(eid)
    backend.touch_access(eid)
    after = backend.get_entity(eid)
    assert after is not None
    assert after.access_count == 2
    assert after.last_accessed_at >= first.last_accessed_at


def test_supersede_creates_supersedes_edge(backend: SQLiteBackend) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    old = backend.upsert_entity(
        type_="PREF", name="adidas", scope="private", owner_id="kendra",
        valid_from=now - timedelta(days=400), learned_at=now - timedelta(days=400),
        source_ref=src,
    )
    new = backend.upsert_entity(
        type_="PREF", name="nike", scope="private", owner_id="kendra",
        valid_from=now, learned_at=now, source_ref=src,
    )
    edge_id = backend.supersede(old_id=old, new_id=new, source_ref=src, at=now)

    edges = backend.edges_from(new)
    assert any(e.id == edge_id and e.relation_type == "SUPERSEDES" for e in edges)
    old_row = backend.get_entity(old)
    assert old_row is not None
    assert old_row.valid_until == now  # retired, but row still present (append-only)
