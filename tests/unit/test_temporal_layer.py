"""Temporal Layer tests — bi-temporal model.

Two distinct time axes per Graphiti (Zep) design:
- valid_time: when the fact was true in the world (world-time)
- learned_at: when the system learned the fact (transaction-time)

"What was true on date X" and "what did we believe on date X" are different
questions. Both must work.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from thought.layers.temporal import TemporalLayer
from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteBackend(tmp_path / "t.db")
    b.migrate()
    yield b
    b.close()


@pytest.fixture()
def temporal(backend):
    return TemporalLayer(backend)


def test_as_of_world_time_filters_by_valid_window(backend, temporal) -> None:
    src = backend.upsert_source("seed")
    base = datetime(2025, 1, 1, tzinfo=UTC)

    # Fact valid Jan-Mar 2025; retired before June.
    old = backend.upsert_entity(
        type_="PREF", name="adidas", scope="private", owner_id="kendra",
        valid_from=base, learned_at=base, source_ref=src,
    )
    backend.supersede(old_id=old, new_id=old, source_ref=src, at=base + timedelta(days=90))

    # Fact valid Apr onward.
    new = backend.upsert_entity(
        type_="PREF", name="nike", scope="private", owner_id="kendra",
        valid_from=base + timedelta(days=90),
        learned_at=base + timedelta(days=90),
        source_ref=src,
    )

    feb = base + timedelta(days=40)
    may = base + timedelta(days=130)

    feb_entities = temporal.entities_valid_at(
        feb, scope_filter=ScopeFilter(scope="all", owner_id="kendra"), kind="valid",
    )
    feb_names = {e.name for e in feb_entities}
    assert "adidas" in feb_names
    assert "nike" not in feb_names

    may_entities = temporal.entities_valid_at(
        may, scope_filter=ScopeFilter(scope="all", owner_id="kendra"), kind="valid",
    )
    may_names = {e.name for e in may_entities}
    assert "nike" in may_names
    assert "adidas" not in may_names


def test_as_of_transaction_time_filters_by_learned_window(backend, temporal) -> None:
    """Even if the fact's valid window includes the query date, transaction-time
    queries should hide facts that were learned *after* the query date.
    """
    src = backend.upsert_source("seed")
    base = datetime(2025, 1, 1, tzinfo=UTC)

    backend.upsert_entity(
        type_="PREF", name="known_in_jan", scope="shared",
        valid_from=base, learned_at=base, source_ref=src,
    )
    # Same valid_from, but learned later
    backend.upsert_entity(
        type_="PREF", name="known_in_june", scope="shared",
        valid_from=base,
        learned_at=base + timedelta(days=150),
        source_ref=src,
    )

    march = base + timedelta(days=60)
    march_entities = temporal.entities_valid_at(
        march, scope_filter=ScopeFilter(scope="shared"), kind="learned",
    )
    march_names = {e.name for e in march_entities}
    assert "known_in_jan" in march_names
    assert "known_in_june" not in march_names


def test_tier_promotion(backend, temporal) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    eid = backend.upsert_entity(
        type_="X", name="foo", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    e = backend.get_entity(eid)
    assert e is not None
    assert e.tier == "hot"

    temporal.set_tier(eid, "warm")
    e = backend.get_entity(eid)
    assert e is not None
    assert e.tier == "warm"


def test_stale_warm_candidates(backend, temporal) -> None:
    src = backend.upsert_source("seed")
    now = datetime.now(UTC)
    old = backend.upsert_entity(
        type_="X", name="old", scope="shared",
        valid_from=now - timedelta(days=60), learned_at=now - timedelta(days=60),
        source_ref=src,
    )
    fresh = backend.upsert_entity(
        type_="X", name="fresh", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    temporal.set_tier(old, "warm")
    temporal.set_tier(fresh, "warm")

    # Simulate that `old` was last accessed 45 days ago.
    backend._conn.execute(  # type: ignore[attr-defined]
        "UPDATE entities SET last_accessed_at = ? WHERE id = ?",
        ((now - timedelta(days=45)).isoformat(), old),
    )
    stale_ids = {e.id for e in temporal.stale_warm_candidates(older_than=timedelta(days=30))}
    assert old in stale_ids
    assert fresh not in stale_ids
