"""Ingest pipeline tests.

The pipeline:
1. Dedup by content_hash (no-op if seen).
2. (Optional) Contextual Retrieval — prepend an LLM-generated context summary
   before embedding. In tests the LLM is None, so the summary is the entity
   name prefix.
3. Entity extraction (regex/spaCy baseline) — produce SPO triples.
4. Jaccard dedup at the triple level.
5. Write source + entities + edges + embedding atomically.
6. Contradiction check — if a unique-predicate is now contested, create a
   CONTRADICTS edge.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.pipeline import IngestPipeline
from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def pipeline(tmp_path):
    backend = SQLiteBackend(tmp_path / "i.db")
    backend.migrate()
    pipe = IngestPipeline(
        backend=backend,
        embedder=DeterministicEmbedder(dim=128),
    )
    yield pipe
    backend.close()


def test_ingest_creates_source_entity_and_embedding(pipeline) -> None:
    result = pipeline.ingest(
        content="Alice owns Acme Corp",
        scope="private",
        owner_id="user1",
        now=datetime.now(UTC),
    )
    assert result.source_id
    assert result.duplicate_of_source is None
    assert len(result.entity_ids) >= 1
    assert result.embeddings_created >= 1


def test_ingest_dedupes_by_content_hash(pipeline) -> None:
    now = datetime.now(UTC)
    r1 = pipeline.ingest(content="Same content.", scope="shared", now=now)
    r2 = pipeline.ingest(content="Same content.", scope="shared", now=now)
    assert r2.duplicate_of_source == r1.source_id


def test_ingest_extracts_entities_from_simple_text(pipeline) -> None:
    pipeline.ingest(
        content="Kendra prefers Nike.",
        scope="private", owner_id="kendra",
        now=datetime.now(UTC),
    )
    entities = pipeline._backend.list_entities(
        ScopeFilter(scope="all", owner_id="kendra")
    )
    names = {e.canonical_name for e in entities}
    # The simple extractor pulls proper-noun-ish tokens.
    assert any("kendra" in n for n in names)
    assert any("nike" in n for n in names)


def test_contradiction_detection_on_unique_predicate(pipeline) -> None:
    """If we record two distinct values for a 'unique-per-subject' predicate,
    a CONTRADICTS edge should appear."""
    now = datetime.now(UTC)
    pipeline.ingest(
        content="Kendra prefers Adidas.",
        scope="private", owner_id="kendra",
        now=now,
    )
    pipeline.ingest(
        content="Kendra prefers Nike.",
        scope="private", owner_id="kendra",
        now=now,
        unique_predicates={"PREFERS"},
    )
    # At least one CONTRADICTS edge should exist
    rows = pipeline._backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS c FROM edges WHERE relation_type = 'CONTRADICTS'"
    ).fetchone()
    assert rows["c"] >= 1


def test_jaccard_triple_dedup(pipeline) -> None:
    """Two near-identical sentences should not produce two near-identical
    triple rows."""
    now = datetime.now(UTC)
    pipeline.ingest(content="Bob owns Acme Corp.", scope="shared", now=now)
    pipeline.ingest(content="Bob owns the Acme Corp.", scope="shared", now=now)
    rows = pipeline._backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS c FROM triples"
    ).fetchone()
    # We expect dedup to collapse the second to <=1 new triple beyond the first.
    assert rows["c"] <= 3
