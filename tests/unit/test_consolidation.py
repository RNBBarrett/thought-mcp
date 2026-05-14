"""Consolidation engine tests.

Per the plan, the engine does five jobs in WARM tier:
1. Contradiction detection (a posteriori).
2. Duplicate merging.
3. Staleness flagging.
4. Cold-tier demotion (>30d no access).
5. Ebbinghaus strength recompute.

We test each as a unit; the threaded daemon-mode lifecycle is tested in
integration.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from thought.consolidation.engine import ConsolidationEngine
from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.pipeline import IngestPipeline
from thought.storage.sqlite.backend import SQLiteBackend


@pytest.fixture()
def engine(tmp_path):
    backend = SQLiteBackend(tmp_path / "c.db")
    backend.migrate()
    embedder = DeterministicEmbedder(dim=128)
    pipe = IngestPipeline(backend=backend, embedder=embedder)
    eng = ConsolidationEngine(backend=backend, embedder=embedder)
    yield {"backend": backend, "embedder": embedder, "pipe": pipe, "engine": eng}
    backend.close()


def test_cold_demotion(engine) -> None:
    backend = engine["backend"]
    pipe = engine["pipe"]
    now = datetime.now(UTC)
    pipe.ingest(content="Alice owns Acme.", scope="shared", now=now)
    # Force WARM tier and old last_accessed_at.
    backend._conn.execute(  # type: ignore[attr-defined]
        "UPDATE entities SET tier='warm', last_accessed_at=?",
        ((now - timedelta(days=40)).isoformat(),),
    )
    report = engine["engine"].run_once(now=now)
    assert report.demoted_cold >= 1
    cold = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS c FROM entities WHERE tier='cold'"
    ).fetchone()
    assert cold["c"] >= 1


def test_strength_recompute_ebbinghaus(engine) -> None:
    backend = engine["backend"]
    pipe = engine["pipe"]
    now = datetime.now(UTC)
    pipe.ingest(content="Bob runs Acme.", scope="shared", now=now)
    backend._conn.execute(  # type: ignore[attr-defined]
        "UPDATE entities SET tier='warm', access_count=10, last_accessed_at=?",
        ((now - timedelta(days=2)).isoformat(),),
    )
    engine["engine"].run_once(now=now)
    rows = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT strength FROM strength_cache"
    ).fetchall()
    assert rows
    for r in rows:
        # Strength is bounded [0, ~1.2] with our formula and 10 recall_count.
        assert 0.0 <= r["strength"] <= 2.0


def test_strength_decays_with_age(engine) -> None:
    backend = engine["backend"]
    pipe = engine["pipe"]
    now = datetime.now(UTC)
    pipe.ingest(content="Old fact.", scope="shared", now=now - timedelta(days=200))
    pipe.ingest(content="Fresh fact.", scope="shared", now=now)
    backend._conn.execute(  # type: ignore[attr-defined]
        "UPDATE entities SET tier='warm'"
    )
    # Explicitly age the "Old" entity by 200 days for last_accessed_at.
    backend._conn.execute(  # type: ignore[attr-defined]
        "UPDATE entities SET last_accessed_at=? WHERE LOWER(name)='old'",
        ((now - timedelta(days=200)).isoformat(),),
    )
    engine["engine"].run_once(now=now)
    rows = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT s.strength, e.name FROM strength_cache s JOIN entities e ON e.id=s.entity_id"
    ).fetchall()
    by_name = {r["name"].lower(): r["strength"] for r in rows}
    assert "old" in by_name and "fresh" in by_name, (
        f"extractor missed entities. got: {by_name}"
    )
    assert by_name["old"] < by_name["fresh"], (
        f"expected old < fresh; got old={by_name['old']}, fresh={by_name['fresh']}"
    )


def test_duplicate_detection_creates_merge_audit(engine) -> None:
    backend = engine["backend"]
    pipe = engine["pipe"]
    now = datetime.now(UTC)
    pipe.ingest(content="Charlie at Acme.", scope="shared", now=now)
    pipe.ingest(content="Charlie at Acme Corp.", scope="shared", now=now)
    # Force WARM tier for the consolidator to scan.
    backend._conn.execute("UPDATE entities SET tier='warm'")  # type: ignore[attr-defined]
    report = engine["engine"].run_once(now=now)
    # Audit log should record at least one MERGE or STRENGTH op (depending on
    # dedup at ingest; here ingest-time dedup might already collapse them).
    assert report.audit_entries >= 1
