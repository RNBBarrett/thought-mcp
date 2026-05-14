"""CodeIngestPipeline tests — file/directory ingest end-to-end.

These cover ingest only (entities, IMPORTS, INHERITS_FROM, DEFINES). Call
graph extraction is a separate test module so we can iterate on each
independently.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.code.pipeline import CodeIngestPipeline
from thought.storage.sqlite.backend import SQLiteBackend

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "code" / "python"


@pytest.fixture()
def pipeline(tmp_path):
    backend = SQLiteBackend(tmp_path / "ci.db")
    backend.migrate()
    embedder = DeterministicEmbedder(dim=64)
    pipe = CodeIngestPipeline(backend=backend, embedder=embedder)
    yield pipe
    backend.close()


def test_ingest_python_file_creates_module_entity(pipeline) -> None:
    result = pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="testsha", now=datetime.now(UTC),
    )
    assert result.entity_ids, "expected at least the module entity"
    backend = pipeline._backend
    # The 'auth' module entity for the ingested file (stub modules for imports
    # exist alongside it — we don't pin total module count).
    row = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT name, type, code_file, code_language, code_commit_sha "
        "FROM entities WHERE type='module' AND code_file='auth.py'"
    ).fetchone()
    assert row is not None
    assert row["code_language"] == "python"
    assert row["code_commit_sha"] == "testsha"


def test_ingest_python_file_extracts_expected_entity_counts(pipeline) -> None:
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    # We check ONLY the entities that have code_file=auth.py — i.e. the things
    # actually parsed out of the file. Stubs for external imports / parent
    # classes have code_file=NULL and aren't counted.
    counts = dict(backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT type, COUNT(*) AS n FROM entities "
        "WHERE code_file='auth.py' GROUP BY type"
    ).fetchall())
    assert counts == {
        "module": 1,
        "function": 2,
        "class": 3,
        "method": 4,
    }


def test_ingest_emits_inherits_from_edge(pipeline) -> None:
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    # The fixture has 2 INHERITS_FROM relationships:
    #   AuthError → Exception      (stubbed parent)
    #   JWTAuth   → AuthBackend    (in-file parent)
    row = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM edges WHERE relation_type='INHERITS_FROM'"
    ).fetchone()
    assert row["n"] == 2


def test_ingest_emits_imports_edges(pipeline) -> None:
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    row = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM edges WHERE relation_type='IMPORTS'"
    ).fetchone()
    assert row["n"] >= 3  # jwt, datetime, .errors


def test_ingest_emits_defines_edges_for_methods(pipeline) -> None:
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    row = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM edges WHERE relation_type='DEFINES'"
    ).fetchone()
    assert row["n"] == 4  # one per method


def test_idempotent_re_ingest_same_commit_is_noop(pipeline) -> None:
    """Re-ingesting the same file at the same commit shouldn't create new rows."""
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    n_before = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM entities"
    ).fetchone()["n"]
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    n_after = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM entities"
    ).fetchone()["n"]
    assert n_before == n_after, (
        f"re-ingest at same commit should be idempotent; got {n_before}→{n_after}"
    )


def test_different_commit_creates_new_versions(pipeline) -> None:
    """Same file at a different commit produces a new set of entities with
    a different code_commit_sha — that's what makes bi-temporal queries work.
    """
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t1", now=datetime.now(UTC),
    )
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha="t2", now=datetime.now(UTC),
    )
    backend = pipeline._backend
    shas = {
        r["code_commit_sha"] for r in backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT DISTINCT code_commit_sha FROM entities WHERE code_commit_sha IS NOT NULL"
        ).fetchall()
    }
    assert shas == {"t1", "t2"}


def test_auto_detect_language_from_extension(pipeline) -> None:
    pipeline.ingest_code_file(
        FIXTURE_DIR / "auth.py", commit_sha=None, now=datetime.now(UTC),
        # language=None — should auto-detect
    )
    backend = pipeline._backend
    langs = {
        r["code_language"] for r in backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT DISTINCT code_language FROM entities WHERE code_language IS NOT NULL"
        ).fetchall()
    }
    assert langs == {"python"}
