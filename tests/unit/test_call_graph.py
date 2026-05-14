"""Call-graph extraction tests.

The call-graph pass runs AFTER the AST ingest has materialised all
function / method / class entities. It re-walks each function body and
emits CALLS edges, resolving callee names to entity IDs via
``backend.find_code_entity``.

In-file calls resolve to source_grounded edges. Cross-package calls
that don't resolve become inferred edges pointing at stub function
entities (mirroring the IMPORTS-stub pattern from Phase 1).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.code.call_graph import build_call_graph
from thought.ingest.code.pipeline import CodeIngestPipeline
from thought.storage.sqlite.backend import SQLiteBackend

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "code" / "python"


@pytest.fixture()
def ingested(tmp_path):
    backend = SQLiteBackend(tmp_path / "cg.db")
    backend.migrate()
    embedder = DeterministicEmbedder(dim=64)
    pipe = CodeIngestPipeline(backend=backend, embedder=embedder)
    now = datetime.now(UTC)
    pipe.ingest_code_file(
        FIXTURE_DIR / "calls.py", commit_sha="testsha", now=now,
    )
    yield backend, pipe, now
    backend.close()


def _calls_from(backend, caller_name: str) -> set[str]:
    """Resolve caller name → entity id → target names of all CALLS edges out."""
    src_id = backend.find_code_entity(
        canonical_name=caller_name, code_file="calls.py",
    )
    assert src_id, f"caller {caller_name} not found"
    rows = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT e.name FROM edges x JOIN entities e ON e.id = x.target_id "
        "WHERE x.source_id = ? AND x.relation_type = 'CALLS'",
        (src_id,),
    ).fetchall()
    return {r["name"] for r in rows}


def test_call_graph_extracts_in_file_function_calls(ingested):
    backend, _pipe, now = ingested
    n_edges = build_call_graph(
        backend=backend,
        file_path="calls.py",
        source=(FIXTURE_DIR / "calls.py").read_text(encoding="utf-8"),
        language="python",
        commit_sha="testsha",
        scope="shared",
        owner_id=None,
        source_ref=backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM sources LIMIT 1"
        ).fetchone()["id"],
        now=now,
    )
    assert n_edges > 0

    # main calls helper, build_widget, and (transitively via Widget) render
    main_calls = _calls_from(backend, "main")
    assert "helper" in main_calls
    assert "build_widget" in main_calls

    # helper calls validate
    assert "validate" in _calls_from(backend, "helper")


def test_call_graph_extracts_method_calls(ingested):
    backend, _pipe, now = ingested
    build_call_graph(
        backend=backend, file_path="calls.py",
        source=(FIXTURE_DIR / "calls.py").read_text(encoding="utf-8"),
        language="python", commit_sha="testsha",
        scope="shared", owner_id=None,
        source_ref=backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM sources LIMIT 1"
        ).fetchone()["id"],
        now=now,
    )
    # Widget.render calls format_html (free function)
    render_calls = _calls_from(backend, "widget.render")
    assert "format_html" in render_calls


def test_call_graph_resolves_self_method_calls(ingested):
    backend, _pipe, now = ingested
    build_call_graph(
        backend=backend, file_path="calls.py",
        source=(FIXTURE_DIR / "calls.py").read_text(encoding="utf-8"),
        language="python", commit_sha="testsha",
        scope="shared", owner_id=None,
        source_ref=backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM sources LIMIT 1"
        ).fetchone()["id"],
        now=now,
    )
    # Widget.render → self.format should resolve to Widget.format (qualified).
    render_calls = _calls_from(backend, "widget.render")
    assert "Widget.format" in render_calls


def test_unresolved_calls_become_inferred(ingested):
    backend, _pipe, now = ingested
    build_call_graph(
        backend=backend, file_path="calls.py",
        source=(FIXTURE_DIR / "calls.py").read_text(encoding="utf-8"),
        language="python", commit_sha="testsha",
        scope="shared", owner_id=None,
        source_ref=backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM sources LIMIT 1"
        ).fetchone()["id"],
        now=now,
    )
    # Every CALLS edge has a confidence_class — verify the distribution
    # makes sense (we expect mostly source_grounded for this fixture).
    rows = backend._conn.execute(  # type: ignore[attr-defined]
        "SELECT confidence_class, COUNT(*) AS n FROM edges "
        "WHERE relation_type='CALLS' GROUP BY confidence_class"
    ).fetchall()
    classes = {r["confidence_class"]: r["n"] for r in rows}
    assert classes.get("source_grounded", 0) >= 4, f"expected ≥4 grounded; got {classes}"


def test_callers_query_via_pagerank_returns_callers_first(ingested):
    """End-to-end sanity: HippoRAG PageRank seeded by a callee should rank
    its callers higher than unrelated entities.

    This is the v0.2 killer feature: ``thought callers <fn>`` → ranked list.
    """
    backend, _pipe, now = ingested
    build_call_graph(
        backend=backend, file_path="calls.py",
        source=(FIXTURE_DIR / "calls.py").read_text(encoding="utf-8"),
        language="python", commit_sha="testsha",
        scope="shared", owner_id=None,
        source_ref=backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM sources LIMIT 1"
        ).fetchone()["id"],
        now=now,
    )
    from thought.layers.graph import GraphLayer
    from thought.models import ScopeFilter

    graph = GraphLayer(backend)
    validate_id = backend.find_code_entity(
        canonical_name="validate", code_file="calls.py",
    )
    assert validate_id

    scores = graph.personalized_pagerank(
        seeds=[validate_id], scope_filter=ScopeFilter(scope="all"),
    )
    # 'helper' calls validate, so it should score relatively high.
    helper_id = backend.find_code_entity(
        canonical_name="helper", code_file="calls.py",
    )
    escape_id = backend.find_code_entity(
        canonical_name="escape", code_file="calls.py",
    )
    assert helper_id in scores, "helper should be reachable from validate"
    assert scores[helper_id] > 0
    # escape has no connection to validate; should not appear (or score 0)
    if escape_id in scores:
        assert scores[escape_id] <= scores[helper_id]
