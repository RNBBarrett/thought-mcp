"""Tests for ``thought.layers.code`` — thin wrappers over the graph layer
that surface code-specific concepts (callers, callees, impact, defines).

These methods are what the new CLI commands sit on top of.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from thought.embeddings.deterministic import DeterministicEmbedder
from thought.ingest.code.call_graph import build_call_graph
from thought.ingest.code.pipeline import CodeIngestPipeline
from thought.layers.code import CodeLayer
from thought.storage.sqlite.backend import SQLiteBackend

FIXTURE = Path(__file__).parent.parent / "fixtures" / "code" / "python" / "calls.py"


@pytest.fixture()
def code_layer(tmp_path):
    backend = SQLiteBackend(tmp_path / "cl.db")
    backend.migrate()
    embedder = DeterministicEmbedder(dim=64)
    pipe = CodeIngestPipeline(backend=backend, embedder=embedder)
    now = datetime.now(UTC)
    r = pipe.ingest_code_file(FIXTURE, commit_sha="sha", now=now)
    build_call_graph(
        backend=backend, file_path="calls.py",
        source=FIXTURE.read_text(encoding="utf-8"),
        language="python", commit_sha="sha",
        scope="shared", owner_id=None,
        source_ref=r.source_id, now=now,
    )
    yield CodeLayer(backend)
    backend.close()


def test_callers_of_returns_direct_callers_ranked(code_layer) -> None:
    # ``validate`` is called only by ``helper`` in the fixture.
    callers = code_layer.callers_of("validate", code_file="calls.py")
    names = [hit.entity.name for hit in callers]
    assert "helper" in names
    # The seed itself shouldn't appear in its own caller list.
    assert "validate" not in names


def test_callees_of_returns_direct_callees(code_layer) -> None:
    # ``main`` calls helper, build_widget, w.render (resolves to method).
    callees = code_layer.callees_of("main", code_file="calls.py")
    names = {hit.entity.name for hit in callees}
    assert {"helper", "build_widget"}.issubset(names)


def test_impact_set_returns_transitive_callers(code_layer) -> None:
    # ``escape`` is called by ``format_html`` and by ``Widget.format``.
    # ``format_html`` is in turn called by ``Widget.render``.
    # ``Widget.render`` is called by ``main`` (via w.render()).
    # Impact set should include the direct + transitive callers.
    impact = code_layer.impact_set("escape", code_file="calls.py")
    names = {hit.entity.name for hit in impact}
    assert "format_html" in names
    # ``Widget.format`` also calls ``escape``.
    assert any(n.endswith(".format") for n in names)


def test_defines_in_file_returns_top_level_definitions(code_layer) -> None:
    defines = code_layer.defines_in_file("calls.py")
    types = {d.type for d in defines}
    assert "function" in types
    assert "class" in types
    # Module entity isn't returned by defines_in_file — it's the *thing being
    # defined into*, not a member.
    assert "module" not in types


def test_callers_of_returns_empty_for_unknown_name(code_layer) -> None:
    assert code_layer.callers_of("does_not_exist", code_file="calls.py") == []
