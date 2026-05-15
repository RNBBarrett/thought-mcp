"""Direct tests for the MCP tool handlers exposed by :mod:`thought.server`.

These exercise the ``remember`` / ``recall`` tool dispatch via FastMCP's
``call_tool`` without binding either transport. Faster than the integration
tests (no subprocess, no MCP client) and catch the same class of bug that the
v0.2.1 ship missed — the tool handlers offloading via ``asyncio.to_thread``
into a worker thread that wasn't allowed to touch the SQLite connection.
"""
from __future__ import annotations

import asyncio
import json
from importlib.util import find_spec

import pytest

from thought.memory import Memory
from thought.server import build_app

MCP_INSTALLED = find_spec("mcp") is not None

pytestmark = pytest.mark.skipif(
    not MCP_INSTALLED, reason="mcp package not installed",
)


@pytest.fixture()
def app_and_mem(tmp_path):
    mem = Memory.open(
        db_path=str(tmp_path / "s.db"),
        embedder_choice="deterministic",
        embedder_dim=128,
    )
    try:
        yield build_app(mem), mem
    finally:
        mem.close()


def _payload(call_result) -> dict:
    """FastMCP returns ``list[TextContent]``; unwrap the first text frame."""
    assert isinstance(call_result, list), call_result
    assert call_result, "tool returned no content"
    return json.loads(call_result[0].text)


def test_tools_register_with_expected_names(app_and_mem) -> None:
    app, _ = app_and_mem
    tm = app._tool_manager  # internal but stable across mcp 1.x
    # v0.3 adds list_topics + browse_topic alongside the v0.1 remember + recall.
    assert set(tm._tools.keys()) == {
        "remember", "recall", "list_topics", "browse_topic",
    }


def test_remember_returns_source_and_entities(app_and_mem) -> None:
    app, _ = app_and_mem
    out = _payload(asyncio.run(
        app.call_tool("remember", {"content": "Alice owns Acme Corp."})
    ))
    assert out["source_id"]
    assert out["entity_ids"]
    assert out["duplicate_of_source"] is None


def test_remember_dedups_identical_content(app_and_mem) -> None:
    app, _ = app_and_mem
    first = _payload(asyncio.run(
        app.call_tool("remember", {"content": "Hello world."})
    ))
    second = _payload(asyncio.run(
        app.call_tool("remember", {"content": "Hello world."})
    ))
    assert second["duplicate_of_source"] == first["source_id"]


def test_recall_returns_query_classification(app_and_mem) -> None:
    app, _ = app_and_mem
    asyncio.run(app.call_tool(
        "remember", {"content": "Alice owns Acme Corp."},
    ))
    out = _payload(asyncio.run(
        app.call_tool("recall", {"query": "alice", "limit": 5}),
    ))
    assert out["query_class"] in {"VIBE", "FACT", "CHANGE", "HYBRID", "CODE"}
    assert "elapsed_ms" in out
    assert isinstance(out["signals"], dict)


def test_recall_empty_kb_returns_low_confidence(app_and_mem) -> None:
    app, _ = app_and_mem
    out = _payload(asyncio.run(
        app.call_tool("recall", {"query": "anything", "limit": 5})
    ))
    assert out["hits"] == []
    assert out["low_confidence"] is True


def test_recall_respects_limit_param(app_and_mem) -> None:
    app, _ = app_and_mem
    for txt in [
        "Alice owns Acme.", "Alice is a founder.",
        "Acme is a startup.", "Bob works at Acme.",
    ]:
        asyncio.run(app.call_tool("remember", {"content": txt}))
    out = _payload(asyncio.run(
        app.call_tool("recall", {"query": "Acme", "limit": 2})
    ))
    assert len(out["hits"]) <= 2


def test_tool_call_survives_threadpool_dispatch(app_and_mem) -> None:
    """Regression: v0.2.1 had ``check_same_thread=True`` on the SQLite
    connection, so the ``asyncio.to_thread`` dispatch inside the tool
    handler raised ``ProgrammingError: SQLite objects created in a thread
    can only be used in that same thread``. This test pins the fix.
    """
    app, _ = app_and_mem
    # Two back-to-back calls force the to_thread dispatch path twice; on
    # the bug they'd both raise from the worker thread.
    for _ in range(2):
        out = _payload(asyncio.run(
            app.call_tool("remember", {"content": "x" * 16})
        ))
        assert out["source_id"]
