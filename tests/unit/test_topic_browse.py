"""Tests for Phase 1: topic browsing.

Covers:
- ``backend.count_by_type`` — aggregation by entity type.
- ``backend.find_anchor_by_name`` — name→Entity resolution.
- ``Memory.list_topics`` / ``Memory.browse_topic`` — facade.
- ``thought topics`` / ``thought browse`` CLI commands.
- ``mcp__thought__list_topics`` / ``browse_topic`` tools.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.memory import Memory
from thought.models import ScopeFilter

MCP_INSTALLED = find_spec("mcp") is not None


@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "topics.db"),
        embedder_choice="deterministic",
        embedder_dim=128,
    )
    # Seed with a varied set of entities by directly upserting — letting the
    # ingest pipeline pick types from text would couple this test to the
    # extractor's heuristics, which isn't what we're testing here.
    now = datetime.now(UTC)
    src = m._backend.upsert_source("test fixture")
    for type_, name in [
        ("PERSON", "Alice"), ("PERSON", "Bob"), ("PERSON", "Dana"),
        ("ORGANIZATION", "Acme"), ("ORGANIZATION", "Beta"),
        ("CONCEPT", "dessert"), ("CONCEPT", "donut"),
        ("CONCEPT", "cake"), ("CONCEPT", "pastry"),
    ]:
        m._backend.upsert_entity(
            type_=type_, name=name, scope="shared",
            valid_from=now, learned_at=now, source_ref=src,
        )
    yield m
    m.close()


# ---------------------------------------------------------------- backend

def test_count_by_type_groups_correctly(mem) -> None:
    counts = mem._backend.count_by_type(ScopeFilter(scope="all"))
    assert counts == {"CONCEPT": 4, "PERSON": 3, "ORGANIZATION": 2}


def test_count_by_type_respects_scope(mem) -> None:
    # All seeded entities are scope=shared; private scope (no owner) is empty.
    counts = mem._backend.count_by_type(ScopeFilter(scope="private"))
    assert counts == {}


def test_find_anchor_by_name_case_insensitive(mem) -> None:
    e = mem._backend.find_anchor_by_name("ACME", ScopeFilter(scope="all"))
    assert e is not None
    assert e.name == "Acme"
    assert e.type == "ORGANIZATION"


def test_find_anchor_by_name_returns_none_for_unknown(mem) -> None:
    e = mem._backend.find_anchor_by_name("Nonesuch", ScopeFilter(scope="all"))
    assert e is None


# ---------------------------------------------------------------- facade

def test_list_topics_returns_sorted_counts_with_examples(mem) -> None:
    topics = mem.list_topics(scope="all", examples_per_type=2)
    assert [t["type"] for t in topics] == ["CONCEPT", "PERSON", "ORGANIZATION"]
    assert topics[0]["count"] == 4
    assert len(topics[0]["examples"]) == 2


def test_list_topics_min_count_filter(mem) -> None:
    topics = mem.list_topics(scope="all", min_count=3)
    # Only CONCEPT(4) and PERSON(3) clear the bar.
    assert {t["type"] for t in topics} == {"CONCEPT", "PERSON"}


def test_browse_topic_by_type_returns_entities_of_that_type(mem) -> None:
    items = mem.browse_topic("CONCEPT", limit=10)
    assert items, items
    assert all(it["type"] == "CONCEPT" for it in items)
    assert all(it["via"] == "type_facet" for it in items)


def test_browse_topic_by_type_case_insensitive(mem) -> None:
    items = mem.browse_topic("concept", limit=10)
    assert items and all(it["type"] == "CONCEPT" for it in items)


def test_browse_topic_unknown_name_returns_empty(mem) -> None:
    assert mem.browse_topic("Nonesuch") == []


def test_browse_topic_by_entity_name_uses_graph(mem, tmp_path) -> None:
    # Add an edge so the graph layer has something to traverse.
    now = datetime.now(UTC)
    src = mem._backend.upsert_source("edge fixture")
    alice = mem._backend.find_anchor_by_name("Alice", ScopeFilter(scope="all"))
    acme = mem._backend.find_anchor_by_name("Acme", ScopeFilter(scope="all"))
    assert alice is not None and acme is not None
    mem._backend.upsert_edge(
        source_id=alice.id, target_id=acme.id,
        relation_type="WORKS_AT", source_ref=src,
        confidence_score=0.9,
        valid_from=now, learned_at=now,
    )
    items = mem.browse_topic("Alice", limit=5)
    assert items, items
    assert items[0]["via"] in {"ppr", "bfs"}
    assert any(it["name"] == "Acme" for it in items)


# ---------------------------------------------------------------- CLI

def _init_cli(tmp_path: Path, monkeypatch) -> CliRunner:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["init", "--db-path", ".thought/cli.db",
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0, r.stdout
    return runner


def test_topics_cli_renders_table(tmp_path, monkeypatch) -> None:
    runner = _init_cli(tmp_path, monkeypatch)
    runner.invoke(app, ["ingest", "Alice owns Acme Corp.", "--scope", "shared"])
    r = runner.invoke(app, ["topics"])
    assert r.exit_code == 0, r.stdout
    assert "Topics" in r.stdout or "topic" in r.stdout.lower()


def test_topics_cli_json(tmp_path, monkeypatch) -> None:
    runner = _init_cli(tmp_path, monkeypatch)
    runner.invoke(app, ["ingest", "Alice owns Acme Corp.", "--scope", "shared"])
    r = runner.invoke(app, ["topics", "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert "topics" in data
    assert isinstance(data["topics"], list)


def test_browse_cli_by_type(tmp_path, monkeypatch) -> None:
    runner = _init_cli(tmp_path, monkeypatch)
    runner.invoke(app, ["ingest", "Alice owns Acme Corp.", "--scope", "shared"])
    r = runner.invoke(app, ["browse", "CONCEPT", "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    # The deterministic extractor produces CONCEPT entities for the proper
    # nouns, so we should get at least one match.
    assert "items" in data


def test_browse_cli_unknown_exits_1(tmp_path, monkeypatch) -> None:
    runner = _init_cli(tmp_path, monkeypatch)
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])
    r = runner.invoke(app, ["browse", "Nonesuch"])
    assert r.exit_code == 1


# ---------------------------------------------------------------- MCP tools

@pytest.mark.skipif(not MCP_INSTALLED, reason="mcp package not installed")
def test_mcp_list_topics_tool(tmp_path) -> None:
    from thought.server import build_app
    m = Memory.open(
        db_path=str(tmp_path / "mcp.db"),
        embedder_choice="deterministic", embedder_dim=128,
    )
    try:
        now = datetime.now(UTC)
        src = m._backend.upsert_source("seed")
        for t, n in [("PERSON", "X"), ("PERSON", "Y"), ("CONCEPT", "Z")]:
            m._backend.upsert_entity(
                type_=t, name=n, scope="shared",
                valid_from=now, learned_at=now, source_ref=src,
            )
        app_ = build_app(m)
        r = asyncio.run(app_.call_tool("list_topics", {}))
        payload = json.loads(r[0].text)  # type: ignore[index,union-attr]
        assert "topics" in payload
        types = {t["type"]: t["count"] for t in payload["topics"]}
        assert types == {"PERSON": 2, "CONCEPT": 1}
    finally:
        m.close()


@pytest.mark.skipif(not MCP_INSTALLED, reason="mcp package not installed")
def test_mcp_browse_topic_by_type(tmp_path) -> None:
    from thought.server import build_app
    m = Memory.open(
        db_path=str(tmp_path / "mcp2.db"),
        embedder_choice="deterministic", embedder_dim=128,
    )
    try:
        now = datetime.now(UTC)
        src = m._backend.upsert_source("seed")
        for n in ["donut", "cake", "pastry"]:
            m._backend.upsert_entity(
                type_="CONCEPT", name=n, scope="shared",
                valid_from=now, learned_at=now, source_ref=src,
            )
        app_ = build_app(m)
        r = asyncio.run(app_.call_tool(
            "browse_topic", {"name": "CONCEPT", "limit": 10},
        ))
        payload = json.loads(r[0].text)  # type: ignore[index,union-attr]
        assert "items" in payload
        names = {it["name"] for it in payload["items"]}
        assert names == {"donut", "cake", "pastry"}
    finally:
        m.close()
