"""Tests for the v0.4 Cypher subset: lex / parse / compile / execute / views / ask."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.memory import Memory
from thought.query import ask as ask_mod
from thought.query import cypher
from thought.query import views as views_mod

# ---------------------------------------------------------------- fixture

@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "q.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    now = datetime.now(UTC)
    src = m._backend.upsert_source("seed")
    alice = m._backend.upsert_entity(
        type_="PERSON", name="Alice", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    bob = m._backend.upsert_entity(
        type_="PERSON", name="Bob", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    acme = m._backend.upsert_entity(
        type_="ORGANIZATION", name="Acme", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    m._backend.upsert_edge(
        source_id=alice, target_id=acme, relation_type="WORKS_AT",
        source_ref=src, confidence_score=0.9, valid_from=now, learned_at=now,
    )
    m._backend.upsert_edge(
        source_id=bob, target_id=acme, relation_type="WORKS_AT",
        source_ref=src, confidence_score=0.9, valid_from=now, learned_at=now,
    )
    yield m
    m.close()


# ---------------------------------------------------------------- lexer

def test_tokenize_basic() -> None:
    tokens = cypher.tokenize("MATCH (p:PERSON) RETURN p")
    kinds = [t.kind for t in tokens]
    assert "MATCH" in kinds
    assert "LPAREN" in kinds
    assert "COLON" in kinds
    assert "IDENT" in kinds
    assert kinds[-1] == "EOF"


def test_tokenize_arrow_with_relation() -> None:
    tokens = cypher.tokenize("MATCH (a)-[r:WORKS_AT]->(b) RETURN a")
    arrow = next(t for t in tokens if t.kind == "ARROW_R")
    assert "WORKS_AT" in arrow.value


def test_tokenize_unsupported_keyword_raises() -> None:
    with pytest.raises(cypher.UnsupportedCypher, match="MERGE"):
        cypher.tokenize("MERGE (n:X)")


# ---------------------------------------------------------------- parser

def test_parse_simple_match() -> None:
    q = cypher.parse("MATCH (p:PERSON) RETURN p")
    assert len(q.patterns) == 1
    assert q.patterns[0].head.type_ == "PERSON"
    assert q.return_items[0].var == "p"


def test_parse_property_map() -> None:
    q = cypher.parse('MATCH (p:PERSON {name: "Alice"}) RETURN p')
    assert q.patterns[0].head.props == {"name": "Alice"}


def test_parse_edge_pattern() -> None:
    q = cypher.parse("MATCH (a)-[:WORKS_AT]->(b) RETURN a, b")
    assert len(q.patterns[0].steps) == 1
    step, _node = q.patterns[0].steps[0]
    assert step.relation == "WORKS_AT"
    assert step.direction == "forward"


def test_parse_where_clause() -> None:
    q = cypher.parse('MATCH (p:PERSON) WHERE p.name = "Alice" RETURN p')
    assert q.where is not None
    assert len(q.where.terms) == 1


def test_parse_limit_and_skip() -> None:
    q = cypher.parse("MATCH (p:PERSON) RETURN p LIMIT 10 SKIP 5")
    assert q.limit == 10
    assert q.skip == 5


def test_parse_as_of() -> None:
    q = cypher.parse('MATCH (p:PERSON) RETURN p AS_OF "2026-01-01"')
    assert q.as_of == "2026-01-01"


def test_parse_rejects_var_length_path() -> None:
    with pytest.raises(cypher.UnsupportedCypher, match="variable-length"):
        cypher.parse("MATCH (a)-[:R*1..3]->(b) RETURN a")


def test_parse_rejects_or_in_where() -> None:
    with pytest.raises(cypher.UnsupportedCypher, match="OR"):
        cypher.parse('MATCH (p) WHERE p.a = 1 OR p.b = 2 RETURN p')


def test_parse_rejects_non_match_start() -> None:
    with pytest.raises(cypher.CypherSyntaxError, match="MATCH"):
        cypher.parse("RETURN 1")


# ---------------------------------------------------------------- compiler

def test_compile_match_returns_sql_and_params() -> None:
    q = cypher.parse('MATCH (p:PERSON {name: "Alice"}) RETURN p.name')
    sql, params, columns = cypher.compile_to_sql(q)
    assert "FROM entities" in sql
    assert "type = ?" in sql
    assert "Alice" in params
    assert columns == ["p.name"]


def test_compile_edge_pattern() -> None:
    q = cypher.parse("MATCH (a:PERSON)-[:WORKS_AT]->(b:ORGANIZATION) RETURN a, b")
    sql, params, _columns = cypher.compile_to_sql(q)
    assert "JOIN edges" in sql
    assert "JOIN entities" in sql
    assert "WORKS_AT" in params


def test_compile_unknown_var_in_return_raises() -> None:
    q = cypher.parse("MATCH (p:PERSON) RETURN q")
    with pytest.raises(cypher.CypherSyntaxError, match="unknown variable"):
        cypher.compile_to_sql(q)


# ---------------------------------------------------------------- executor

def test_execute_match_by_type(mem) -> None:
    rows = cypher.execute(mem, "MATCH (p:PERSON) RETURN p.name")
    names = {r["p.name"] for r in rows}
    assert names == {"Alice", "Bob"}


def test_execute_match_by_property(mem) -> None:
    rows = cypher.execute(mem, 'MATCH (p:PERSON {name: "Alice"}) RETURN p.name')
    assert rows == [{"p.name": "Alice"}]


def test_execute_pattern_across_edge(mem) -> None:
    rows = cypher.execute(
        mem, "MATCH (p:PERSON)-[:WORKS_AT]->(o:ORGANIZATION) RETURN p.name, o.name",
    )
    pairs = {(r["p.name"], r["o.name"]) for r in rows}
    assert pairs == {("Alice", "Acme"), ("Bob", "Acme")}


def test_execute_where_clause(mem) -> None:
    rows = cypher.execute(mem, 'MATCH (p:PERSON) WHERE p.name = "Alice" RETURN p.name')
    assert rows == [{"p.name": "Alice"}]


def test_execute_limit(mem) -> None:
    rows = cypher.execute(mem, "MATCH (p:PERSON) RETURN p.name LIMIT 1")
    assert len(rows) == 1


def test_execute_full_entity_returns_json(mem) -> None:
    rows = cypher.execute(mem, 'MATCH (p:PERSON {name: "Alice"}) RETURN p')
    assert len(rows) == 1
    assert isinstance(rows[0]["p"], dict)
    assert rows[0]["p"]["name"] == "Alice"


def test_execute_starts_with(mem) -> None:
    rows = cypher.execute(
        mem, 'MATCH (p:PERSON) WHERE p.name STARTS WITH "Al" RETURN p.name',
    )
    assert {r["p.name"] for r in rows} == {"Alice"}


# ---------------------------------------------------------------- saved views

def test_save_and_run_view(mem) -> None:
    views_mod.save_view(mem, "all_people", "MATCH (p:PERSON) RETURN p.name")
    rows = views_mod.run_view(mem, "all_people")
    assert {r["p.name"] for r in rows} == {"Alice", "Bob"}


def test_view_replace_overwrites(mem) -> None:
    views_mod.save_view(mem, "x", "MATCH (p:PERSON) RETURN p.name")
    views_mod.save_view(mem, "x", "MATCH (o:ORGANIZATION) RETURN o.name", replace=True)
    rows = views_mod.run_view(mem, "x")
    assert {r["o.name"] for r in rows} == {"Acme"}


def test_view_invalid_name_rejected(mem) -> None:
    with pytest.raises(views_mod.ViewError, match="invalid view name"):
        views_mod.save_view(mem, "bad-name; DROP TABLE entities", "MATCH (p) RETURN p")


def test_view_validates_cypher_on_save(mem) -> None:
    with pytest.raises(cypher.UnsupportedCypher):
        views_mod.save_view(mem, "bad", "MERGE (n:X)")


def test_list_and_delete_views(mem) -> None:
    views_mod.save_view(mem, "v1", "MATCH (p:PERSON) RETURN p")
    views_mod.save_view(mem, "v2", "MATCH (o:ORGANIZATION) RETURN o")
    assert {v["name"] for v in views_mod.list_views(mem)} == {"v1", "v2"}
    assert views_mod.delete_view(mem, "v1") is True
    assert {v["name"] for v in views_mod.list_views(mem)} == {"v2"}


def test_run_view_missing_raises(mem) -> None:
    with pytest.raises(views_mod.ViewError, match="no saved view"):
        views_mod.run_view(mem, "nope")


# ---------------------------------------------------------------- thought ask (mocked)

def test_ask_no_provider_falls_back_to_recall(mem) -> None:
    """provider='none' should fall back to plain recall."""
    class FakeLLM:
        provider = "none"
    r = ask_mod.ask(mem, "who is in the KB?", llm_cfg=FakeLLM())
    assert r.fallback_used
    # Recall might or might not return hits with deterministic embedder;
    # the contract is that we got back AskResult.rows, not None.
    assert r.rows is not None


def test_ask_no_provider_with_no_fallback_errors(mem) -> None:
    class FakeLLM:
        provider = "none"
    r = ask_mod.ask(mem, "x", llm_cfg=FakeLLM(), no_fallback=True)
    assert r.error is not None
    assert "[llm] provider" in r.error


def test_ask_translates_to_cypher_via_mocked_anthropic(mem) -> None:
    """Mock anthropic to return a valid Cypher query; assert it gets executed."""
    fake_translation = 'MATCH (p:PERSON {name: "Alice"}) RETURN p.name'

    class _Block:
        type = "text"
        text = fake_translation

    class _Resp:
        content: list = [_Block()]  # noqa: RUF012 — test-only mock

    class _Messages:
        def create(self, **_kw): return _Resp()

    class _FakeAnthropic:
        def __init__(self, *_a, **_kw): pass
        @property
        def messages(self): return _Messages()

    class FakeLLM:
        provider = "anthropic"
        model = None

    import sys as _sys
    saved = _sys.modules.get("anthropic")
    _sys.modules["anthropic"] = type("anthropic", (), {"Anthropic": _FakeAnthropic})  # type: ignore[assignment]
    try:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            r = ask_mod.ask(mem, "who is Alice?", llm_cfg=FakeLLM())
    finally:
        if saved is None:
            del _sys.modules["anthropic"]
        else:
            _sys.modules["anthropic"] = saved
    assert not r.fallback_used
    assert r.cypher == fake_translation
    assert r.rows == [{"p.name": "Alice"}]


def test_ask_invalid_cypher_falls_back(mem) -> None:
    """If the LLM emits garbage, we fall back to recall."""
    class _Block:
        type = "text"
        text = "this is not cypher at all 🤖"

    class _Resp:
        content: list = [_Block()]  # noqa: RUF012 — test-only mock

    class _Messages:
        def create(self, **_kw): return _Resp()

    class _FakeAnthropic:
        def __init__(self, *_a, **_kw): pass
        @property
        def messages(self): return _Messages()

    class FakeLLM:
        provider = "anthropic"
        model = None

    import sys as _sys
    saved = _sys.modules.get("anthropic")
    _sys.modules["anthropic"] = type("anthropic", (), {"Anthropic": _FakeAnthropic})  # type: ignore[assignment]
    try:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            r = ask_mod.ask(mem, "what?", llm_cfg=FakeLLM())
    finally:
        if saved is None:
            del _sys.modules["anthropic"]
        else:
            _sys.modules["anthropic"] = saved
    assert r.fallback_used
    assert "invalid" in (r.fallback_reason or "").lower()


# ---------------------------------------------------------------- CLI

@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cli_init_with_data(workspace: Path, runner: CliRunner) -> None:
    r = runner.invoke(
        app,
        ["init", "--db-path", str(workspace / ".thought" / "q.db"),
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0, r.stdout
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])


def test_schema_cli(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, ["schema", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert "entity_types" in data
    assert "relation_types" in data


def test_query_cli(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, ["query", "MATCH (p:CONCEPT) RETURN p.name LIMIT 5", "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert "rows" in data


def test_query_cli_explain(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, ["query", "MATCH (p:CONCEPT) RETURN p.name", "--explain"])
    assert r.exit_code == 0, r.stdout
    assert "SQL:" in r.stdout


def test_query_cli_bad_cypher_exits_2(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, ["query", "RETURN 1"])
    assert r.exit_code == 2


def test_view_save_run_delete_cli(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, [
        "view", "save", "concepts", "MATCH (c:CONCEPT) RETURN c.name",
    ])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["view", "list", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert any(v["name"] == "concepts" for v in data["views"])
    r = runner.invoke(app, ["view", "run", "concepts", "--json"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["view", "delete", "concepts"])
    assert r.exit_code == 0


def test_view_run_missing_exits_1(workspace) -> None:
    runner = CliRunner()
    _cli_init_with_data(workspace, runner)
    r = runner.invoke(app, ["view", "run", "nope"])
    assert r.exit_code == 1
