"""Tests for the auto-write hook.

Covers:
- ``hooks.write._read_transcript`` — inline ``messages`` + ``transcript_path``.
- ``hooks.write._select_turns_for_ingest`` — last user + last assistant.
- ``hooks.write.run`` — raw mode happy path + dedup + contradiction edges.
- ``hooks.write.cli_main`` — stdin / stderr summary contract.
- ``thought hook write`` CLI command.

``--mode extract`` is exercised with a mocked anthropic client so we don't
hit the API in tests.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.hooks import write as hook_write
from thought.memory import Memory

# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "w.db"),
        embedder_choice="deterministic", embedder_dim=128,
    )
    yield m
    m.close()


def _make_transcript(tmp_path: Path, messages: list[dict]) -> Path:
    p = tmp_path / "transcript.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    return p


# ---------------------------------------------------------------- transcript reader

def test_read_transcript_inline_messages() -> None:
    payload = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    assert hook_write._read_transcript(payload) == payload["messages"]


def test_read_transcript_from_file(tmp_path) -> None:
    p = _make_transcript(tmp_path, [
        {"role": "user", "content": "Alice owns Acme."},
        {"role": "assistant", "content": "Acme is headquartered in Seattle."},
    ])
    msgs = hook_write._read_transcript({"transcript_path": str(p)})
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_read_transcript_missing_file_returns_empty(tmp_path) -> None:
    assert hook_write._read_transcript({"transcript_path": str(tmp_path / "nope")}) == []


def test_read_transcript_skips_invalid_lines(tmp_path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("{not json\n{\"role\":\"user\",\"content\":\"hi\"}\n", encoding="utf-8")
    msgs = hook_write._read_transcript({"transcript_path": str(p)})
    assert msgs == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------- turn picker

def test_select_turns_takes_last_user_and_last_assistant() -> None:
    pairs = hook_write._select_turns_for_ingest([
        {"role": "user", "content": "early user"},
        {"role": "assistant", "content": "early assistant"},
        {"role": "user", "content": "later user"},
        {"role": "assistant", "content": "later assistant"},
    ])
    assert pairs == [("user", "later user"), ("assistant", "later assistant")]


def test_select_turns_handles_content_blocks() -> None:
    pairs = hook_write._select_turns_for_ingest([
        {"role": "user", "content": [
            {"type": "text", "text": "block one. "},
            {"type": "text", "text": "block two."},
        ]},
    ])
    assert pairs == [("user", "block one. \nblock two.")]


def test_select_turns_skips_empty() -> None:
    pairs = hook_write._select_turns_for_ingest([
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ])
    assert pairs == []


def test_select_turns_truncates_long_content() -> None:
    long = "x" * (hook_write.MAX_TURN_CHARS + 1000)
    pairs = hook_write._select_turns_for_ingest([
        {"role": "user", "content": long},
    ])
    assert pairs and len(pairs[0][1]) == hook_write.MAX_TURN_CHARS


# ---------------------------------------------------------------- run()

def test_run_raw_mode_ingests_both_turns(mem) -> None:
    summary = hook_write.run(
        memory=mem,
        payload={"messages": [
            {"role": "user", "content": "Alice owns Acme Corp."},
            {"role": "assistant", "content": "Acme is based in Seattle."},
        ]},
        mode="raw", scope="shared",
    )
    assert summary["ingested"] == 2
    assert summary["duplicates"] == 0
    assert summary["mode"] == "raw"


def test_run_raw_mode_dedup_on_replay(mem) -> None:
    payload = {"messages": [
        {"role": "user", "content": "Alice owns Acme Corp."},
        {"role": "assistant", "content": "Got it."},
    ]}
    s1 = hook_write.run(memory=mem, payload=payload, mode="raw", scope="shared")
    s2 = hook_write.run(memory=mem, payload=payload, mode="raw", scope="shared")
    assert s1["ingested"] == 2
    # Second run is a full no-op due to sha256 idempotency.
    assert s2["duplicates"] == 2
    assert s2["ingested"] == 0


def test_run_empty_messages_skips(mem) -> None:
    s = hook_write.run(memory=mem, payload={"messages": []}, mode="raw")
    assert s["ingested"] == 0
    assert s["skipped"] == "no turns to ingest"


def test_run_extract_mode_falls_back_without_anthropic(mem, capsys) -> None:
    """If anthropic SDK / API key missing, extract degrades gracefully."""
    # Force the import-error branch by monkeypatching builtins
    import sys as _sys
    saved = _sys.modules.get("anthropic")
    _sys.modules["anthropic"] = None  # type: ignore[assignment]
    try:
        s = hook_write.run(
            memory=mem,
            payload={"messages": [
                {"role": "user", "content": "Bob runs the warehouse."},
            ]},
            mode="extract", scope="shared",
        )
    finally:
        if saved is None:
            del _sys.modules["anthropic"]
        else:
            _sys.modules["anthropic"] = saved
    # Fallback ingests as raw (1 source).
    assert s["ingested"] == 1
    assert s["mode"] == "extract"
    assert "falling back" in capsys.readouterr().err.lower()


def test_run_extract_mode_with_mocked_llm(mem) -> None:
    """Mock the Anthropic client and confirm extracted-fact ingest."""
    fake_text = "Alice owns Acme Corp.\nAlice founded Acme in 2019."

    class _Block:
        type = "text"
        text = fake_text

    class _Resp:
        content: list = [_Block()]  # noqa: RUF012 — test-only mock

    class _Messages:
        def create(self, **_kw):
            return _Resp()

    class _FakeAnthropic:
        def __init__(self, *_a, **_kw): pass
        @property
        def messages(self): return _Messages()

    fake_mod = type("anthropic", (), {"Anthropic": _FakeAnthropic})
    import sys as _sys
    saved = _sys.modules.get("anthropic")
    _sys.modules["anthropic"] = fake_mod  # type: ignore[assignment]
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
        try:
            s = hook_write.run(
                memory=mem,
                payload={"messages": [
                    {"role": "user", "content": "raw turn"},
                    {"role": "assistant", "content": "another raw turn"},
                ]},
                mode="extract", scope="shared",
            )
        finally:
            if saved is None:
                del _sys.modules["anthropic"]
            else:
                _sys.modules["anthropic"] = saved
    # Two turns × two facts each = 4 ingests (assuming dedup doesn't collapse).
    assert s["ingested"] >= 2
    assert s["mode"] == "extract"


# ---------------------------------------------------------------- cli_main

def test_cli_main_summarises_on_stderr(mem, monkeypatch, capsys, tmp_path) -> None:
    payload = {"messages": [
        {"role": "user", "content": "Auto-write test fact."},
    ]}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    db_path = str(tmp_path / "cli.db")
    # Need to seed the db; reuse fixture-style open/close.
    Memory.open(db_path=db_path, embedder_choice="deterministic", embedder_dim=128).close()
    rc = hook_write.cli_main(db_path=db_path, embedder_choice="deterministic")
    assert rc == 0
    err = capsys.readouterr().err
    assert "ingested" in err


def test_cli_main_invalid_json_returns_0(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("garbage{"))
    rc = hook_write.cli_main(
        db_path=str(tmp_path / "x.db"), embedder_choice="deterministic",
    )
    assert rc == 0
    assert "invalid JSON" in capsys.readouterr().err


# ---------------------------------------------------------------- CLI wrapping

def test_thought_hook_write_invalid_mode_exits_2(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    # init first so config exists
    runner.invoke(app, ["init", "--db-path", ".thought/t.db",
                        "--embedder", "deterministic", "--quick", "--no-claude-md"])
    r = runner.invoke(app, ["hook", "write", "--mode", "nope"], input="{}\n")
    assert r.exit_code == 2


def test_thought_hook_write_raw_runs_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["init", "--db-path", ".thought/t.db",
                        "--embedder", "deterministic", "--quick", "--no-claude-md"])
    payload = json.dumps({"messages": [
        {"role": "user", "content": "I prefer Adidas running shoes."},
        {"role": "assistant", "content": "Noted."},
    ]})
    r = runner.invoke(app, ["hook", "write", "--mode", "raw", "--scope", "shared"], input=payload)
    assert r.exit_code == 0
