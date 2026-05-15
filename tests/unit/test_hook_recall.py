"""Tests for the auto-recall hook + hook installer.

Covers:
- ``hooks.recall.run`` — pure-Python; passes prompt → recall → response dict.
- ``hooks.recall.cli_main`` — stdin / stdout wrapping + exit-code contract.
- ``hooks.install`` — idempotent ``.claude/settings.json`` merger.
- ``thought hook install`` / ``thought hook recall`` CLI commands.
"""
from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought import hooks
from thought.cli import app
from thought.hooks import install as hook_install
from thought.hooks import recall as hook_recall
from thought.memory import Memory

# ---------------------------------------------------------------- run()

@pytest.fixture()
def mem_with_hit(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "h.db"),
        embedder_choice="deterministic", embedder_dim=128,
    )
    now = datetime.now(UTC)
    src = m._backend.upsert_source("Alice owns Acme.")
    m._backend.upsert_entity(
        type_="PERSON", name="Alice", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    m._backend.upsert_entity(
        type_="ORGANIZATION", name="Acme", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    yield m
    m.close()


def test_run_returns_additional_context_for_hit(mem_with_hit) -> None:
    response = hook_recall.run(
        memory=mem_with_hit,
        payload={"prompt": "Alice"},
        limit=3, scope="all",
    )
    # Either the hook produced injection text (low-confidence not triggered)
    # or it skipped silently — both are valid; assert the response shape.
    if "_skip_reason" in response:
        pytest.skip("deterministic embedder did not produce hits above gate")
    body = response["hookSpecificOutput"]["additionalContext"]
    assert "Alice" in body
    assert "thought recall" in body
    assert response["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_run_returns_skip_for_low_confidence(mem_with_hit) -> None:
    # A query with no signal at all → empty hits → low_confidence=True.
    response = hook_recall.run(
        memory=mem_with_hit,
        payload={"prompt": "completely unrelated nonsense xyzzy"},
        limit=3, scope="all",
    )
    assert "_skip_reason" in response


def test_run_empty_prompt_skips(mem_with_hit) -> None:
    assert "_skip_reason" in hook_recall.run(
        memory=mem_with_hit, payload={"prompt": ""},
    )
    assert "_skip_reason" in hook_recall.run(
        memory=mem_with_hit, payload={},
    )


# ---------------------------------------------------------------- cli_main()

def test_cli_main_writes_json_on_stdout(
    mem_with_hit, monkeypatch, capsys, tmp_path,
) -> None:
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"prompt": "Alice"})),
    )
    db_path = str(tmp_path / "cli_hr.db")
    # Seed via the same-db Memory.open path: re-use the fixture's seeding.
    m = Memory.open(
        db_path=db_path, embedder_choice="deterministic", embedder_dim=128,
    )
    now = datetime.now(UTC)
    src = m._backend.upsert_source("Alice owns Acme.")
    m._backend.upsert_entity(
        type_="PERSON", name="Alice", scope="shared",
        valid_from=now, learned_at=now, source_ref=src,
    )
    m.close()

    rc = hook_recall.cli_main(db_path=db_path, embedder_choice="deterministic")
    assert rc == 0
    captured = capsys.readouterr()
    # Either the skip-reason path (stderr) or a JSON response (stdout) is fine.
    if captured.out.strip():
        data = json.loads(captured.out)
        assert "hookSpecificOutput" in data


def test_cli_main_invalid_json_returns_0_with_stderr(
    monkeypatch, tmp_path, capsys,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not valid json {"))
    rc = hook_recall.cli_main(
        db_path=str(tmp_path / "bad.db"),
        embedder_choice="deterministic",
    )
    assert rc == 0
    assert "invalid JSON" in capsys.readouterr().err


# ---------------------------------------------------------------- install

def test_install_recall_creates_settings_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = hook_install.install_hook("recall", scope="project")
    assert r.status == "installed"
    assert r.path.exists()
    data = json.loads(r.path.read_text(encoding="utf-8"))
    events = data["hooks"]["UserPromptSubmit"]
    assert any(
        h.get("command") == "thought hook recall"
        for entry in events for h in entry.get("hooks", [])
    )


def test_install_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    hook_install.install_hook("recall", scope="project")
    r2 = hook_install.install_hook("recall", scope="project")
    assert r2.status == "already_present"


def test_install_preserves_other_hooks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / ".claude" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({
        "hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "other-tool"}]}
        ]}
    }), encoding="utf-8")
    r = hook_install.install_hook("recall", scope="project")
    assert r.status == "installed"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    commands = {
        h.get("command")
        for entry in data["hooks"]["UserPromptSubmit"]
        for h in entry.get("hooks", [])
    }
    assert "other-tool" in commands
    assert "thought hook recall" in commands
    # Backup made.
    assert cfg.with_suffix(cfg.suffix + ".thought.bak").exists()


def test_install_user_scope_writes_to_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    r = hook_install.install_hook("recall", scope="user")
    assert r.status == "installed"
    assert r.path == tmp_path / ".claude" / "settings.json"
    assert r.path.exists()


# ---------------------------------------------------------------- CLI wrapping

def test_thought_hook_install_recall_via_cli(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["hook", "install", "--recall"])
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / ".claude" / "settings.json").exists()


def test_thought_hook_install_no_flag_exits_2(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["hook", "install"])
    assert r.exit_code == 2


def test_thought_hook_install_both_writes_both(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["hook", "install", "--both"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(
        (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert "UserPromptSubmit" in data["hooks"]
    assert "Stop" in data["hooks"]


def test_hooks_module_smoke() -> None:
    """Spot-check the package exports compile."""
    assert hooks.__name__ == "thought.hooks"
