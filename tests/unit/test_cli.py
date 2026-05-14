"""End-to-end CLI tests via :class:`typer.testing.CliRunner`.

Covers every ``thought`` subcommand. Each test runs against an isolated
tmp_path used as both ``$HOME`` (so the MCP-client installer probes don't
touch the user's real configs) and as the workspace cwd (so ``thought init``
writes ``thought.toml`` and ``.thought/thought.db`` into the temp dir).

We use the deterministic embedder so every test is offline-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought import __version__
from thought.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Isolate every CLI invocation in a clean tmp dir.

    - ``Path.home()`` → ``tmp_path`` (MCP-client installer config writes
      land in the sandbox, not the user's real home).
    - ``$APPDATA`` (Windows-only) → ``tmp_path/AppData/Roaming`` for the
      Cline path detector.
    - cwd → ``tmp_path`` so ``thought.toml`` and ``.thought/`` are local.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _init(runner: CliRunner, workspace: Path) -> None:
    """Most commands need an initialised KB; run ``init`` first."""
    result = runner.invoke(
        app,
        ["init", "--db-path", str(workspace / ".thought" / "thought.db"),
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")


# ---------------------------------------------------------------- root

def test_version_flag_prints_version(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.stdout


def test_help_renders_without_subcommand(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, [])
    # Typer returns 0 when invoked without a command if we route to help.
    assert r.exit_code == 0
    assert "thought" in r.stdout.lower()


# ---------------------------------------------------------------- init

def test_init_creates_config_db_and_claude_md(
    runner: CliRunner, workspace: Path,
) -> None:
    r = runner.invoke(
        app,
        ["init", "--db-path", ".thought/t.db",
         "--embedder", "deterministic", "--quick"],
    )
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    assert (workspace / "thought.toml").exists()
    assert (workspace / ".thought" / "t.db").exists()
    assert (workspace / "CLAUDE.md").exists()


def test_init_with_no_claude_md_skips_template(
    runner: CliRunner, workspace: Path,
) -> None:
    r = runner.invoke(
        app,
        ["init", "--db-path", ".thought/t.db",
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0
    assert not (workspace / "CLAUDE.md").exists()


# ---------------------------------------------------------------- install / detect

def test_install_detect_renders_table(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, ["install", "--detect"])
    assert r.exit_code == 0
    out = r.stdout
    # All five client names should appear in the detect table.
    for client in ("claude-code", "cursor", "cline", "continue", "windsurf"):
        assert client in out


def test_install_cursor_writes_config(
    runner: CliRunner, workspace: Path,
) -> None:
    r = runner.invoke(app, ["install", "--client", "cursor"])
    assert r.exit_code == 0
    cfg = workspace / ".cursor" / "mcp.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert "thought" in data["mcpServers"]
    assert data["mcpServers"]["thought"]["command"] == "uvx"


def test_install_unknown_client_exits_2(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, ["install", "--client", "notreal"])
    assert r.exit_code == 2
    assert "notreal" in (r.stderr or r.stdout)


def test_install_no_args_exits_2(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, ["install"])
    assert r.exit_code == 2
    assert "specify" in (r.stderr or r.stdout).lower()


def test_install_all_attempts_every_client(
    runner: CliRunner, workspace,
) -> None:
    r = runner.invoke(app, ["install", "--all"])
    assert r.exit_code == 0
    out = r.stdout
    for client in ("claude-code", "cursor", "cline", "continue", "windsurf"):
        assert client in out


# ---------------------------------------------------------------- upgrade

def test_upgrade_cursor_pins_specific_version(
    runner: CliRunner, workspace: Path,
) -> None:
    # First install the unpinned form, then upgrade to a specific version.
    runner.invoke(app, ["install", "--client", "cursor"])
    r = runner.invoke(
        app, ["upgrade", "--client", "cursor", "--version", "0.2.2"],
    )
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    cfg = json.loads(
        (workspace / ".cursor" / "mcp.json").read_text(encoding="utf-8")
    )
    args = cfg["mcpServers"]["thought"]["args"]
    assert any("thought-mcp[mcp,sqlite-vec]==0.2.2" in a for a in args)


def test_upgrade_all_handles_missing_clients(
    runner: CliRunner, workspace,
) -> None:
    # Most clients won't exist in this tmp home; upgrade should not crash,
    # just report no_path / installed per client.
    r = runner.invoke(
        app, ["upgrade", "--all", "--version", "0.2.2"],
    )
    assert r.exit_code == 0


def test_upgrade_unknown_client_exits_2(runner: CliRunner, workspace) -> None:
    r = runner.invoke(app, ["upgrade", "--client", "notreal"])
    assert r.exit_code == 2


# ---------------------------------------------------------------- ingest

def test_ingest_text_returns_json(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    r = runner.invoke(app, ["ingest", "Alice owns Acme."])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    data = json.loads(r.stdout)
    assert data["source_id"]


def test_ingest_file_returns_json(
    runner: CliRunner, workspace: Path,
) -> None:
    _init(runner, workspace)
    f = workspace / "note.txt"
    f.write_text("Bob runs the warehouse.", encoding="utf-8")
    r = runner.invoke(app, ["ingest", "--file", str(f)])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    data = json.loads(r.stdout)
    assert data["source_id"]


def test_ingest_glob_bulk_path(
    runner: CliRunner, workspace: Path,
) -> None:
    _init(runner, workspace)
    for i in range(3):
        (workspace / f"note{i}.txt").write_text(
            f"Note {i}: Alice owns project {i}.", encoding="utf-8",
        )
    r = runner.invoke(app, ["ingest", "--glob", "note*.txt"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    assert "items processed" in r.stdout or "items processed" in (r.stderr or "")


def test_ingest_stdin_bulk_path(
    runner: CliRunner, workspace,
) -> None:
    _init(runner, workspace)
    r = runner.invoke(
        app, ["ingest", "--stdin"],
        input="Alice owns Acme.\nBob owns Beta.\n",
    )
    assert r.exit_code == 0, r.stdout + (r.stderr or "")


def test_ingest_no_input_exits_1(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    r = runner.invoke(app, ["ingest"])
    assert r.exit_code == 1
    assert "must provide" in (r.stderr or r.stdout).lower()


# ---------------------------------------------------------------- recall

def test_recall_pretty_table(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme."])
    r = runner.invoke(app, ["recall", "alice"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")


def test_recall_json_output(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme."])
    r = runner.invoke(app, ["recall", "alice", "--json"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    data = json.loads(r.stdout)
    assert "hits" in data
    assert data["query_class"] in {"VIBE", "FACT", "CHANGE", "HYBRID", "CODE"}


# ---------------------------------------------------------------- stats / forget / consolidate

def test_stats_renders_summary(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme."])
    r = runner.invoke(app, ["stats"])
    assert r.exit_code == 0
    assert "entities" in r.stdout.lower()


def test_forget_with_yes_flag_runs(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme."])
    r = runner.invoke(app, ["forget", "alice%", "--yes"])
    assert r.exit_code == 0
    assert "retired" in r.stdout.lower()


def test_consolidate_prints_audit_count(runner: CliRunner, workspace) -> None:
    _init(runner, workspace)
    r = runner.invoke(app, ["consolidate"])
    assert r.exit_code == 0
    assert "consolidation" in r.stdout.lower()


# ---------------------------------------------------------------- doctor

def test_doctor_renders_environment_table(
    runner: CliRunner, workspace,
) -> None:
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0
    out = r.stdout
    assert __version__ in out
    assert "python" in out.lower()


# ---------------------------------------------------------------- serve transport validation

def test_serve_rejects_unknown_transport(runner: CliRunner, workspace) -> None:
    """The transport flag is the headline v0.2.2 fix; validate the guard."""
    _init(runner, workspace)
    r = runner.invoke(
        app, ["serve", "--transport", "websocket", "--skip-precheck"],
    )
    assert r.exit_code == 2
    assert "transport" in (r.stderr or r.stdout).lower()


# ---------------------------------------------------------------- code vertical

def test_ingest_code_walks_fixture_dir(
    runner: CliRunner, workspace: Path,
) -> None:
    _init(runner, workspace)
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "code" / "python"
    r = runner.invoke(
        app, ["ingest-code", str(fixture), "--glob", "*.py"],
    )
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    assert "entities" in r.stdout.lower() or "files processed" in r.stdout.lower()


def test_callers_returns_results_after_code_ingest(
    runner: CliRunner, workspace: Path,
) -> None:
    _init(runner, workspace)
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "code" / "python"
    runner.invoke(app, ["ingest-code", str(fixture), "--glob", "*.py"])
    # ``_decode_token`` is called by ``authenticate_user`` in auth.py.
    r = runner.invoke(app, ["callers", "_decode_token"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")


def test_impact_returns_results_after_code_ingest(
    runner: CliRunner, workspace: Path,
) -> None:
    _init(runner, workspace)
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "code" / "python"
    runner.invoke(app, ["ingest-code", str(fixture), "--glob", "*.py"])
    r = runner.invoke(app, ["impact", "_decode_token"])
    assert r.exit_code == 0, r.stdout + (r.stderr or "")


# ---------------------------------------------------------------- repl (smoke)

def test_repl_exits_on_empty_input(runner: CliRunner, workspace) -> None:
    """REPL is interactive; just verify it shuts down cleanly on empty
    stdin / quit so the function isn't outright broken.
    """
    _init(runner, workspace)
    r = runner.invoke(app, ["repl"], input="q\n")
    assert r.exit_code == 0
