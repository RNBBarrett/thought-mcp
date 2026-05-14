"""Tests for the MCP-client config installer.

Each test monkeypatches ``Path.home`` to a tmp dir so we don't touch the
user's real config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from thought import clients


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Also fake the Windows APPDATA path so Cline detection has a home.
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    return tmp_path


def test_install_into_fresh_cursor_config(home) -> None:
    r = clients.install("cursor")
    assert r.status == "installed"
    assert r.path is not None
    data = json.loads(r.path.read_text(encoding="utf-8"))
    assert "thought" in data["mcpServers"]
    assert data["mcpServers"]["thought"]["command"] == "uvx"


def test_install_preserves_existing_servers(home) -> None:
    path = clients._cursor_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mcpServers": {"otherTool": {"command": "echo"}}}),
        encoding="utf-8",
    )
    r = clients.install("cursor")
    assert r.status == "installed"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["mcpServers"].keys()) == {"otherTool", "thought"}
    assert path.with_suffix(path.suffix + ".thought.bak").exists()


def test_install_idempotent_when_block_matches(home) -> None:
    clients.install("cursor")
    r2 = clients.install("cursor")
    assert r2.status == "already_present"


def test_install_rejects_invalid_existing_json(home) -> None:
    path = clients._cursor_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    r = clients.install("cursor")
    assert r.status == "error"
    assert "valid JSON" in r.detail


def test_install_many_returns_one_result_per_client(home) -> None:
    results = clients.install_many(clients.ALL_CLIENTS)
    assert len(results) == len(clients.ALL_CLIENTS)
    assert {r.client for r in results} == set(clients.ALL_CLIENTS)
