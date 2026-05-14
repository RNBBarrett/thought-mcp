"""Tests for the version-pinning helper used by ``thought upgrade``.

The CLI command itself is exercised end-to-end via smoke tests; here we
cover the underlying ``pin_server_block`` / ``upgrade_clients`` functions
that produce + write the new ``mcpServers`` entry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from thought import clients


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    return tmp_path


def test_pin_server_block_produces_uvx_from_form_with_extras() -> None:
    block = clients.pin_server_block(version="0.2.0")
    # ``[mcp,sqlite-vec]`` extras are required: the server crashes at
    # startup without ``mcp``, and ANN is unusably slow without ``sqlite-vec``.
    assert block == {
        "command": "uvx",
        "args": [
            "--from", "thought-mcp[mcp,sqlite-vec]==0.2.0",
            "thought", "serve",
        ],
    }


def test_pin_server_block_respects_custom_extras() -> None:
    block = clients.pin_server_block(version="0.3.0", extras=("code",))
    assert block["args"][1] == "thought-mcp[code]==0.3.0"


def test_pin_server_block_supports_empty_extras() -> None:
    block = clients.pin_server_block(version="0.2.0", extras=())
    assert block["args"][1] == "thought-mcp==0.2.0"


def test_pin_server_block_omitting_version_pins_to_latest() -> None:
    # ``version=None`` means "pin to whatever is the running CLI version",
    # which we read from thought.__version__. The resulting args should
    # contain the version string.
    block = clients.pin_server_block(version=None)
    assert block["command"] == "uvx"
    assert any("==" in a for a in block["args"])


def test_upgrade_writes_pinned_entry_and_backs_up(home) -> None:
    # Pre-install with the unpinned form first.
    clients.install("cursor")
    path = clients._cursor_path()
    assert path is not None
    before = json.loads(path.read_text(encoding="utf-8"))
    assert before["mcpServers"]["thought"]["args"] == ["thought-mcp", "serve"]

    # Now upgrade-pin to 0.2.0.
    r = clients.upgrade("cursor", version="0.2.0")
    assert r.status == "installed"
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["mcpServers"]["thought"]["args"] == [
        "--from", "thought-mcp[mcp,sqlite-vec]==0.2.0", "thought", "serve"
    ]
    # Backup of the prior (unpinned) state exists.
    assert path.with_suffix(path.suffix + ".thought.bak").exists()


def test_upgrade_idempotent_when_block_matches(home) -> None:
    clients.upgrade("cursor", version="0.2.0")
    r = clients.upgrade("cursor", version="0.2.0")
    assert r.status == "already_present"


def test_upgrade_returns_no_path_when_client_unknown_to_user(home, monkeypatch) -> None:
    # Force the resolver registry entry to return None — simulates "client
    # not installed on this machine."
    monkeypatch.setitem(clients._PATH_FNS, "cursor", lambda: None)
    r = clients.upgrade("cursor", version="0.2.0")
    assert r.status == "no_path"
