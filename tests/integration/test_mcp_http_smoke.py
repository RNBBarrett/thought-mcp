"""Smoke test for the ``streamable-http`` transport.

The full MCP roundtrip via a streamable-http client lives in the SDK's
own integration suite; here we just confirm that the CLI honors
``--transport streamable-http`` by binding a TCP listener on the requested
port. Combined with :mod:`tests.unit.test_server_tools` (which exercises
the tool handlers without any transport) this is enough to lock the
v0.2.2 transport contract.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("mcp") is None, reason="mcp SDK not installed",
    ),
]


def _free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then release it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(host: str, port: int, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def test_serve_streamable_http_binds_port(tmp_path: Path, monkeypatch) -> None:
    """``thought serve --transport streamable-http`` binds the requested
    port. v0.2.1 had this hardcoded as the default; v0.2.2 makes it
    explicit, so we re-verify the path still works.
    """
    from typer.testing import CliRunner

    from thought.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["init", "--db-path", ".thought/thought.db",
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0, r.stdout

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "thought.cli", "serve",
         "--transport", "streamable-http",
         "--host", "127.0.0.1", "--port", str(port),
         "--skip-precheck"],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        bound = _wait_for_port("127.0.0.1", port, timeout=15.0)
        assert bound, (
            "server never started listening on the requested port"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
