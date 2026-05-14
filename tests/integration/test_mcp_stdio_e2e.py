"""End-to-end MCP test over the stdio transport.

Spawns ``thought serve --transport stdio`` as a subprocess and drives it with
the official ``mcp`` SDK client. This is the test that would have caught the
v0.2.1 ship bug — every MCP-client config wired up by
``thought install`` / ``thought upgrade`` invokes the server via stdio, but the
default transport was ``streamable-http`` so Claude Code timed out at 30s.

Marked ``@pytest.mark.integration`` so the default unit run stays fast.
"""
from __future__ import annotations

import json
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        find_spec("mcp") is None, reason="mcp SDK not installed",
    ),
    pytest.mark.skipif(
        shutil.which("thought") is None,
        reason="thought CLI not on PATH (pip install -e . first)",
    ),
]


@pytest.fixture()
async def initialized_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Create a fresh thought.toml + .thought/thought.db in a tmp dir."""
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
    return tmp_path


@pytest.mark.asyncio
async def test_stdio_remember_then_recall_roundtrip(initialized_workspace):
    """Spawn the server, list tools, exercise both, shut down cleanly."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    cwd = initialized_workspace
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "thought.cli", "serve",
              "--transport", "stdio", "--skip-precheck"],
        cwd=str(cwd),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert "remember" in tool_names, tool_names
        assert "recall" in tool_names, tool_names

        r = await session.call_tool(
            "remember", {"content": "Alice owns Acme Corp."},
        )
        assert not r.isError, r
        payload = json.loads(r.content[0].text)  # type: ignore[union-attr]
        assert payload["source_id"], payload
        assert payload["entity_ids"], payload

        r = await session.call_tool(
            "recall", {"query": "alice", "limit": 5},
        )
        assert not r.isError, r
        payload = json.loads(r.content[0].text)  # type: ignore[union-attr]
        assert payload["query_class"] in {
            "VIBE", "FACT", "CHANGE", "HYBRID", "CODE",
        }
        assert "elapsed_ms" in payload


@pytest.mark.asyncio
async def test_stdio_dedup_via_tool_call(initialized_workspace):
    """Calling ``remember`` twice with the same content over stdio dedups."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "thought.cli", "serve",
              "--transport", "stdio", "--skip-precheck"],
        cwd=str(initialized_workspace),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        r1 = await session.call_tool(
            "remember", {"content": "Identical content."},
        )
        r2 = await session.call_tool(
            "remember", {"content": "Identical content."},
        )
        p1 = json.loads(r1.content[0].text)  # type: ignore[union-attr]
        p2 = json.loads(r2.content[0].text)  # type: ignore[union-attr]
        assert p2["duplicate_of_source"] == p1["source_id"]
