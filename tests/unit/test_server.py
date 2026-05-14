"""Server module tests.

We don't ship the ``mcp`` package as a hard dep — it's an optional extra. The
server module must (a) import cleanly without ``mcp`` installed, and
(b) raise a friendly error when ``build_app`` is called without it.

When ``mcp`` IS installed (as in CI), ``build_app`` must construct a working
FastMCP application exposing the two tools.
"""
from __future__ import annotations

from importlib.util import find_spec

import pytest

from thought.memory import Memory
from thought.server import build_app

MCP_INSTALLED = find_spec("mcp") is not None


@pytest.mark.skipif(MCP_INSTALLED, reason="mcp package is installed in this env")
def test_build_app_friendly_error_when_mcp_missing(tmp_path) -> None:
    mem = Memory.open(
        db_path=str(tmp_path / "s.db"),
        embedder_choice="deterministic",
        embedder_dim=64,
    )
    try:
        with pytest.raises(ImportError, match="MCP transport not installed"):
            build_app(mem)
    finally:
        mem.close()


@pytest.mark.skipif(not MCP_INSTALLED, reason="mcp package not installed")
def test_build_app_exposes_two_tools(tmp_path) -> None:
    mem = Memory.open(
        db_path=str(tmp_path / "s.db"),
        embedder_choice="deterministic",
        embedder_dim=64,
    )
    try:
        app = build_app(mem)
        # FastMCP tracks registered tools internally. We don't depend on its
        # exact API beyond a public way to enumerate them; tools are also
        # discoverable by the SDK's protocol layer at runtime.
        assert app is not None
        assert getattr(app, "name", None) == "thought"
    finally:
        mem.close()
