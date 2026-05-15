"""MCP server exposing the two tools ``remember`` and ``recall``.

Uses the Anthropic MCP Python SDK (``mcp`` package) with the FastMCP convenience
wrapper. Bound to Streamable HTTP per the Nov-2025 MCP spec.

The server is a thin shim — all logic lives in :class:`thought.memory.Memory`.

Tool handlers are ``async`` and offload the sync ``Memory`` work to a thread
pool via ``asyncio.to_thread``. This lets the Streamable HTTP transport
service concurrent recalls without serialising them on the event loop.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Literal

from .memory import Memory


def build_app(memory: Memory):
    """Construct a FastMCP application that delegates to ``memory``.

    Raises ``ImportError`` with a friendly message if the ``mcp`` package is
    not installed — the server is an optional extra (``pip install thought-mcp[mcp]``).
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "MCP transport not installed. Run: pip install 'thought-mcp[mcp]' "
            "(or 'thought-mcp[all]')."
        ) from e

    app = FastMCP("thought")

    @app.tool()
    async def remember(
        content: str,
        source_ref: str | None = None,
        scope: Literal["shared", "private"] = "private",
        owner_id: str | None = None,
    ) -> dict:
        """Persist ``content`` to long-term memory.

        Extracts entities and relationships, embeds them, links provenance to
        the raw source, and runs write-time contradiction detection on the
        configured unique predicates. Idempotent on (content sha256).
        """
        def _do() -> dict:
            r = memory.remember(
                content=content, source_ref=source_ref, scope=scope,
                owner_id=owner_id,
            )
            return r.model_dump(mode="json")
        return await asyncio.to_thread(_do)

    @app.tool()
    async def recall(
        query: str,
        limit: int = 10,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
        as_of: str | None = None,
        as_of_kind: Literal["valid", "learned"] = "valid",
    ) -> dict:
        """Retrieve up to ``limit`` (≤10) hits relevant to ``query``.

        Internally classifies the query (VIBE / FACT / CHANGE / HYBRID) and
        dispatches to the appropriate layer(s). Returns hits annotated with
        their layer of origin, an epistemic confidence class, and source
        provenance. ``as_of`` can be ISO-8601; ``as_of_kind="valid"`` filters
        by world-time, ``"learned"`` by transaction-time.
        """
        as_of_dt: datetime | None = (
            datetime.fromisoformat(as_of) if as_of else None
        )
        def _do() -> dict:
            r = memory.recall(
                query=query, limit=limit, scope=scope, owner_id=owner_id,
                as_of=as_of_dt, as_of_kind=as_of_kind,
            )
            return r.model_dump(mode="json")
        return await asyncio.to_thread(_do)

    @app.tool()
    async def list_topics(
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
        min_count: int = 1,
    ) -> dict:
        """List entity-type buckets currently in the KB.

        Returns ``{"topics": [{"type": "...", "count": N, "examples": [...]}]}``
        ordered by population descending. Cheap aggregation — single SQL
        GROUP BY. Use this to discover what *kinds* of facts the memory holds
        before drilling down with ``browse_topic``.
        """
        def _do() -> dict:
            return {"topics": memory.list_topics(
                scope=scope, owner_id=owner_id, min_count=min_count,
            )}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def schema() -> dict:
        """Show entity types and relation types currently in the KB.

        Use this before composing Cypher queries via ``query`` or ``ask`` so
        you know what's available. Returns
        ``{"entity_types": {type: count}, "relation_types": {type: count}}``.
        """
        def _do() -> dict:
            return memory.schema_summary()
        return await asyncio.to_thread(_do)

    @app.tool()
    async def query(
        cypher: str,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
    ) -> dict:
        """Run a Cypher query against the KB (read-only subset).

        Returns ``{"rows": [{col: value, ...}], "columns": [...]}``.
        Cypher errors are returned as ``{"error": ..., "kind": "syntax|unsupported|other"}``.
        """
        from .query import cypher as _cypher

        def _do() -> dict:
            try:
                rows = _cypher.execute(memory, cypher, scope=scope, owner_id=owner_id)
            except _cypher.UnsupportedCypher as e:
                return {"error": str(e), "kind": "unsupported"}
            except _cypher.CypherSyntaxError as e:
                return {"error": str(e), "kind": "syntax"}
            except Exception as e:
                return {"error": str(e), "kind": "other"}
            return {"rows": rows}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def view_save(name: str, cypher: str) -> dict:
        """Save a Cypher query as a named view (re-evaluates on each call)."""
        from .query import views

        def _do() -> dict:
            try:
                return views.save_view(memory, name, cypher, replace=True)
            except Exception as e:
                return {"error": str(e)}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def view_list() -> dict:
        """List saved views with their stored Cypher and last-run stats."""
        from .query import views

        def _do() -> dict:
            return {"views": views.list_views(memory)}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def view_run(
        name: str,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
    ) -> dict:
        """Run a saved view; rows are pull-evaluated against the live KB."""
        from .query import views

        def _do() -> dict:
            try:
                return {"rows": views.run_view(memory, name, scope=scope, owner_id=owner_id)}
            except Exception as e:
                return {"error": str(e)}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def working_context(
        target: str,
        role: str | None = None,
        budget_tokens: int = 2000,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
    ) -> dict:
        """Universal *"what does my agent need to know about X right now"* primitive.

        ``target`` can be ``"function:authenticate"`` / ``"chapter:5"`` / a plain entity name.
        Returns ``{"anchor", "neighbours", "recent_contradictions", "role_view"}``
        token-budgeted to ``budget_tokens``.
        """
        def _do() -> dict:
            return memory.working_context(
                target, role=role, budget_tokens=budget_tokens,
                scope=scope, owner_id=owner_id,
            )
        return await asyncio.to_thread(_do)

    @app.tool()
    async def scan(
        repo_path: str,
        agent: str | None = None,
        since: str | None = None,
        max_files: int | None = None,
        note: str | None = None,
    ) -> dict:
        """Incremental code-scan primitive. Walks ``repo_path``, ingests
        changed/new files, records a row in ``scan_log`` so the next call
        picks up where this one left off.
        """
        def _do() -> dict:
            return memory.scan(
                repo_path, agent=agent, since=since,
                max_files=max_files, note=note,
            )
        return await asyncio.to_thread(_do)

    @app.tool()
    async def scan_log_list(
        agent: str | None = None, limit: int = 10,
    ) -> dict:
        """List the last N scan runs (optionally filtered by agent name)."""
        def _do() -> dict:
            return {"scans": memory.scan_log(agent=agent, limit=limit)}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def register_agent(
        name: str,
        description: str | None = None,
        capabilities: list[str] | None = None,
    ) -> dict:
        """First-time setup for a named agent (vuln-scanner / writer / etc.)."""
        def _do() -> dict:
            return memory.register_agent(
                name, description=description, capabilities=capabilities,
            )
        return await asyncio.to_thread(_do)

    @app.tool()
    async def view_delete(name: str) -> dict:
        """Delete a saved view."""
        from .query import views

        def _do() -> dict:
            return {"deleted": views.delete_view(memory, name)}
        return await asyncio.to_thread(_do)

    @app.tool()
    async def browse_topic(
        name: str,
        depth: int = 1,
        limit: int = 20,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
    ) -> dict:
        """Drill into a topic by name.

        ``name`` is matched first against entity-type names (``PERSON``,
        ``CONCEPT``, ``function``, …) for a type facet; if no type matches,
        it's resolved as an entity name and the PPR-ranked neighbourhood is
        returned. Returns ``{"items": [{"id", "name", "type", "score", "via"}]}``
        where ``via`` is one of ``type_facet`` / ``ppr`` / ``bfs``.
        """
        def _do() -> dict:
            return {"items": memory.browse_topic(
                name, depth=depth, limit=limit,
                scope=scope, owner_id=owner_id,
            )}
        return await asyncio.to_thread(_do)

    return app
