"""Saved-views CRUD over the ``saved_views`` table (migration 0003).

A saved view is a named Cypher query persisted in the KB. Calling ``run_view``
re-evaluates it against the live data — pull-evaluated, not snapshot.
"""
from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from . import cypher

# Per-view-name allowlist: SQL-injection-safe even though we use parameters.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ViewError(Exception):
    pass


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ViewError(
            f"invalid view name {name!r} — must match [A-Za-z_][A-Za-z0-9_]{{0,63}}"
        )


def save_view(memory, name: str, source: str, *, replace: bool = True) -> dict[str, Any]:
    """Persist a named Cypher query. Parses the source first so we never store junk."""
    validate_name(name)
    # Validate by parsing (raises CypherSyntaxError / UnsupportedCypher if bad).
    cypher.parse(source)
    now = datetime.now(UTC).isoformat()
    if replace:
        memory._backend._conn.execute(
            "INSERT OR REPLACE INTO saved_views "
            "(name, cypher_source, created_at, scope) VALUES (?, ?, ?, 'all')",
            (name, source, now),
        )
    else:
        try:
            memory._backend._conn.execute(
                "INSERT INTO saved_views "
                "(name, cypher_source, created_at, scope) VALUES (?, ?, ?, 'all')",
                (name, source, now),
            )
        except Exception as e:
            raise ViewError(f"view {name!r} already exists; pass replace=True") from e
    memory._backend._touch_write()  # type: ignore[attr-defined]
    return {"name": name, "saved": True}


def list_views(memory) -> list[dict[str, Any]]:
    rows = memory._backend._conn.execute(
        "SELECT name, cypher_source, created_at, last_run_at, last_run_ms, last_run_count "
        "FROM saved_views ORDER BY name"
    ).fetchall()
    return [
        {
            "name": r["name"],
            "cypher": r["cypher_source"],
            "created_at": r["created_at"],
            "last_run_at": r["last_run_at"],
            "last_run_ms": r["last_run_ms"],
            "last_run_count": r["last_run_count"],
        }
        for r in rows
    ]


def show_view(memory, name: str) -> dict[str, Any] | None:
    validate_name(name)
    row = memory._backend._conn.execute(
        "SELECT name, cypher_source, created_at, last_run_at, last_run_ms, last_run_count "
        "FROM saved_views WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return {
        "name": row["name"],
        "cypher": row["cypher_source"],
        "created_at": row["created_at"],
        "last_run_at": row["last_run_at"],
        "last_run_ms": row["last_run_ms"],
        "last_run_count": row["last_run_count"],
    }


def delete_view(memory, name: str) -> bool:
    validate_name(name)
    rc = memory._backend._conn.execute(
        "DELETE FROM saved_views WHERE name = ?", (name,)
    ).rowcount
    memory._backend._touch_write()  # type: ignore[attr-defined]
    return bool(rc)


def run_view(
    memory, name: str, *,
    scope: str = "all", owner_id: str | None = None,
) -> list[dict[str, Any]]:
    validate_name(name)
    row = memory._backend._conn.execute(
        "SELECT cypher_source FROM saved_views WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise ViewError(f"no saved view named {name!r}")
    src = row["cypher_source"]
    t0 = time.perf_counter()
    results = cypher.execute(memory, src, scope=scope, owner_id=owner_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    memory._backend._conn.execute(
        "UPDATE saved_views SET last_run_at = ?, last_run_ms = ?, last_run_count = ? "
        "WHERE name = ?",
        (datetime.now(UTC).isoformat(), elapsed_ms, len(results), name),
    )
    return results
