"""Call-graph extraction — Phase 2.

Run after ``CodeIngestPipeline`` has materialised function / method / class
entities for a file. This module walks each function body once more,
finds call expressions, resolves the callee name to an existing entity
ID via the backend's ``find_code_entity``, and writes ``CALLS`` edges.

Resolution strategy (Python, intra-package):

1. Walk the function body looking for ``call`` nodes whose ``function``
   field is either an ``identifier`` or an ``attribute`` (e.g. ``foo()``
   or ``self.foo()`` / ``mod.foo()``).
2. For bare identifiers, search for an entity with that canonical name
   in the same file first, then in any file with the current commit SHA.
3. For ``self.X``, qualify with the enclosing class → search for
   ``ClassName.X`` as a ``method``.
4. For ``mod.X``, treat as cross-package and create a stub function
   entity tagged ``confidence_class="inferred"``.

Unresolved calls don't become edges — we don't want pollution. Stubs
are created only when a target name appears in a context that strongly
suggests a real callable (e.g. ``module.something()``).
"""
from __future__ import annotations

import builtins
from datetime import datetime
from typing import TYPE_CHECKING

from ...models import ScopeFilter, ScopeName
from ...storage.sqlite.backend import SQLiteBackend
from .python_extractor import _get_parser, _text

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

# Python builtins + dunder methods + common Iterator protocol names.
# Calls to these are noise on the impact graph: they're universally
# available, never user-defined in the codebase, and would create stub
# entities that pollute PageRank rankings (the v0.2 dogfood demo showed
# ``len`` / ``float`` / ``sum`` outranking real callers).
_PY_BUILTINS = frozenset(dir(builtins)) | frozenset({
    # Common method names on stdlib types that aren't in ``dir(builtins)``
    # but appear constantly in code and should also be filtered:
    "append", "extend", "items", "keys", "values", "get", "pop",
    "split", "join", "strip", "lstrip", "rstrip", "replace", "format",
    "encode", "decode", "lower", "upper", "startswith", "endswith",
    "read", "write", "readline", "readlines", "close", "open", "seek",
    "execute", "fetchone", "fetchall", "executemany", "commit", "rollback",
    "add", "remove", "discard", "update", "copy", "clear",
    "sort", "reverse", "index", "count", "insert",
    "__init__", "__str__", "__repr__", "__eq__", "__hash__", "__len__",
    "__iter__", "__next__", "__enter__", "__exit__", "__call__",
    "__getitem__", "__setitem__", "__contains__", "__bool__",
    "now", "today", "utcnow", "fromisoformat", "isoformat", "astimezone",
    "Path", "exists", "is_file", "is_dir", "mkdir", "unlink", "rglob",
    # numpy / scipy / common deps
    "array", "zeros", "ones", "dot", "concatenate", "asarray",
})


def _is_builtin(name: str) -> bool:
    """True for names we should NOT create stub entities for."""
    return name in _PY_BUILTINS


def build_call_graph(
    *,
    backend: SQLiteBackend,
    file_path: str,
    source: str,
    language: str,
    commit_sha: str | None,
    scope: ScopeName,
    owner_id: str | None,
    source_ref: str,
    now: datetime,
) -> int:
    """Walk ``source`` and emit CALLS edges. Returns edge count.

    Idempotent on (caller, callee, line) — re-running on the same source
    is a no-op because ``upsert_edge`` dedups on existing identity.

    Phase 1 only supports Python; TypeScript arrives in Phase 5.
    """
    if language != "python":
        return 0

    parser = _get_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    n_edges = 0
    backend.begin()
    try:
        # Walk each top-level function and each class's methods.
        for child in tree.root_node.named_children:
            if child.type in ("function_definition", "async_function_definition"):
                n_edges += _emit_calls_from_function(
                    child, enclosing_class=None,
                    backend=backend, source_bytes=source_bytes,
                    file_path=file_path, commit_sha=commit_sha,
                    scope=scope, owner_id=owner_id,
                    source_ref=source_ref, now=now,
                )
            elif child.type == "class_definition":
                class_name = _name(child, source_bytes)
                body = child.child_by_field_name("body")
                if body is None or class_name is None:
                    continue
                for m in body.named_children:
                    if m.type in ("function_definition", "async_function_definition"):
                        n_edges += _emit_calls_from_function(
                            m, enclosing_class=class_name,
                            backend=backend, source_bytes=source_bytes,
                            file_path=file_path, commit_sha=commit_sha,
                            scope=scope, owner_id=owner_id,
                            source_ref=source_ref, now=now,
                        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise
    return n_edges


def _name(func: Node, source_bytes: bytes) -> str | None:
    name_node = func.child_by_field_name("name")
    return _text(name_node, source_bytes) if name_node else None


def _emit_calls_from_function(
    func: Node,
    *,
    enclosing_class: str | None,
    backend: SQLiteBackend,
    source_bytes: bytes,
    file_path: str,
    commit_sha: str | None,
    scope: ScopeName,
    owner_id: str | None,
    source_ref: str,
    now: datetime,
) -> int:
    """Walk ``func``'s body emitting CALLS edges. Returns edges emitted."""
    fn_name = _name(func, source_bytes)
    if fn_name is None:
        return 0
    caller_name = f"{enclosing_class}.{fn_name}" if enclosing_class else fn_name
    caller_type = "method" if enclosing_class else "function"
    caller_id = backend.find_code_entity(
        canonical_name=caller_name, code_file=file_path, type_=caller_type,
    )
    if caller_id is None:
        return 0

    body = func.child_by_field_name("body")
    if body is None:
        return 0

    sf = ScopeFilter(scope="all", owner_id=owner_id)
    n_edges = 0
    for call_node in _walk_calls(body):
        callee_name = _resolve_callee_name(call_node, source_bytes, enclosing_class)
        if callee_name is None:
            continue

        # Resolution priority (most-specific → least-specific). The stub
        # path is LAST so a qualified method always wins over a bare-name
        # stub created by an earlier call expression.
        #
        # 1. In-file match — the callee is defined in the same file.
        # 2. Unique qualified suffix match — ``obj.method()`` resolves to
        #    the only ``ClassName.method`` in the KB.
        # 3. Cross-file bare-name match — a top-level function defined
        #    in another file.
        # 4. Stub creation (caller path, not in this function).
        tgt_id = backend.find_code_entity(
            canonical_name=callee_name, scope_filter=sf, code_file=file_path,
        )
        if tgt_id is None and "." not in callee_name:
            # Unique qualified suffix match.
            rows = backend._conn.execute(  # type: ignore[attr-defined]
                "SELECT id FROM entities "
                "WHERE type IN ('method','function') AND valid_until IS NULL "
                "AND canonical_name LIKE ? "
                "AND code_file IS NOT NULL "
                "AND COALESCE(code_commit_sha, '') = COALESCE(?, '')",
                (f"%.{callee_name.lower()}", commit_sha),
            ).fetchall()
            if len(rows) == 1:
                tgt_id = rows[0]["id"]
        if tgt_id is None:
            # Bare-name cross-file (top-level functions in other files).
            # Explicitly require code_file IS NOT NULL so we don't fall back
            # to picking up a pre-existing stub.
            row = backend._conn.execute(  # type: ignore[attr-defined]
                "SELECT id FROM entities "
                "WHERE canonical_name = ? AND valid_until IS NULL "
                "AND type IN ('function','method') "
                "AND code_file IS NOT NULL "
                "AND COALESCE(code_commit_sha, '') = COALESCE(?, '') "
                "LIMIT 1",
                (callee_name.lower(), commit_sha),
            ).fetchone()
            tgt_id = row["id"] if row else None
        if tgt_id is None:
            # Finally try the original find_code_entity which would pick up
            # an existing stub if one was already created on a prior call.
            tgt_id = backend.find_code_entity(
                canonical_name=callee_name, scope_filter=sf,
                code_commit_sha=commit_sha,
            )

        unresolved = tgt_id is None
        if tgt_id is None:
            # External / unknown.
            if _is_builtin(callee_name):
                # Calls to Python builtins (len, print, sum, .append, …) are
                # noise — they'd dominate the impact graph because every
                # function uses them. Skip stub creation entirely.
                continue
            # Otherwise create an inferred stub function so the graph
            # traversal has somewhere to land.
            tgt_id = backend.upsert_entity(
                type_="function",
                name=callee_name,
                scope=scope,
                owner_id=owner_id,
                valid_from=now,
                learned_at=now,
                source_ref=source_ref,
                tier="hot",
                attrs={"stub": True, "reason": "unresolved call target"},
            )

        backend.upsert_edge(
            source_id=caller_id,
            target_id=tgt_id,
            relation_type="CALLS",
            source_ref=source_ref,
            confidence_score=0.5 if unresolved else 1.0,
            confidence_class="inferred" if unresolved else "source_grounded",
            valid_from=now,
            learned_at=now,
            attrs={"line": call_node.start_point[0] + 1},
        )
        n_edges += 1
    return n_edges


def _walk_calls(node: Node):
    """Recursive iterator over all ``call`` nodes under ``node``."""
    if node.type == "call":
        yield node
    for child in node.named_children:
        yield from _walk_calls(child)


def _resolve_callee_name(
    call_node: Node, source_bytes: bytes, enclosing_class: str | None,
) -> str | None:
    """Map a call expression's function field to a canonical lookup name.

    - ``foo()``         → ``foo``
    - ``self.bar()``    → ``ClassName.bar``  (when ``enclosing_class`` is set)
    - ``mod.baz()``     → ``baz`` (we don't track the module prefix yet —
      cross-package resolution arrives in v0.3; we look up by bare name)
    - ``a.b.c()``       → ``c`` (same simplification)
    - ``foo()[0].x()``  → ``x``  (rightmost attribute)
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn, source_bytes)
    if fn.type == "attribute":
        # ``object.attribute`` — get the attribute name (rightmost part).
        attr = fn.child_by_field_name("attribute")
        obj = fn.child_by_field_name("object")
        if attr is None:
            return None
        attr_name = _text(attr, source_bytes)
        # self.method → qualify with the enclosing class.
        if (
            obj is not None
            and obj.type == "identifier"
            and _text(obj, source_bytes) == "self"
            and enclosing_class is not None
        ):
            return f"{enclosing_class}.{attr_name}"
        return attr_name
    return None
