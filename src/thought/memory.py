"""Memory — the single facade exposing ``remember`` and ``recall``.

This is the public Python entry point. The MCP server wraps this class; CLI
commands use it; tests use it directly. Keeping the orchestration in one place
means we have one set of guarantees to reason about, not two.
"""
from __future__ import annotations

import collections
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .consolidation.engine import ConsolidationEngine
from .embeddings.base import Embedder
from .embeddings.deterministic import DeterministicEmbedder
from .ingest.pipeline import IngestItem, IngestPipeline
from .models import RecallResult, RememberResult, ScopeFilter
from .router.classifier import RuleBasedClassifier
from .router.dispatcher import Dispatcher
from .storage.sqlite.backend import SQLiteBackend


def _load_embedder(choice: str, *, dim: int) -> Embedder:
    if choice == "auto":
        # Production-grade if available, fall back gracefully.
        try:
            from .embeddings.sentence_transformer import (  # noqa: F401
                SentenceTransformerEmbedder,
            )
            import sys
            sys.stderr.write(
                "[thought] auto-selected embedder: "
                "sentence-transformers/all-MiniLM-L6-v2 (384d, dense)\n"
            )
            return SentenceTransformerEmbedder()
        except ImportError:
            import sys
            sys.stderr.write(
                "[thought] auto-selected embedder: deterministic (test-grade). "
                "Install 'thought-mcp[embeddings-local]' for production quality.\n"
            )
            return DeterministicEmbedder(dim=dim)
    if choice == "deterministic":
        return DeterministicEmbedder(dim=dim)
    if choice == "minilm":  # pragma: no cover — optional dep
        try:
            from .embeddings.sentence_transformer import (
                SentenceTransformerEmbedder,
            )
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed — `pip install thought-mcp[embeddings-local]`"
            ) from e
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder choice: {choice}")


class Memory:
    """Single-process facade composing storage, ingest, dispatcher, consolidator."""

    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        consolidation_enabled: bool = False,
        recall_cache_size: int = 256,
        touch_flush_threshold: int = 32,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._pipeline = IngestPipeline(backend=backend, embedder=embedder)
        self._dispatcher = Dispatcher(
            backend=backend, embedder=embedder,
            classifier=RuleBasedClassifier.with_defaults(),
        )
        self._consolidator = ConsolidationEngine(backend=backend, embedder=embedder)
        # Recall LRU: (key) → RecallResult. Invalidates implicitly via
        # write_version embedded in the key — bumped writes don't share keys
        # with pre-write recalls.
        self._recall_cache: "collections.OrderedDict[tuple, RecallResult]" = (
            collections.OrderedDict()
        )
        self._recall_cache_size = recall_cache_size
        # Touch-access flush queue — defers the per-hit UPDATE off the hot path.
        self._touch_queue: list[str] = []
        self._touch_flush_threshold = touch_flush_threshold
        if consolidation_enabled:
            self._consolidator.start()

    # ----------------------------------------------------- factory

    @classmethod
    def open(
        cls,
        *,
        db_path: str = ".thought/thought.db",
        embedder_choice: str = "deterministic",
        embedder_dim: int = 384,
        consolidation_enabled: bool = False,
    ) -> Memory:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend = SQLiteBackend(path)
        backend.migrate()
        embedder = _load_embedder(embedder_choice, dim=embedder_dim)
        return cls(
            backend=backend, embedder=embedder,
            consolidation_enabled=consolidation_enabled,
        )

    # ----------------------------------------------------- public API

    def remember(
        self,
        *,
        content: str,
        source_ref: str | None = None,  # noqa: ARG002 — reserved for v0.2
        scope: Literal["shared", "private"] = "private",
        owner_id: str | None = None,
        now: datetime | None = None,
        unique_predicates: set[str] | None = None,
    ) -> RememberResult:
        ts = now or datetime.now(UTC)
        result = self._pipeline.ingest(
            content=content, scope=scope, owner_id=owner_id, now=ts,
            unique_predicates=unique_predicates,
        )
        return result.to_remember_result()

    def remember_many(
        self,
        items: list[IngestItem] | list[dict] | list[str],
        *,
        scope: Literal["shared", "private"] = "private",
        owner_id: str | None = None,
        now: datetime | None = None,
    ) -> list[RememberResult]:
        """Bulk-ingest path.

        ``items`` accepts ``list[IngestItem]`` for full control, ``list[dict]``
        with ``content`` / ``scope`` / ``owner_id`` keys, or ``list[str]``
        (one content per string, picking up the defaults from kwargs).
        """
        ts = now or datetime.now(UTC)
        normalised: list[IngestItem] = []
        for it in items:
            if isinstance(it, IngestItem):
                normalised.append(it)
            elif isinstance(it, dict):
                normalised.append(IngestItem(
                    content=it["content"],
                    scope=it.get("scope", scope),
                    owner_id=it.get("owner_id", owner_id),
                    unique_predicates=frozenset(it.get("unique_predicates", ())),
                ))
            else:
                normalised.append(IngestItem(
                    content=str(it), scope=scope, owner_id=owner_id,
                ))
        results = self._pipeline.ingest_many(normalised, now=ts)
        return [r.to_remember_result() for r in results]

    def recall(
        self,
        *,
        query: str,
        limit: int = 10,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
        as_of: datetime | None = None,
        as_of_kind: Literal["valid", "learned"] = "valid",
    ) -> RecallResult:
        bounded_limit = min(max(limit, 1), 10)
        # Cache key — write_version embeds the KB digest, so any write since
        # the cache entry was created invalidates this key implicitly.
        cache_key = (
            self._backend.write_version(),
            query,
            bounded_limit,
            scope,
            owner_id,
            as_of.isoformat() if as_of else None,
            as_of_kind,
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            self._recall_cache.move_to_end(cache_key)
            return cached

        result = self._dispatcher.recall(
            query=query,
            limit=bounded_limit,
            scope_filter=ScopeFilter(scope=scope, owner_id=owner_id),
            as_of=as_of,
            as_of_kind=as_of_kind,
            touch_queue=self._touch_queue,
        )
        # Flush touch-access updates if the queue is full enough to amortise.
        if len(self._touch_queue) >= self._touch_flush_threshold:
            self._flush_touch_queue()
        self._recall_cache[cache_key] = result
        if len(self._recall_cache) > self._recall_cache_size:
            self._recall_cache.popitem(last=False)
        return result

    def _flush_touch_queue(self) -> None:
        if not self._touch_queue:
            return
        self._backend.touch_access_many(self._touch_queue)
        self._touch_queue.clear()

    # ----------------------------------------------------- lifecycle

    def consolidate(self) -> int:
        self._flush_touch_queue()
        report = self._consolidator.run_once()
        return report.audit_entries

    def stats(self) -> dict[str, object]:
        """Snapshot of what's currently in the memory.

        Cheap — single SQL aggregate queries. Used by ``thought stats`` and
        by the REPL banner.
        """
        c = self._backend._conn  # type: ignore[attr-defined]
        rows = c.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM entities) AS n_entities, "
            "(SELECT COUNT(*) FROM entities WHERE valid_until IS NULL) AS n_current, "
            "(SELECT COUNT(*) FROM edges) AS n_edges, "
            "(SELECT COUNT(*) FROM edges WHERE relation_type='CONTRADICTS') AS n_contradictions, "
            "(SELECT COUNT(*) FROM sources) AS n_sources, "
            "(SELECT COUNT(*) FROM entities WHERE tier='hot') AS n_hot, "
            "(SELECT COUNT(*) FROM entities WHERE tier='warm') AS n_warm, "
            "(SELECT COUNT(*) FROM entities WHERE tier='cold') AS n_cold "
        ).fetchone()
        top = c.execute(
            "SELECT name, access_count FROM entities "
            "WHERE valid_until IS NULL ORDER BY access_count DESC LIMIT 10"
        ).fetchall()
        return {
            "entities_total": rows["n_entities"],
            "entities_current": rows["n_current"],
            "edges_total": rows["n_edges"],
            "contradictions": rows["n_contradictions"],
            "sources": rows["n_sources"],
            "tier_hot": rows["n_hot"],
            "tier_warm": rows["n_warm"],
            "tier_cold": rows["n_cold"],
            "top_accessed": [
                {"name": r["name"], "count": r["access_count"]} for r in top
            ],
            "write_version": self._backend.write_version(),
        }

    def forget(
        self,
        pattern: str,
        *,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Soft-delete entities whose canonical name matches the SQL ``LIKE``
        pattern. Sets ``valid_until = now`` on currently-valid rows and
        writes a ``FORGET`` row to the consolidation_log.

        Returns the list of retired entity IDs. Append-only: the original
        rows stay, only their validity window closes.
        """
        ts = now or datetime.now(UTC)
        sf = ScopeFilter(scope=scope, owner_id=owner_id)
        where_sql, params = sf.sql_where()
        c = self._backend._conn  # type: ignore[attr-defined]
        rows = c.execute(
            f"SELECT e.id, e.canonical_name FROM entities e "
            f"WHERE {where_sql} AND e.canonical_name LIKE ? "
            f"AND e.valid_until IS NULL",
            [*params, pattern.lower()],
        ).fetchall()
        retired: list[str] = []
        run_id = f"forget-{ts.isoformat()}"
        import json
        for r in rows:
            eid = r["id"]
            c.execute(
                "UPDATE entities SET valid_until = ? WHERE id = ?",
                (ts.isoformat(), eid),
            )
            c.execute(
                "INSERT INTO consolidation_log (run_id, op, target_kind, target_id, "
                "before_json, after_json, occurred_at, actor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, "FORGET", "entity", eid, None,
                 json.dumps({"canonical_name": r["canonical_name"]}),
                 ts.isoformat(), "user"),
            )
            retired.append(eid)
        # Invalidate caches.
        self._backend._touch_write()  # type: ignore[attr-defined]
        return retired

    def close(self) -> None:
        self._flush_touch_queue()
        self._consolidator.stop()
        self._backend.close()
