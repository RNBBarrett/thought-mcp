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


def _load_embedder(choice: str, *, dim: int, embedding_cfg=None) -> Embedder:
    if choice == "auto":
        # Production-grade if available, fall back gracefully.
        # Probe the underlying ``sentence_transformers`` package, not just our
        # wrapper — the wrapper imports cleanly even when the real dep is
        # missing (lazy load), so we have to check both.
        import importlib.util
        import sys
        if importlib.util.find_spec("sentence_transformers") is not None:
            from .embeddings.sentence_transformer import (
                SentenceTransformerEmbedder,
            )
            sys.stderr.write(
                "[thought] auto-selected embedder: "
                "sentence-transformers/all-MiniLM-L6-v2 (384d, dense)\n"
            )
            return SentenceTransformerEmbedder()
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
    # v0.4 local-LLM + remote OpenAI-compatible embedders
    if choice == "ollama":
        from .embeddings.ollama import OllamaEmbedder
        cfg = embedding_cfg
        host = getattr(cfg, "ollama_host", "http://localhost:11434")
        model = getattr(cfg, "ollama_model", "nomic-embed-text")
        return OllamaEmbedder(host=host, model=model, dim=dim)
    if choice == "lmstudio":
        from .embeddings.openai_compat import LMStudioEmbedder
        cfg = embedding_cfg
        return LMStudioEmbedder(
            base_url=getattr(cfg, "lmstudio_url", "http://localhost:1234/v1"),
            model=getattr(cfg, "lmstudio_model", "nomic-embed-text-v1.5"),
            dim=dim,
        )
    if choice == "openai-compat":
        from .embeddings.openai_compat import OpenAICompatibleEmbedder
        cfg = embedding_cfg
        return OpenAICompatibleEmbedder(
            base_url=getattr(cfg, "openai_compat_url", "http://localhost:8000/v1"),
            model=getattr(cfg, "openai_compat_model", "text-embedding-3-small"),
            api_key=(getattr(cfg, "openai_compat_api_key", "") or None),
            dim=dim,
        )
    if choice == "openai":
        from .embeddings.openai_compat import OpenAIEmbedder
        cfg = embedding_cfg
        return OpenAIEmbedder(
            model=getattr(cfg, "openai_compat_model", "text-embedding-3-small"),
            api_key=(getattr(cfg, "openai_compat_api_key", "") or None),
            dim=dim,
        )
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
        self._recall_cache: collections.OrderedDict[tuple, RecallResult] = (
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
        embedding_cfg=None,
    ) -> Memory:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend = SQLiteBackend(path)
        backend.migrate()
        embedder = _load_embedder(
            embedder_choice, dim=embedder_dim, embedding_cfg=embedding_cfg,
        )
        return cls(
            backend=backend, embedder=embedder,
            consolidation_enabled=consolidation_enabled,
        )

    # ----------------------------------------------------- public API

    def remember(
        self,
        *,
        content: str,
        source_ref: str | None = None,
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

    # ----------------------------------------------------- db lifecycle (v0.4)

    def db_size(self) -> dict:
        """Disk usage + entity/edge counts. Powers ``thought db size``."""
        sizes = self._backend.file_sizes()
        s = self.stats()
        return {
            "path": str(self._backend.path),
            **sizes,
            "entities_current": s["entities_current"],
            "entities_total": s["entities_total"],
            "edges": s["edges_total"],
            "sources": s["sources"],
        }

    def flush(
        self,
        *,
        confirm: bool,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: Literal["created", "valid", "learned"] = "created",
    ) -> dict[str, int]:
        """Destructive wipe. ``confirm=True`` is required; SDK guard rail."""
        if not confirm:
            raise ValueError(
                "Memory.flush() requires confirm=True. This is a destructive "
                "operation that drops or deletes data."
            )
        self._flush_touch_queue()
        # Invalidate the recall cache — flushed data must not surface.
        self._recall_cache.clear()
        return self._backend.flush(
            before=before, since=since, time_axis=time_axis,
        )

    def backup_to(
        self,
        path: str | Path,
        *,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: Literal["created", "valid", "learned"] = "created",
        force: bool = False,
    ) -> int:
        """Snapshot the current DB to ``path``. Returns bytes written."""
        p = Path(path)
        if p.exists() and not force:
            raise FileExistsError(
                f"{p} already exists. Pass force=True (or --force on the CLI) to overwrite."
            )
        self._flush_touch_queue()
        return self._backend.backup_to(
            p, before=before, since=since, time_axis=time_axis,
        )

    def load_from(
        self,
        path: str | Path,
        *,
        merge: bool = False,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: Literal["created", "valid", "learned"] = "created",
    ) -> dict:
        """Load a snapshot.

        - ``merge=False`` (default): caller is responsible for swapping the
          underlying file. This method validates the source + returns a
          summary; the actual file swap is performed by the CLI which closes
          this Memory instance, moves files, and re-opens.
        - ``merge=True``: row-level merge into the live DB via the backend's
          INSERT-OR-IGNORE path. Idempotent.

        For ``merge=False``, returns ``{"action": "replace", "source": ..., "size": ...}``
        and the CLI handles file IO. For ``merge=True``, returns
        ``{"action": "merge", "new_entities": N, "new_edges": M, "new_sources": K}``.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"snapshot file not found: {p}")
        # Quickly validate it's a SQLite DB with a compatible schema.
        try:
            tmp = self._backend.__class__.open_readonly(p)
        except Exception as e:
            raise ValueError(
                f"{p} doesn't look like a valid THOUGHT snapshot ({e})"
            ) from e
        try:
            src_ver = tmp.schema_version()
        finally:
            tmp.close()
        cur_ver = self._backend.schema_version()
        if src_ver > cur_ver:
            raise ValueError(
                f"snapshot schema_version={src_ver} is higher than this "
                f"binary's schema_version={cur_ver}; upgrade thought-mcp first."
            )

        if merge:
            self._flush_touch_queue()
            self._recall_cache.clear()
            counts = self._backend.merge_from(
                p, before=before, since=since, time_axis=time_axis,
            )
            return {"action": "merge", **counts, "source": str(p)}
        return {
            "action": "replace",
            "source": str(p),
            "size": p.stat().st_size,
            "schema_version": src_ver,
        }

    # ----------------------------------------------------- agent identity (v0.5)

    def register_agent(
        self, name: str, *, description: str | None = None,
        capabilities: list[str] | None = None,
    ) -> dict:
        """Register-or-touch a named agent. Returns agent record."""
        aid = self._backend.upsert_agent(
            name=name, description=description, capabilities=capabilities,
        )
        return self._backend.find_agent_by_name(name) or {"id": aid, "name": name}

    def list_agents(self) -> list[dict]:
        return self._backend.list_agents()

    # ----------------------------------------------------- scan (v0.5)

    def scan(
        self,
        repo_path: str | Path,
        *,
        agent: str | None = None,
        since: str | None = None,
        max_files: int | None = None,
        language: str | None = None,
        note: str | None = None,
    ) -> dict:
        """Incremental code-scan primitive for agent loops.

        Walks ``repo_path``, ingests changed/new files via the existing
        :class:`CodeIngestPipeline`, retires entities for deleted files, and
        records a row in ``scan_log`` so the next call picks up where this
        one left off without manual state tracking.

        Returns a structured summary:
        ``{"scan_id", "files_scanned", "files_changed", "entities_added",
           "entities_retired", "edges_added", "edges_retired", "duration_ms"}``.
        """
        from .ingest.code.ast_extractor import detect_language
        from .ingest.code.pipeline import CodeIngestPipeline

        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"scan path not found: {repo}")

        started = datetime.now(UTC)
        agent_id: str | None = None
        if agent:
            agent_id = self._backend.upsert_agent(name=agent)

        # If ``since`` wasn't provided, use the last successful scan's head_sha
        # for this (agent, repo) tuple — the auto-cursor.
        if since is None:
            last = self._backend.last_scan(
                agent_id=agent_id, repo_path=str(repo),
            )
            if last is not None:
                since = last.get("head_sha") or None

        # Find current HEAD SHA via subprocess (no pygit2 dep needed).
        head_sha: str | None = None
        try:
            import subprocess
            head_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo), encoding="utf-8",
            ).strip() or None
        except Exception:  # pragma: no cover — repo without git
            head_sha = None

        # Determine which files to scan. Two modes:
        # (a) `since` provided + repo is a git repo → use `git diff --name-only since..HEAD`
        # (b) otherwise → full scan of all supported source files.
        files_to_scan: list[Path] = []
        if since and head_sha is not None:
            try:
                import subprocess
                diff = subprocess.check_output(
                    ["git", "diff", "--name-only", f"{since}..HEAD"],
                    cwd=str(repo), encoding="utf-8",
                )
                changed = [
                    Path(repo) / line.strip()
                    for line in diff.splitlines()
                    if line.strip()
                ]
                files_to_scan = [
                    f for f in changed
                    if f.exists() and detect_language(f.name) is not None
                ]
            except Exception:  # pragma: no cover
                files_to_scan = []
        if not files_to_scan:
            # Full scan.
            for f in repo.rglob("*"):
                if not f.is_file():
                    continue
                lang = detect_language(f.name)
                if lang is None:
                    continue
                if language is not None and lang != language:
                    continue
                files_to_scan.append(f)
        if max_files is not None:
            files_to_scan = files_to_scan[:max_files]

        # Snapshot entity / edge counts so we can compute deltas.
        before_ent = int(self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM entities WHERE valid_until IS NULL"
        ).fetchone()["n"])
        before_edge = int(self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM edges WHERE valid_until IS NULL"
        ).fetchone()["n"])

        pipeline = CodeIngestPipeline(
            backend=self._backend, embedder=self._embedder,
            scope="shared",
        )

        files_scanned = 0
        for f in files_to_scan:
            try:
                rel = f.resolve().relative_to(repo).as_posix()
            except ValueError:
                rel = str(f)
            try:
                pipeline.ingest_code_file(
                    f, commit_sha=head_sha,
                    language=detect_language(f.name),
                    repo_root=repo,
                    now=started,
                )
                files_scanned += 1
            except Exception as e:  # pragma: no cover — extractor errors
                import sys
                print(
                    f"thought scan: skipping {rel} ({e})",
                    file=sys.stderr,
                )

        after_ent = int(self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM entities WHERE valid_until IS NULL"
        ).fetchone()["n"])
        after_edge = int(self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM edges WHERE valid_until IS NULL"
        ).fetchone()["n"])

        finished = datetime.now(UTC)
        duration_ms = (finished - started).total_seconds() * 1000

        scan_id = self._backend.record_scan(
            agent_id=agent_id, repo_path=str(repo), since=since,
            head_sha=head_sha,
            started_at=started, finished_at=finished,
            files_scanned=files_scanned,
            files_changed=len(files_to_scan),
            entities_added=max(0, after_ent - before_ent),
            entities_retired=0,
            edges_added=max(0, after_edge - before_edge),
            edges_retired=0,
            duration_ms=duration_ms,
            note=note,
        )
        return {
            "scan_id": scan_id,
            "agent": agent,
            "head_sha": head_sha,
            "since": since,
            "files_scanned": files_scanned,
            "files_changed": len(files_to_scan),
            "entities_added": max(0, after_ent - before_ent),
            "edges_added": max(0, after_edge - before_edge),
            "duration_ms": round(duration_ms, 2),
        }

    def scan_log(
        self, *, agent: str | None = None, limit: int = 10,
    ) -> list[dict]:
        agent_id: str | None = None
        if agent:
            rec = self._backend.find_agent_by_name(agent)
            if rec is None:
                return []
            agent_id = rec["id"]
        return self._backend.list_scans(agent_id=agent_id, limit=limit)

    # ----------------------------------------------------- working_context (v0.5)

    def working_context(
        self,
        target: str,
        *,
        role: str | None = None,
        budget_tokens: int = 2000,
        scope: str = "all",
        owner_id: str | None = None,
    ) -> dict:
        """Universal *"what does my agent need to know right now"* primitive.

        Returns a structured + token-budgeted context payload for an agent
        working on ``target``. Combines:

        - PPR-ranked neighbours of the seed entity
        - Recent contradictions in scope
        - Saved views relevant to the role (if a view named like the role exists)

        ``target`` is matched as ``"<type>:<name>"`` (e.g. ``"function:authenticate"``)
        or just an entity name.
        """
        from .layers.graph import GraphLayer

        # Approximate-tokens budget. Rough heuristic: ~4 chars per token.
        budget_chars = int(budget_tokens * 4)

        # Parse "type:name" if present.
        type_filter: str | None = None
        name = target
        if ":" in target:
            type_filter, _, name = target.partition(":")

        scope_filter = ScopeFilter(scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
        anchor = self._backend.find_anchor_by_name(name, scope_filter)
        if anchor is None and type_filter is not None:
            # Try `find_code_entity` for code-vertical lookups.
            try:
                eid = self._backend.find_code_entity(
                    canonical_name=name, type_=type_filter,
                )
                anchor = self._backend.get_entity(eid) if eid else None
            except Exception:
                anchor = None

        graph = GraphLayer(self._backend)
        neighbours: list[dict] = []
        if anchor is not None:
            scores = graph.personalized_pagerank(
                seeds=[anchor.id], scope_filter=scope_filter,
            )
            top = sorted(scores.items(), key=lambda kv: -kv[1])[: 20]
            for eid, score in top:
                if eid == anchor.id:
                    continue
                e = self._backend.get_entity(eid)
                if e is None:
                    continue
                neighbours.append({
                    "name": e.name, "type": e.type,
                    "score": float(score),
                })

        # Recent contradictions in scope.
        recent_contradictions = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT source_id, target_id, relation_type, detected_at "
            "FROM edges WHERE relation_type='CONTRADICTS' "
            "ORDER BY detected_at DESC LIMIT 5"
        ).fetchall()

        # Saved view named after the role, if any.
        role_view = None
        if role:
            try:
                from .query import views as views_mod
                role_view = views_mod.show_view(self, role)
            except Exception:
                role_view = None

        payload = {
            "target": target,
            "role": role,
            "anchor": (
                {"id": anchor.id, "name": anchor.name, "type": anchor.type}
                if anchor else None
            ),
            "neighbours": neighbours,
            "recent_contradictions": [dict(r) for r in recent_contradictions],
            "role_view": role_view,
        }
        # Rough char-budget trim: drop low-score neighbours until we fit.
        import json as _json
        while len(_json.dumps(payload)) > budget_chars and neighbours:
            neighbours.pop()
            payload["neighbours"] = neighbours
        return payload

    def schema_summary(self) -> dict[str, dict[str, int]]:
        """Counts of entity types and edge relation types currently in the KB.

        Powers ``thought schema`` and is injected into auto-recall + auto-context
        hooks so the agent knows what's queryable. Cheap — two GROUP BY queries.
        """
        c = self._backend._conn  # type: ignore[attr-defined]
        etypes = {
            r["t"]: r["c"] for r in c.execute(
                "SELECT type AS t, COUNT(*) AS c FROM entities "
                "WHERE valid_until IS NULL GROUP BY type ORDER BY c DESC"
            ).fetchall()
        }
        relations = {
            r["t"]: r["c"] for r in c.execute(
                "SELECT relation_type AS t, COUNT(*) AS c FROM edges "
                "WHERE valid_until IS NULL GROUP BY relation_type ORDER BY c DESC"
            ).fetchall()
        }
        return {"entity_types": etypes, "relation_types": relations}

    def reembed_to(
        self,
        new_embedder_choice: str,
        *,
        new_dim: int | None = None,
        embedding_cfg=None,
        batch_size: int = 32,
        progress: object | None = None,
    ) -> dict:
        """Re-embed every entity through a different embedder.

        Lets users start with ``deterministic`` and upgrade to Ollama /
        sentence-transformers later without re-ingesting from source.
        Re-embeds the entity's ``name`` (the same signal the ingest pipeline
        uses for canonical-name lookups). Returns ``{"reembedded": N, "dim": D, "model": ...}``.

        ``progress`` is an optional callable ``progress(advance: int)`` for
        CLI integration with ``rich.Progress``.
        """
        new_embedder = _load_embedder(
            new_embedder_choice,
            dim=new_dim if new_dim is not None else self._embedder.dim,
            embedding_cfg=embedding_cfg,
        )
        # All currently-valid entities; iterate by name (the ingest signal).
        rows = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id, name FROM entities WHERE valid_until IS NULL"
        ).fetchall()
        from .embeddings.base import vector_to_bytes
        n = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["name"] for r in batch]
            vectors = new_embedder.embed_many(texts)
            for r, v in zip(batch, vectors, strict=True):
                self._backend.store_embedding(
                    entity_id=r["id"],
                    model_name=new_embedder.model_name,
                    model_version=new_embedder.model_version,
                    dim=new_embedder.dim,
                    vector=vector_to_bytes(v),
                )
                n += 1
            if progress is not None:
                progress(len(batch))  # type: ignore[misc]
        # Recall cache embeds model identity implicitly via write_version,
        # but be defensive: blow it away so nothing surfaces from a stale embed.
        self._recall_cache.clear()
        return {
            "reembedded": n,
            "model": new_embedder.model_name,
            "dim": new_embedder.dim,
        }

    def inspect_file(
        self,
        path: str | Path,
        *,
        include_schema: bool = False,
    ) -> dict:
        """Stats (+ optional schema) of a backup file without loading it."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"snapshot file not found: {p}")
        backend = self._backend.__class__.open_readonly(p)
        try:
            rows = backend._conn.execute(  # type: ignore[attr-defined]
                "SELECT "
                "(SELECT COUNT(*) FROM entities) AS n_entities, "
                "(SELECT COUNT(*) FROM entities WHERE valid_until IS NULL) AS n_current, "
                "(SELECT COUNT(*) FROM edges) AS n_edges, "
                "(SELECT COUNT(*) FROM edges WHERE relation_type='CONTRADICTS') AS n_contradictions, "
                "(SELECT COUNT(*) FROM sources) AS n_sources"
            ).fetchone()
            summary = {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "schema_version": backend.schema_version(),
                "entities_total": rows["n_entities"],
                "entities_current": rows["n_current"],
                "edges": rows["n_edges"],
                "contradictions": rows["n_contradictions"],
                "sources": rows["n_sources"],
            }
            if include_schema:
                # Mini schema-summary: counts by entity type + edge relation.
                etypes = {
                    r["t"]: r["c"] for r in backend._conn.execute(  # type: ignore[attr-defined]
                        "SELECT type AS t, COUNT(*) AS c FROM entities "
                        "WHERE valid_until IS NULL GROUP BY type ORDER BY c DESC"
                    ).fetchall()
                }
                relations = {
                    r["t"]: r["c"] for r in backend._conn.execute(  # type: ignore[attr-defined]
                        "SELECT relation_type AS t, COUNT(*) AS c FROM edges "
                        "GROUP BY relation_type ORDER BY c DESC"
                    ).fetchall()
                }
                summary["entity_types"] = etypes
                summary["relation_types"] = relations
            return summary
        finally:
            backend.close()

    # ----------------------------------------------------- topic browsing

    def list_topics(
        self,
        *,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
        min_count: int = 1,
        examples_per_type: int = 3,
    ) -> list[dict[str, object]]:
        """Return entity-type aggregations + a small example list per type.

        Powers ``thought topics`` / ``mcp__thought__list_topics``. Cheap —
        one GROUP BY + one SELECT per type (capped at the ten most-populous
        types so a runaway type doesn't dominate the query budget).
        """
        sf = ScopeFilter(scope=scope, owner_id=owner_id)
        counts = self._backend.count_by_type(sf)
        # Keep all types meeting min_count; ordered by count desc from the
        # backend already. Limit example fetches to the top types.
        topics: list[dict[str, object]] = []
        for t, c in counts.items():
            if c < min_count:
                continue
            where_sql, params = sf.sql_where()
            rows = self._backend._conn.execute(  # type: ignore[attr-defined]
                f"SELECT e.name FROM entities e WHERE {where_sql} "
                f"AND e.valid_until IS NULL AND e.type = ? "
                f"ORDER BY e.access_count DESC, e.importance DESC "
                f"LIMIT ?",
                [*params, t, examples_per_type],
            ).fetchall()
            topics.append({
                "type": t,
                "count": c,
                "examples": [r["name"] for r in rows],
            })
        return topics

    def browse_topic(
        self,
        name: str,
        *,
        depth: int = 1,
        limit: int = 20,
        scope: Literal["shared", "private", "all"] = "all",
        owner_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Drill into a topic anchored at ``name``.

        Two-step resolution:
          1. If ``name`` matches a known entity-type (e.g. ``PERSON``,
             ``function``), return the top-access-count entities of that type.
          2. Otherwise treat ``name`` as an entity name, find the matching
             anchor entity, and return its PPR-ranked neighbourhood. Falls
             back to BFS-neighbours if PPR returns nothing meaningful.
        """
        sf = ScopeFilter(scope=scope, owner_id=owner_id)
        # Case 1: literal type match (case-insensitive).
        types = self._backend.count_by_type(sf)
        type_key = next((k for k in types if k.lower() == name.lower()), None)
        if type_key is not None:
            where_sql, params = sf.sql_where()
            rows = self._backend._conn.execute(  # type: ignore[attr-defined]
                f"SELECT e.* FROM entities e WHERE {where_sql} "
                f"AND e.valid_until IS NULL AND e.type = ? "
                f"ORDER BY e.access_count DESC, e.importance DESC, e.created_at "
                f"LIMIT ?",
                [*params, type_key, limit],
            ).fetchall()
            return [
                {
                    "id": self._backend._row_to_entity(r).id,  # type: ignore[attr-defined]
                    "name": r["name"],
                    "type": r["type"],
                    "score": None,
                    "via": "type_facet",
                }
                for r in rows
            ]

        # Case 2: anchor-by-name → graph neighbourhood.
        anchor = self._backend.find_anchor_by_name(name, sf)
        if anchor is None:
            return []

        from .layers.graph import GraphLayer
        gl = GraphLayer(self._backend)
        # Try PPR for ranking; fall back to BFS if PPR is empty (e.g. an
        # isolated anchor with no outgoing edges).
        scores = gl.personalized_pagerank(seeds=[anchor.id], scope_filter=sf)
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit + 1]
            results: list[dict[str, object]] = []
            for eid, score in ranked:
                if eid == anchor.id:
                    continue
                e = self._backend.get_entity(eid)
                if e is None:
                    continue
                results.append({
                    "id": e.id, "name": e.name, "type": e.type,
                    "score": float(score), "via": "ppr",
                })
                if len(results) >= limit:
                    break
            if results:
                return results
        # Fallback: BFS neighbours.
        neighbours = gl.neighbors(anchor.id, depth=depth, scope_filter=sf)
        return [
            {"id": n.id, "name": n.name, "type": n.type,
             "score": None, "via": "bfs"}
            for n in neighbours[:limit]
        ]

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
