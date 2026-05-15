"""SQLite + sqlite-vec storage backend.

Append-only semantics — supersession / retirement set ``valid_until`` and
``unlearned_at`` columns and add new ``SUPERSEDES`` edges, never delete rows.

ANN: when the sqlite-vec extension can be loaded, per-dim ``vec_float_<dim>``
and ``vec_bit_<dim>`` virtual tables are created lazily so the vector layer
can issue ``MATCH``-based KNN queries that complete in O(log N) instead of
the Python-side O(N) brute-force fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from ulid import ULID

from ...models import Edge, Entity, ScopeFilter, ScopeName
from ..base import StorageBackend

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_log = logging.getLogger(__name__)


def _sign_pack(vector_bytes: bytes, dim: int) -> bytes:
    """Sign-quantize an fp32 vector into a bit-packed binary blob.

    Each input dimension contributes one bit (1 if >= 0, else 0). The result
    is ``ceil(dim/8)`` bytes — ~32× smaller than the fp32 source. Hamming
    distance over these bits approximates cosine similarity (Charikar 2002,
    "Similarity estimation techniques from rounding algorithms"). sqlite-vec
    BIT[N] expects exactly ceil(N/8) bytes.
    """
    v = np.frombuffer(vector_bytes, dtype=np.float32, count=dim)
    bits = (v >= 0).astype(np.uint8)
    # Pack 8 bits → 1 byte, MSB-first (sqlite-vec convention).
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the sqlite-vec loadable extension.

    Returns True iff vec0 virtual tables can be created on ``conn``. Failures
    are logged at DEBUG level — they're expected on Python distros that lack
    ``enable_load_extension`` (Anaconda) and we degrade to the pure-Python
    fallback.
    """
    if not hasattr(conn, "enable_load_extension"):
        _log.debug("sqlite-vec: enable_load_extension unavailable on this Python")
        return False
    try:
        import sqlite_vec  # type: ignore[import-untyped]
    except ImportError:
        _log.debug("sqlite-vec: package not installed")
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except sqlite3.OperationalError as e:  # pragma: no cover — env-specific
        _log.debug("sqlite-vec: load failed: %s", e)
        return False


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _canon(name: str) -> str:
    return name.strip().lower()


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(ULID())


class SQLiteBackend(StorageBackend):
    """A single-file SQLite backend.

    Pass ``":memory:"`` for tests, or a filesystem path for persistence.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        # ``check_same_thread=False`` is required because (a) the MCP server
        # dispatches each tool call into a worker thread via
        # ``asyncio.to_thread`` so the connection is touched from a thread
        # different from the one that created it, and (b) the consolidation
        # engine runs in its own background thread. SQLite's C-level mutex
        # (FULLMUTEX, the Python default) serializes access at the engine
        # level, so this is safe in practice; the Python ``check_same_thread``
        # guard is a debug aid, not a correctness requirement.
        self._conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Perf tuning: larger page cache, memory-mapped reads, NORMAL sync
        # (WAL guarantees durability for committed transactions even at this
        # level — full FSYNC is overkill for a single-process memory store).
        self._conn.execute("PRAGMA cache_size = -65536")   # 64 MiB
        self._conn.execute("PRAGMA mmap_size = 268435456")  # 256 MiB
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        # Wait up to 5s on lock contention before raising — needed because
        # WAL mode permits concurrent readers but a writer can still block
        # briefly on checkpointing.
        self._conn.execute("PRAGMA busy_timeout = 5000")

        self._vec_available = _try_load_sqlite_vec(self._conn)
        self._vec_tables: set[tuple[int, str]] = set()  # cache: (dim, kind)
        # Monotonic write-version: bumped on every mutating call. Read-side
        # caches (recall LRU, PPR matrix) snapshot this token and invalidate
        # automatically when it advances.
        self._write_version: int = 0

    # ---- lifecycle ----

    def migrate(self) -> None:
        # Track applied migrations by filename so each file runs exactly once,
        # even across multiple ``migrate()`` calls on the same DB. ``ALTER
        # TABLE ADD COLUMN`` isn't idempotent in SQLite, so we can't rely on
        # the migrations themselves being safe to re-run.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS applied_migrations ("
            " filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            r["filename"] for r in
            self._conn.execute("SELECT filename FROM applied_migrations").fetchall()
        }
        for sql_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            if sql_file.name in applied:
                continue
            self._conn.executescript(sql_file.read_text(encoding="utf-8"))
            self._conn.execute(
                "INSERT INTO applied_migrations (filename, applied_at) VALUES (?, ?)",
                (sql_file.name, _utc_iso(_now_utc())),
            )

    def close(self) -> None:
        self._conn.close()

    def write_version(self) -> int:
        """Monotonic token. Increments on every mutating call.

        Read-side caches embed this into their key and re-fetch when they
        notice a bump.
        """
        return self._write_version

    def _touch_write(self) -> None:
        self._write_version += 1

    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [r["name"] for r in rows]

    # ---- sources ----

    def upsert_source(
        self,
        content: str,
        *,
        mime_type: str = "text/plain",
        context_summary: str | None = None,
    ) -> str:
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        row = self._conn.execute(
            "SELECT id FROM sources WHERE content_hash = ?", (h,)
        ).fetchone()
        if row is not None:
            return row["id"]
        sid = _new_id()
        self._conn.execute(
            "INSERT INTO sources (id, content, content_hash, mime_type, ingested_at, context_summary) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, content, h, mime_type, _utc_iso(_now_utc()), context_summary),
        )
        return sid

    def get_source_id_by_hash(self, content: str) -> str | None:
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        row = self._conn.execute(
            "SELECT id FROM sources WHERE content_hash = ?", (h,)
        ).fetchone()
        return row["id"] if row else None

    # ---- entities ----

    def upsert_entity(
        self,
        *,
        type_: str,
        name: str,
        scope: ScopeName,
        valid_from: datetime,
        learned_at: datetime,
        source_ref: str,
        owner_id: str | None = None,
        importance: float = 0.5,
        tier: str = "hot",
        attrs: dict[str, object] | None = None,
        code_file: str | None = None,
        code_language: str | None = None,
        code_commit_sha: str | None = None,
    ) -> str:
        # Append-only: check for an existing currently-valid entity with same identity.
        # If one exists with valid_until IS NULL we reuse it; otherwise add a new row.
        # For code entities, identity ALSO includes (code_file, code_commit_sha) so we
        # don't merge functions of the same name from different files.
        canonical = _canon(name)
        existing = self._conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ? AND type = ? "
            "AND scope = ? AND COALESCE(owner_id, '') = COALESCE(?, '') "
            "AND COALESCE(code_file, '') = COALESCE(?, '') "
            "AND COALESCE(code_commit_sha, '') = COALESCE(?, '') "
            "AND valid_until IS NULL "
            "LIMIT 1",
            (canonical, type_, scope, owner_id, code_file, code_commit_sha),
        ).fetchone()
        if existing is not None:
            return existing["id"]

        eid = _new_id()
        now = _now_utc()
        self._conn.execute(
            "INSERT INTO entities ("
            " id, type, name, canonical_name, owner_id, scope, tier, importance,"
            " valid_from, valid_until, learned_at, unlearned_at,"
            " created_at, last_accessed_at, access_count, attrs_json,"
            " code_file, code_language, code_commit_sha"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, 0, ?, ?, ?, ?)",
            (
                eid,
                type_,
                name,
                canonical,
                owner_id,
                scope,
                tier,
                importance,
                _utc_iso(valid_from),
                _utc_iso(learned_at),
                _utc_iso(now),
                _utc_iso(now),
                json.dumps(attrs or {}),
                code_file,
                code_language,
                code_commit_sha,
            ),
        )
        return eid

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            type=row["type"],
            name=row["name"],
            canonical_name=row["canonical_name"],
            owner_id=row["owner_id"],
            scope=row["scope"],
            tier=row["tier"],
            importance=row["importance"],
            valid_from=_parse_dt(row["valid_from"]),  # type: ignore[arg-type]
            valid_until=_parse_dt(row["valid_until"]),
            learned_at=_parse_dt(row["learned_at"]),  # type: ignore[arg-type]
            unlearned_at=_parse_dt(row["unlearned_at"]),
            created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
            last_accessed_at=_parse_dt(row["last_accessed_at"]),  # type: ignore[arg-type]
            access_count=row["access_count"],
            attrs=json.loads(row["attrs_json"]) if row["attrs_json"] else {},
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def list_entities(self, scope_filter: ScopeFilter) -> list[Entity]:
        where_sql, params = scope_filter.sql_where()
        rows = self._conn.execute(
            f"SELECT e.* FROM entities e WHERE {where_sql} ORDER BY e.created_at",
            params,
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def count_by_type(self, scope_filter: ScopeFilter) -> dict[str, int]:
        where_sql, params = scope_filter.sql_where()
        rows = self._conn.execute(
            f"SELECT e.type AS t, COUNT(*) AS c FROM entities e "
            f"WHERE {where_sql} AND e.valid_until IS NULL "
            f"GROUP BY e.type ORDER BY c DESC",
            params,
        ).fetchall()
        return {r["t"]: r["c"] for r in rows}

    def find_anchor_by_name(
        self, name: str, scope_filter: ScopeFilter,
    ) -> Entity | None:
        where_sql, params = scope_filter.sql_where()
        canonical = _canon(name)
        row = self._conn.execute(
            f"SELECT e.* FROM entities e "
            f"WHERE {where_sql} "
            f"  AND e.valid_until IS NULL "
            f"  AND e.canonical_name = ? "
            f"ORDER BY e.access_count DESC, e.importance DESC, e.created_at ASC "
            f"LIMIT 1",
            [*params, canonical],
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_code_entity(
        self,
        *,
        canonical_name: str,
        scope_filter: ScopeFilter | None = None,
        code_file: str | None = None,
        type_: str | None = None,
        code_commit_sha: str | None = None,
    ) -> str | None:
        """Look up a code entity ID by name + optional disambiguators.

        Used by the call-graph resolver: given a callee name observed in a
        function body, find the entity that name refers to. ``code_file``
        narrows to intra-file calls; ``type_`` filters to function | method;
        ``code_commit_sha`` pins to a specific snapshot.
        """
        clauses = ["canonical_name = ?", "valid_until IS NULL"]
        params: list[object] = [_canon(canonical_name)]
        if scope_filter is not None:
            where_sql, sp = scope_filter.sql_where()
            # ``where_sql`` references ``e.`` aliases; rewrite for table-only context.
            clauses.append(where_sql.replace("e.", ""))
            params.extend(sp)
        if code_file is not None:
            clauses.append("code_file = ?")
            params.append(code_file)
        if type_ is not None:
            clauses.append("type = ?")
            params.append(type_)
        if code_commit_sha is not None:
            clauses.append("code_commit_sha = ?")
            params.append(code_commit_sha)
        row = self._conn.execute(
            f"SELECT id FROM entities WHERE {' AND '.join(clauses)} LIMIT 1",
            params,
        ).fetchone()
        return row["id"] if row else None

    def list_entity_ids(self, scope_filter: ScopeFilter) -> set[str]:
        """Fast scope-membership lookup — IDs only, no Pydantic hydration.

        Profile-driven: ``list_entities`` was the dominant cost on hot recall
        paths (376ms / call at 5k entities) because callers only ever needed
        the ID set. Hydrating every row through Pydantic + JSON parse to
        produce a set comprehension is pure waste.
        """
        where_sql, params = scope_filter.sql_where()
        rows = self._conn.execute(
            f"SELECT e.id FROM entities e WHERE {where_sql}", params
        ).fetchall()
        return {r["id"] for r in rows}

    def fetch_edges_in_scope(
        self,
        allowed_ids: set[str],
        *,
        exclude_relations: frozenset[str] = frozenset({"CONTRADICTS"}),
    ) -> list[tuple[str, str, float, str]]:
        """Bulk-fetch all in-scope edges as lightweight tuples.

        Returns ``(source_id, target_id, confidence_score, relation_type)``
        for every edge whose endpoints are both in ``allowed_ids`` and whose
        relation type is not in ``exclude_relations``.

        Pushes the relation-type filter down to SQL via the
        ``idx_edges_relation`` index so we don't scan and filter in Python.
        Endpoint membership is still checked in Python because emitting a
        WHERE IN (...) of arbitrary size kills query planning above ~999
        params; the row hydration here is cheap anyway (no JSON, no
        Pydantic).
        """
        if not allowed_ids:
            return []
        # ``NOT IN (...)`` over a small fixed set of meta-relations is fast.
        placeholders = ",".join("?" for _ in exclude_relations) or "''"
        rows = self._conn.execute(
            f"SELECT source_id, target_id, confidence_score, relation_type "
            f"FROM edges WHERE relation_type NOT IN ({placeholders})",
            list(exclude_relations),
        ).fetchall()
        out: list[tuple[str, str, float, str]] = []
        for r in rows:
            sid = r["source_id"]
            tid = r["target_id"]
            if sid not in allowed_ids or tid not in allowed_ids:
                continue
            out.append((sid, tid, float(r["confidence_score"]), r["relation_type"]))
        return out

    def sources_for_entities(self, entity_ids: list[str]) -> list[dict]:
        """One-query source provenance fetch for a list of result entities.

        Returns ``[{id, content_hash, ingested_at}]`` deduped by source id.
        Replaces the N+M roundtrip pattern (``edges_to`` per hit, ``SELECT
        source`` per ref) with a single JOIN.
        """
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self._conn.execute(
            f"SELECT DISTINCT s.id, s.content_hash, s.ingested_at "
            f"FROM sources s "
            f"JOIN edges e ON e.source_ref = s.id "
            f"WHERE e.target_id IN ({placeholders}) OR e.source_id IN ({placeholders})",
            list(entity_ids) + list(entity_ids),
        ).fetchall()
        return [
            {"id": r["id"], "content_hash": r["content_hash"], "ingested_at": r["ingested_at"]}
            for r in rows
        ]

    def list_entities_at(
        self,
        when: datetime,
        scope_filter: ScopeFilter,
        *,
        kind: str = "valid",
    ) -> list[Entity]:
        """Return entities present at ``when`` on either temporal axis.

        - ``kind='valid'``: world-time. Entity is included if its valid_from
          <= when AND (valid_until IS NULL OR valid_until > when).
        - ``kind='learned'``: transaction-time. Same logic on learned_at /
          unlearned_at.
        """
        where_sql, params = scope_filter.sql_where()
        when_iso = _utc_iso(when)
        if kind == "valid":
            window = "e.valid_from <= ? AND (e.valid_until IS NULL OR e.valid_until > ?)"
        elif kind == "learned":
            window = "e.learned_at <= ? AND (e.unlearned_at IS NULL OR e.unlearned_at > ?)"
        else:
            raise ValueError(f"kind must be 'valid' or 'learned', got {kind!r}")
        rows = self._conn.execute(
            f"SELECT e.* FROM entities e WHERE {where_sql} AND {window} "
            f"ORDER BY e.created_at",
            [*params, when_iso, when_iso],
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def set_tier(self, entity_id: str, tier: str) -> None:
        if tier not in {"hot", "warm", "cold"}:
            raise ValueError(f"invalid tier: {tier!r}")
        self._conn.execute(
            "UPDATE entities SET tier = ? WHERE id = ?", (tier, entity_id)
        )
        self._touch_write()

    def stale_warm_candidates(self, cutoff: datetime) -> list[Entity]:
        rows = self._conn.execute(
            "SELECT e.* FROM entities e WHERE tier = 'warm' "
            "AND last_accessed_at < ? "
            "ORDER BY last_accessed_at",
            [_utc_iso(cutoff)],
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def touch_access(self, entity_id: str) -> None:
        self._conn.execute(
            "UPDATE entities SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE id = ?",
            (_utc_iso(_now_utc()), entity_id),
        )
        # Touch-access doesn't change graph structure, so we deliberately do
        # NOT bump _write_version here — that would invalidate the recall
        # LRU on every successful read. The touched row's last_accessed_at
        # is metadata, not retrieval-relevant.

    def touch_access_many(self, entity_ids: list[str]) -> None:
        """Batched touch-access — flushes the dispatcher's deferred queue."""
        if not entity_ids:
            return
        now_iso = _utc_iso(_now_utc())
        self._conn.executemany(
            "UPDATE entities SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE id = ?",
            [(now_iso, eid) for eid in entity_ids],
        )

    # ---- edges ----

    def upsert_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        source_ref: str,
        confidence_score: float,
        valid_from: datetime,
        learned_at: datetime,
        confidence_class: str = "source_grounded",
        attrs: dict[str, object] | None = None,
    ) -> str:
        # Sanity: verify the source_ref exists (catch programmer errors early).
        if not self._conn.execute(
            "SELECT 1 FROM sources WHERE id = ?", (source_ref,)
        ).fetchone():
            raise ValueError(f"source_ref {source_ref!r} does not exist")

        # Check cross_scope.
        src_scope = self._conn.execute(
            "SELECT scope, owner_id FROM entities WHERE id = ?", (source_id,)
        ).fetchone()
        tgt_scope = self._conn.execute(
            "SELECT scope, owner_id FROM entities WHERE id = ?", (target_id,)
        ).fetchone()
        if src_scope is None or tgt_scope is None:
            raise ValueError("source_id or target_id refers to unknown entity")
        cross_scope = int(
            src_scope["scope"] != tgt_scope["scope"]
            or src_scope["owner_id"] != tgt_scope["owner_id"]
        )

        eid = _new_id()
        now = _now_utc()
        self._conn.execute(
            "INSERT INTO edges ("
            " id, source_id, target_id, relation_type, source_ref,"
            " confidence_score, confidence_class,"
            " valid_from, valid_until, learned_at, unlearned_at, detected_at,"
            " cross_scope, attrs_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?)",
            (
                eid,
                source_id,
                target_id,
                relation_type,
                source_ref,
                confidence_score,
                confidence_class,
                _utc_iso(valid_from),
                _utc_iso(learned_at),
                _utc_iso(now),
                cross_scope,
                json.dumps(attrs or {}),
            ),
        )
        return eid

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            source_ref=row["source_ref"],
            confidence_score=row["confidence_score"],
            confidence_class=row["confidence_class"],
            valid_from=_parse_dt(row["valid_from"]),  # type: ignore[arg-type]
            valid_until=_parse_dt(row["valid_until"]),
            learned_at=_parse_dt(row["learned_at"]),  # type: ignore[arg-type]
            unlearned_at=_parse_dt(row["unlearned_at"]),
            detected_at=_parse_dt(row["detected_at"]),  # type: ignore[arg-type]
            cross_scope=bool(row["cross_scope"]),
            attrs=json.loads(row["attrs_json"]) if row["attrs_json"] else {},
        )

    def edges_from(
        self, entity_id: str, *, relation_type: str | None = None
    ) -> list[Edge]:
        if relation_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? AND relation_type = ?",
                (entity_id, relation_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (entity_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def edges_to(
        self, entity_id: str, *, relation_type: str | None = None
    ) -> list[Edge]:
        if relation_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ? AND relation_type = ?",
                (entity_id, relation_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ?", (entity_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def supersede(
        self,
        *,
        old_id: str,
        new_id: str,
        source_ref: str,
        at: datetime,
    ) -> str:
        # Retire the old row by setting valid_until — but DO NOT delete it.
        self._conn.execute(
            "UPDATE entities SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
            (_utc_iso(at), old_id),
        )
        # Append a SUPERSEDES edge from new -> old (new replaces old).
        return self.upsert_edge(
            source_id=new_id,
            target_id=old_id,
            relation_type="SUPERSEDES",
            source_ref=source_ref,
            confidence_score=1.0,
            valid_from=at,
            learned_at=at,
        )

    # ---- triples (atomic-fact projection) ----

    def upsert_triple(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        edge_id: str,
        fingerprint: str,
        valid_from: datetime,
    ) -> str:
        existing = self._conn.execute(
            "SELECT id FROM triples WHERE subject_id = ? AND predicate = ? "
            "AND object_id = ? AND valid_from = ?",
            (subject_id, predicate, object_id, _utc_iso(valid_from)),
        ).fetchone()
        if existing:
            return existing["id"]
        tid = _new_id()
        self._conn.execute(
            "INSERT INTO triples (id, subject_id, predicate, object_id, edge_id, "
            "fingerprint, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, subject_id, predicate, object_id, edge_id, fingerprint, _utc_iso(valid_from)),
        )
        return tid

    def find_triple_by_fingerprint(self, fingerprint: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM triples WHERE fingerprint = ? LIMIT 1", (fingerprint,)
        ).fetchone()
        return row["id"] if row else None

    # ---- embeddings ----

    # ---- batched inserts (perf-critical bulk-ingest path) ----

    def begin(self) -> None:
        """Open an explicit BEGIN — used by bulk callers to batch many writes."""
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.execute("COMMIT")
        self._touch_write()

    def rollback(self) -> None:
        self._conn.execute("ROLLBACK")

    def store_embedding(
        self,
        *,
        entity_id: str,
        model_name: str,
        model_version: str,
        dim: int,
        vector: bytes,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(entity_id, model_name, model_version, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, model_name, model_version, dim, vector, _utc_iso(_now_utc())),
        )
        # ANN index mirrors (best effort — fallback path still works).
        # vec0 virtual tables don't honour ``INSERT OR REPLACE`` upserts, so
        # we DELETE-then-INSERT to keep the index in sync when an entity is
        # re-embedded.
        if self._vec_available:
            self._ensure_vec_table(dim, kind="float")
            self._conn.execute(
                f"DELETE FROM vec_float_{dim} WHERE entity_id = ?", (entity_id,)
            )
            self._conn.execute(
                f"INSERT INTO vec_float_{dim} (entity_id, embedding) VALUES (?, ?)",
                (entity_id, vector),
            )
            # Binary mirror: sign-pack the fp32 vector → 1 bit per dim, ~32×
            # smaller, Hamming-distance approximates cosine (Charikar 2002).
            self._ensure_vec_table(dim, kind="bit")
            bits = _sign_pack(vector, dim)
            self._conn.execute(
                f"DELETE FROM vec_bit_{dim} WHERE entity_id = ?", (entity_id,)
            )
            self._conn.execute(
                f"INSERT INTO vec_bit_{dim} (entity_id, embedding) "
                f"VALUES (?, vec_bit(?))",
                (entity_id, bits),
            )

    # ---- ANN (sqlite-vec) ----

    def vec_available(self) -> bool:
        return self._vec_available

    def _ensure_vec_table(self, dim: int, *, kind: str) -> None:
        """Create vec_<kind>_<dim> virtual table if not already present.

        ``kind`` is ``"float"`` (fp32 cosine) or ``"bit"`` (1-bit Hamming).
        Idempotent and cheap on subsequent calls (cached in ``_vec_tables``).
        """
        if (dim, kind) in self._vec_tables:
            return
        type_decl = f"FLOAT[{dim}]" if kind == "float" else f"BIT[{dim}]"
        table = f"vec_{kind}_{dim}"
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"entity_id TEXT PRIMARY KEY, embedding {type_decl})"
        )
        self._vec_tables.add((dim, kind))

    def vec_nearest_float(
        self, query_blob: bytes, dim: int, *, k: int
    ) -> list[tuple[str, float]]:
        """KNN cosine over the float vec table. Returns (entity_id, distance)."""
        if not self._vec_available:
            return []
        self._ensure_vec_table(dim, kind="float")
        rows = self._conn.execute(
            f"SELECT entity_id, distance FROM vec_float_{dim} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_blob, k),
        ).fetchall()
        return [(r["entity_id"], float(r["distance"])) for r in rows]

    def vec_nearest_bit(
        self, query_bits: bytes, dim: int, *, k: int
    ) -> list[tuple[str, float]]:
        """KNN Hamming over the bit vec table. Returns (entity_id, distance)."""
        if not self._vec_available:
            return []
        self._ensure_vec_table(dim, kind="bit")
        rows = self._conn.execute(
            f"SELECT entity_id, distance FROM vec_bit_{dim} "
            f"WHERE embedding MATCH vec_bit(?) AND k = ? ORDER BY distance",
            (query_bits, k),
        ).fetchall()
        return [(r["entity_id"], float(r["distance"])) for r in rows]

    def get_embedding(
        self, entity_id: str, *, model_name: str, model_version: str
    ) -> tuple[int, bytes] | None:
        row = self._conn.execute(
            "SELECT dim, vector FROM embeddings WHERE entity_id = ? "
            "AND model_name = ? AND model_version = ?",
            (entity_id, model_name, model_version),
        ).fetchone()
        return (row["dim"], row["vector"]) if row else None

    def iter_embeddings(
        self, *, model_name: str, model_version: str
    ):
        rows = self._conn.execute(
            "SELECT entity_id, dim, vector FROM embeddings "
            "WHERE model_name = ? AND model_version = ?",
            (model_name, model_version),
        ).fetchall()
        for r in rows:
            yield r["entity_id"], r["dim"], r["vector"]
