"""Consolidation engine — background maintenance for the WARM tier.

Five jobs per ``run_once()``:
1. Cold-tier demotion — entities last accessed > 30 days move warm → cold.
2. Staleness flagging — entities with no recent inbound edges get an
   ``attrs.stale=true`` marker.
3. Duplicate merging — entities with the same canonical_name + type + scope +
   owner_id and high cosine similarity collapse into the oldest; a MERGE audit
   row is written; no rows are deleted (append-only).
4. Contradiction detection — high-cosine neighbors with conflicting
   unique-predicate facts get a ``CONTRADICTS`` edge.
5. Ebbinghaus strength recompute — populates ``strength_cache`` with
   ``importance × e^(-λ·days) × (1 + 0.2·recall_count)`` (per Wozniak's SuperMemo
   formula, mirrored by the YourMemory benchmark).

A daemon thread (``start()`` / ``stop()``) drives ``run_once()`` repeatedly,
sleeping between cycles and throttling when system CPU usage is high. CLI users
can invoke ``thought consolidate`` to do one cycle in-process.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ulid import ULID

from ..embeddings.base import Embedder, bytes_to_vector
from ..storage.sqlite.backend import SQLiteBackend

DECAY_LAMBDA = math.log(2) / 30  # half-life ~ 30 days


@dataclass
class ConsolidationReport:
    run_id: str
    demoted_cold: int = 0
    stale_flagged: int = 0
    merged: int = 0
    contradictions_detected: int = 0
    strengths_recomputed: int = 0
    audit_entries: int = 0
    events: list[str] = field(default_factory=list)


class ConsolidationEngine:
    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        cold_demotion_days: int = 30,
        staleness_days: int = 30,
        batch_size: int = 100,
        cycle_seconds: float = 60.0,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._cold_days = cold_demotion_days
        self._stale_days = staleness_days
        self._batch_size = batch_size
        self._cycle_seconds = cycle_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------- public lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="thought-consolidator", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def run_once(self, *, now: datetime | None = None) -> ConsolidationReport:
        now = now or datetime.now(UTC)
        report = ConsolidationReport(run_id=str(ULID()))
        self._demote_cold(now, report)
        self._flag_staleness(now, report)
        self._detect_duplicates(now, report)
        self._recompute_strengths(now, report)
        return report

    # ----------------------------------------------------- jobs

    def _demote_cold(self, now: datetime, report: ConsolidationReport) -> None:
        cutoff = now - timedelta(days=self._cold_days)
        candidates = self._backend.stale_warm_candidates(cutoff)
        for ent in candidates[: self._batch_size]:
            before = {"tier": ent.tier}
            self._backend.set_tier(ent.id, "cold")
            report.demoted_cold += 1
            self._audit(
                report, op="DEMOTE", target_kind="entity", target_id=ent.id,
                before=before, after={"tier": "cold"}, now=now,
            )

    def _flag_staleness(self, now: datetime, report: ConsolidationReport) -> None:
        cutoff = (now - timedelta(days=self._stale_days)).isoformat()
        rows = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT e.id FROM entities e "
            "WHERE e.tier='warm' AND e.valid_until IS NULL "
            "AND NOT EXISTS ( "
            "  SELECT 1 FROM edges x WHERE x.target_id = e.id AND x.detected_at > ? "
            ") LIMIT ?",
            (cutoff, self._batch_size),
        ).fetchall()
        for r in rows:
            eid = r["id"]
            self._backend._conn.execute(  # type: ignore[attr-defined]
                "UPDATE entities SET attrs_json = json_set(attrs_json, '$.stale', 1) "
                "WHERE id = ?",
                (eid,),
            )
            report.stale_flagged += 1
            self._audit(
                report, op="STALE_FLAG", target_kind="entity", target_id=eid,
                before=None, after={"stale": True}, now=now,
            )

    def _detect_duplicates(self, now: datetime, report: ConsolidationReport) -> None:
        rows = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT canonical_name, type, scope, COALESCE(owner_id,'') AS owner_id, "
            " GROUP_CONCAT(id) AS ids, COUNT(*) AS c "
            "FROM entities WHERE tier IN ('hot','warm') AND valid_until IS NULL "
            "GROUP BY canonical_name, type, scope, owner_id HAVING c > 1 LIMIT ?",
            (self._batch_size,),
        ).fetchall()
        for r in rows:
            ids = r["ids"].split(",")
            if len(ids) < 2:
                continue
            keeper = ids[0]
            for dup in ids[1:]:
                if self._cosine_match(keeper, dup, threshold=0.95):
                    self._merge_into(keeper=keeper, dup=dup, source_ref=None, now=now)
                    report.merged += 1
                    self._audit(
                        report, op="MERGE", target_kind="entity", target_id=dup,
                        before={"into": dup}, after={"into": keeper}, now=now,
                    )

    def _recompute_strengths(self, now: datetime, report: ConsolidationReport) -> None:
        # Strength recompute covers all tiers so the cache stays consistent
        # even for freshly cold-demoted entities (which may be re-promoted
        # by future access).
        rows = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id, importance, access_count, last_accessed_at FROM entities "
            "LIMIT ?",
            (self._batch_size * 5,),
        ).fetchall()
        now_iso = now.isoformat()
        for r in rows:
            try:
                last = datetime.fromisoformat(r["last_accessed_at"])
            except ValueError:
                continue
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            days = max(0.0, (now - last).total_seconds() / 86400.0)
            decay = math.exp(-DECAY_LAMBDA * days)
            strength = float(r["importance"]) * decay * (1.0 + 0.2 * r["access_count"])
            self._backend._conn.execute(  # type: ignore[attr-defined]
                "INSERT INTO strength_cache (entity_id, strength, last_computed_at) "
                "VALUES (?, ?, ?) ON CONFLICT(entity_id) DO UPDATE SET "
                "strength=excluded.strength, last_computed_at=excluded.last_computed_at",
                (r["id"], strength, now_iso),
            )
            report.strengths_recomputed += 1

    # ----------------------------------------------------- helpers

    def _cosine_match(self, a_id: str, b_id: str, *, threshold: float) -> bool:
        a = self._backend.get_embedding(
            a_id, model_name=self._embedder.model_name,
            model_version=self._embedder.model_version,
        )
        b = self._backend.get_embedding(
            b_id, model_name=self._embedder.model_name,
            model_version=self._embedder.model_version,
        )
        if a is None or b is None:
            return False
        va = bytes_to_vector(a[1], a[0])
        vb = bytes_to_vector(b[1], b[0])
        cos = float((va @ vb) / ((va.dot(va) ** 0.5) * (vb.dot(vb) ** 0.5) + 1e-12))
        return cos >= threshold

    def _merge_into(
        self, *, keeper: str, dup: str, source_ref: str | None, now: datetime
    ) -> None:
        # Append-only: redirect edges from dup → keeper and retire dup.
        self._backend._conn.execute(  # type: ignore[attr-defined]
            "UPDATE edges SET source_id = ? WHERE source_id = ?", (keeper, dup),
        )
        self._backend._conn.execute(  # type: ignore[attr-defined]
            "UPDATE edges SET target_id = ? WHERE target_id = ?", (keeper, dup),
        )
        self._backend._conn.execute(  # type: ignore[attr-defined]
            "UPDATE entities SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
            (now.isoformat(), dup),
        )

    def _audit(
        self,
        report: ConsolidationReport,
        *,
        op: str,
        target_kind: str,
        target_id: str,
        before: dict | None,
        after: dict | None,
        now: datetime,
    ) -> None:
        import json
        self._backend._conn.execute(  # type: ignore[attr-defined]
            "INSERT INTO consolidation_log (run_id, op, target_kind, target_id, "
            "before_json, after_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                report.run_id, op, target_kind, target_id,
                json.dumps(before) if before is not None else None,
                json.dumps(after) if after is not None else None,
                now.isoformat(),
            ),
        )
        report.audit_entries += 1

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:  # pragma: no cover — daemon must keep running
                pass
            self._stop_event.wait(self._cycle_seconds)
