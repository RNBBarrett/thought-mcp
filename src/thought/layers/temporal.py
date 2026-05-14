"""Temporal Layer — bi-temporal lifecycle management.

Tracks two distinct time axes per fact:
- valid_time (``valid_from`` / ``valid_until``): when the fact was true in the
  world.
- transaction_time (``learned_at`` / ``unlearned_at``): when the system became
  aware of the fact.

Models the Zep / Graphiti bi-temporal pattern (arXiv 2501.13956). Both axes are
queryable independently via ``entities_valid_at(when, kind=...)``.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..models import Entity, ScopeFilter, Tier
from ..storage.sqlite.backend import SQLiteBackend


class TemporalLayer:
    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    def entities_valid_at(
        self,
        when: datetime,
        *,
        scope_filter: ScopeFilter,
        kind: str = "valid",
    ) -> list[Entity]:
        return self._backend.list_entities_at(when, scope_filter, kind=kind)

    def set_tier(self, entity_id: str, tier: Tier) -> None:
        self._backend.set_tier(entity_id, tier)

    def stale_warm_candidates(
        self, *, older_than: timedelta, now: datetime | None = None
    ) -> list[Entity]:
        now = now or datetime.now(UTC)
        return self._backend.stale_warm_candidates(now - older_than)
