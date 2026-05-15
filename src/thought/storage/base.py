"""Abstract storage backend interface.

The MVP ships a SQLite implementation; a Postgres adapter mirrors the same
contract. Layers (graph / vector / temporal) speak to this interface only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Edge, Entity, ScopeFilter, ScopeName


class StorageBackend(ABC):
    @abstractmethod
    def migrate(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def schema_version(self) -> int: ...

    @abstractmethod
    def upsert_source(
        self, content: str, *, mime_type: str = "text/plain",
        context_summary: str | None = None,
    ) -> str: ...

    @abstractmethod
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
    ) -> str: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def list_entities(self, scope_filter: ScopeFilter) -> list[Entity]: ...

    @abstractmethod
    def count_by_type(self, scope_filter: ScopeFilter) -> dict[str, int]:
        """Return ``{type: count}`` of currently-valid entities in scope.

        Powers ``thought topics`` / ``mcp__thought__list_topics``.
        """
        ...

    @abstractmethod
    def find_anchor_by_name(
        self, name: str, scope_filter: ScopeFilter,
    ) -> Entity | None:
        """Look up a single anchor entity by name (canonical-matched).

        Returns the highest-access-count entity matching the name within scope,
        or ``None`` if no match. Used by ``browse_topic`` to resolve a string
        topic into a graph seed.
        """
        ...

    @abstractmethod
    def touch_access(self, entity_id: str) -> None: ...

    @abstractmethod
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
    ) -> str: ...

    @abstractmethod
    def edges_from(self, entity_id: str, *, relation_type: str | None = None) -> list[Edge]: ...

    @abstractmethod
    def edges_to(self, entity_id: str, *, relation_type: str | None = None) -> list[Edge]: ...

    @abstractmethod
    def supersede(
        self,
        *,
        old_id: str,
        new_id: str,
        source_ref: str,
        at: datetime,
    ) -> str: ...

    # ---- v0.4 db lifecycle ----

    @abstractmethod
    def file_sizes(self) -> dict[str, int]:
        """On-disk size in bytes for main / wal / shm + total. 0 for missing sidecars."""
        ...

    @abstractmethod
    def checkpoint_wal(self) -> None:
        """Force a WAL TRUNCATE checkpoint. Used before backups."""
        ...

    @abstractmethod
    def flush(
        self,
        *,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: str = "created",
    ) -> dict[str, int]:
        """Wipe data from the KB. Destructive.

        Full flush (no date bounds) drops + recreates user tables. Date-bounded
        flush deletes entities whose chosen timestamp falls outside the kept
        range; cascades clean up edges / triples / embeddings.
        """
        ...

    @abstractmethod
    def backup_to(
        self,
        target_path,
        *,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: str = "created",
    ) -> int:
        """Snapshot the DB to ``target_path``. Returns bytes written."""
        ...

    @abstractmethod
    def merge_from(
        self,
        source_path,
        *,
        before: datetime | None = None,
        since: datetime | None = None,
        time_axis: str = "created",
    ) -> dict[str, int]:
        """Merge another DB file into this one. Returns {new_*: count} by table."""
        ...
