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
    ) -> str: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def list_entities(self, scope_filter: ScopeFilter) -> list[Entity]: ...

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
