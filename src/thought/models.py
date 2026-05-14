"""Public Pydantic DTOs and core enums for THOUGHT.

Kept deliberately minimal — the storage layer uses dataclasses internally; this
module is what crosses the MCP boundary and is exported to callers.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ScopeName = Literal["shared", "private"]
ScopeQuery = Literal["shared", "private", "all"]
ConfidenceClass = Literal["source_grounded", "inferred", "hallucination_risk"]
Tier = Literal["hot", "warm", "cold"]


class QueryClass(StrEnum):
    VIBE = "VIBE"
    FACT = "FACT"
    CHANGE = "CHANGE"
    HYBRID = "HYBRID"
    CODE = "CODE"


class ScopeFilter(BaseModel):
    """Storage-layer scope predicate. Composed into every query.

    A query for 'all' returns shared facts plus the requester's private facts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ScopeQuery = "all"
    owner_id: str | None = None

    def sql_where(self) -> tuple[str, list[object]]:
        """Return a SQL WHERE fragment and its parameters.

        The fragment references ``e.scope`` / ``e.owner_id`` — callers alias the
        entity table as ``e``.
        """
        if self.scope == "shared":
            return "e.scope = ?", ["shared"]
        if self.scope == "private":
            if self.owner_id is None:
                # private with no owner: empty set, not all-private
                return "1 = 0", []
            return "e.scope = ? AND e.owner_id = ?", ["private", self.owner_id]
        # 'all'
        if self.owner_id is None:
            return "e.scope = ?", ["shared"]
        return (
            "(e.scope = ? OR (e.scope = ? AND e.owner_id = ?))",
            ["shared", "private", self.owner_id],
        )


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    name: str
    canonical_name: str
    owner_id: str | None = None
    scope: ScopeName
    tier: Tier
    importance: float = Field(ge=0.0, le=1.0)
    valid_from: datetime
    valid_until: datetime | None = None
    learned_at: datetime
    unlearned_at: datetime | None = None
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    attrs: dict[str, object] = Field(default_factory=dict)


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    target_id: str
    relation_type: str
    source_ref: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_class: ConfidenceClass = "source_grounded"
    valid_from: datetime
    valid_until: datetime | None = None
    learned_at: datetime
    unlearned_at: datetime | None = None
    detected_at: datetime
    cross_scope: bool = False
    attrs: dict[str, object] = Field(default_factory=dict)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content_hash: str
    ingested_at: datetime


class Hit(BaseModel):
    """One result returned by ``recall``.

    Carries provenance: which layer it came from, the source reference, and
    its epistemic confidence class.
    """

    model_config = ConfigDict(extra="forbid")

    entity: Entity
    score: float
    layer: Literal["vector", "graph", "temporal"]
    confidence_class: ConfidenceClass = "source_grounded"
    expansion_path: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class ContradictionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_a: str
    entity_b: str
    edge_id: str
    detected_at: datetime


class RememberInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=64_000)
    source_ref: str | None = None
    scope: ScopeName = "private"
    owner_id: str | None = None


class RememberResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    duplicate_of_source: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    embeddings_created: int = 0
    contradictions_detected: list[ContradictionRef] = Field(default_factory=list)


class RecallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4_000)
    limit: int = Field(default=10, ge=1, le=10)
    scope: ScopeQuery = "all"
    owner_id: str | None = None
    as_of: datetime | None = None
    as_of_kind: Literal["valid", "learned"] = "valid"


class RecallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[Hit] = Field(default_factory=list)
    query_class: QueryClass
    sources: list[SourceRef] = Field(default_factory=list)
    elapsed_ms: float
    low_confidence: bool = False
    signals: dict[str, object] = Field(default_factory=dict)
