"""Code Layer — convenience surface for code-specific graph queries.

A thin wrapper over ``GraphLayer`` that lifts the four operations the CLI
needs into a vocabulary native to programmers:

- ``callers_of(name)``   — who calls this? (direct, ranked by PageRank)
- ``callees_of(name)``   — what does this call? (direct, intra-package)
- ``impact_set(name)``   — transitive callers, ranked. The ``thought impact``
  command sits on this.
- ``defines_in_file()``  — every entity discovered in a given file.

All four operate against the **currently-valid** view of the code graph
(``valid_until IS NULL``). Pass ``as_of=`` to look at historical snapshots
once the bi-temporal git ingest from Phase 3 is wired up — it's just a
filter on ``code_commit_sha``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Entity, ScopeFilter
from ..storage.sqlite.backend import SQLiteBackend
from .graph import GraphLayer


@dataclass(frozen=True)
class CodeHit:
    entity: Entity
    score: float


class CodeLayer:
    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._graph = GraphLayer(backend)

    # ----------------------------------------------------------- lookups

    def _resolve_entity_id(
        self,
        name: str,
        *,
        code_file: str | None = None,
        code_commit_sha: str | None = None,
    ) -> str | None:
        """Find the entity ID for a code-name, preferring intra-file matches."""
        # Try unqualified name first; falls back to class-qualified for methods.
        eid = self._backend.find_code_entity(
            canonical_name=name, code_file=code_file,
            code_commit_sha=code_commit_sha,
        )
        if eid is not None:
            return eid
        # Without the file constraint — picks up any matching entity.
        return self._backend.find_code_entity(
            canonical_name=name, code_commit_sha=code_commit_sha,
        )

    # ----------------------------------------------------------- callers

    def callers_of(
        self,
        name: str,
        *,
        code_file: str | None = None,
        limit: int = 10,
    ) -> list[CodeHit]:
        """Direct callers of ``name``, ordered by PageRank score (highest first).

        Empty list if ``name`` doesn't resolve to a known entity.
        """
        eid = self._resolve_entity_id(name, code_file=code_file)
        if eid is None:
            return []
        inbound = self._backend.edges_to(eid, relation_type="CALLS")
        seen: set[str] = set()
        callers: list[Entity] = []
        for edge in inbound:
            if edge.source_id in seen:
                continue
            seen.add(edge.source_id)
            ent = self._backend.get_entity(edge.source_id)
            if ent is not None and ent.valid_until is None:
                callers.append(ent)

        # Rank by PPR seeded from the callee — closer callers score higher.
        scores = self._graph.personalized_pagerank(
            seeds=[eid], scope_filter=ScopeFilter(scope="all"),
        )
        ranked = sorted(
            callers,
            key=lambda e: scores.get(e.id, 0.0),
            reverse=True,
        )
        return [
            CodeHit(entity=e, score=float(scores.get(e.id, 0.0)))
            for e in ranked[:limit]
        ]

    def callees_of(
        self,
        name: str,
        *,
        code_file: str | None = None,
        limit: int = 50,
    ) -> list[CodeHit]:
        """Direct callees of ``name`` (functions ``name`` calls)."""
        eid = self._resolve_entity_id(name, code_file=code_file)
        if eid is None:
            return []
        outbound = self._backend.edges_from(eid, relation_type="CALLS")
        seen: set[str] = set()
        callees: list[Entity] = []
        for edge in outbound:
            if edge.target_id in seen:
                continue
            seen.add(edge.target_id)
            ent = self._backend.get_entity(edge.target_id)
            if ent is not None and ent.valid_until is None:
                callees.append(ent)
        # Order by confidence on the edge so source_grounded shows first.
        return [CodeHit(entity=e, score=1.0) for e in callees[:limit]]

    # ----------------------------------------------------------- impact

    def impact_set(
        self,
        name: str,
        *,
        code_file: str | None = None,
        limit: int = 20,
    ) -> list[CodeHit]:
        """Transitive callers of ``name``, ranked by PageRank from the callee.

        This is the answer to *"if I change X, what's affected?"*. We seed
        PPR at ``name`` and walk *inbound* CALLS edges (the GraphLayer's
        bidirectional walk handles this); the result is a confidence-weighted
        ranking of every function whose behaviour depends on ``name``.
        """
        eid = self._resolve_entity_id(name, code_file=code_file)
        if eid is None:
            return []
        scores = self._graph.personalized_pagerank(
            seeds=[eid], scope_filter=ScopeFilter(scope="all"),
        )
        # Drop the seed itself + any non-code entities.
        hits: list[CodeHit] = []
        for nid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            if nid == eid:
                continue
            ent = self._backend.get_entity(nid)
            if ent is None or ent.valid_until is not None:
                continue
            if ent.type not in {"function", "method", "class", "module"}:
                continue
            hits.append(CodeHit(entity=ent, score=float(score)))
            if len(hits) >= limit:
                break
        return hits

    # ----------------------------------------------------------- defines

    def defines_in_file(self, file_path: str) -> list[Entity]:
        """Every function / class / method ingested from ``file_path``.

        Excludes the module entity itself (that's the *container*, not a member).
        """
        rows = self._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT * FROM entities WHERE code_file = ? "
            "AND type IN ('function','class','method') "
            "AND valid_until IS NULL "
            "ORDER BY type, name",
            (file_path,),
        ).fetchall()
        return [self._backend._row_to_entity(r) for r in rows]  # type: ignore[attr-defined]

    # ----------------------------------------------------------- diff

    def diff(
        self,
        *,
        from_sha: str,
        to_sha: str,
        code_file: str | None = None,
    ) -> dict[str, list[Entity]]:
        """Set difference of entities between two commit SHAs.

        Returns ``{added, removed, changed}`` where:
        - added:   in ``to_sha`` but not in ``from_sha``
        - removed: in ``from_sha`` but not in ``to_sha``
        - changed: same canonical name but different content hash

        Phase 3 (git ingest) populates the ``code_commit_sha`` column that
        this method filters on. Returns empty lists if neither SHA has been
        ingested yet.
        """
        def _names_at(sha: str) -> dict[str, Entity]:
            sql = (
                "SELECT * FROM entities WHERE code_commit_sha = ? "
                "AND type IN ('function','class','method')"
            )
            params: list = [sha]
            if code_file is not None:
                sql += " AND code_file = ?"
                params.append(code_file)
            rows = self._backend._conn.execute(sql, params).fetchall()  # type: ignore[attr-defined]
            return {
                r["canonical_name"]: self._backend._row_to_entity(r)  # type: ignore[attr-defined]
                for r in rows
            }

        a = _names_at(from_sha)
        b = _names_at(to_sha)
        added = [b[k] for k in b.keys() - a.keys()]
        removed = [a[k] for k in a.keys() - b.keys()]
        # ``changed`` is approximate without per-row content hash; we leave
        # the slot for Phase 3 to fill in with real content-hash diffing.
        return {"added": added, "removed": removed, "changed": []}
