"""Dispatcher — routes a classified query to the right layer(s), merges, and
applies the CRAG-style retrieval evaluator.

Per-class strategy:
- VIBE:   Vector layer with graph expansion enabled (Matryoshka 2-pass + GraphRAG).
- FACT:   Graph layer using Personalized PageRank (HippoRAG) seeded by entity hits.
- CHANGE: Temporal layer scoping all reads to the bi-temporal `as_of` window.
- HYBRID: All three in parallel; results merged, deduped, reranked, bounded.

The dispatcher returns at most ``limit ≤ 10`` hits (plan-mandated bound).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Literal

from ..embeddings.base import Embedder
from ..layers.graph import GraphLayer
from ..layers.temporal import TemporalLayer
from ..layers.vector import VectorLayer
from ..models import Hit, QueryClass, RecallResult, ScopeFilter, SourceRef
from ..storage.sqlite.backend import SQLiteBackend
from .classifier import RuleBasedClassifier


class Dispatcher:
    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        classifier: RuleBasedClassifier,
        crag_threshold: float = 0.2,
    ) -> None:
        self._backend = backend
        self._classifier = classifier
        self._vector = VectorLayer(backend=backend, embedder=embedder)
        self._graph = GraphLayer(backend)
        self._temporal = TemporalLayer(backend)
        self._crag_threshold = crag_threshold

    def recall(
        self,
        *,
        query: str,
        limit: int,
        scope_filter: ScopeFilter,
        as_of: datetime | None = None,
        as_of_kind: Literal["valid", "learned"] = "valid",
        touch_queue: list[str] | None = None,
    ) -> RecallResult:
        start = time.perf_counter()
        cls, signals = self._classifier.classify(query)

        # Layer routing.
        if cls == QueryClass.VIBE:
            hits = self._dispatch_vibe(query, limit, scope_filter)
        elif cls == QueryClass.FACT:
            hits = self._dispatch_fact(query, limit, scope_filter)
        elif cls == QueryClass.CHANGE:
            hits = self._dispatch_change(
                query, limit, scope_filter, as_of=as_of, as_of_kind=as_of_kind,
            )
        elif cls == QueryClass.CODE:
            hits = self._dispatch_code(query, limit, scope_filter)
        else:
            hits = self._dispatch_hybrid(
                query, limit, scope_filter, as_of=as_of, as_of_kind=as_of_kind,
            )

        # CRAG evaluator — low confidence if top score < threshold.
        low_confidence = False
        if hits:
            top = max(h.score for h in hits)
            if top < self._crag_threshold:
                low_confidence = True
        else:
            low_confidence = True

        # Touch access for returned entities (Ebbinghaus strength input).
        # Defer to a Memory-owned flush queue when one is provided so the
        # UPDATE writes don't sit on the recall hot path.
        if touch_queue is not None:
            touch_queue.extend(h.entity.id for h in hits)
        else:
            for h in hits:
                self._backend.touch_access(h.entity.id)

        sources = self._collect_sources(hits)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        bounded = hits[: min(limit, 10)]
        return RecallResult(
            hits=bounded,
            query_class=cls,
            sources=sources,
            elapsed_ms=elapsed_ms,
            low_confidence=low_confidence,
            signals=dict(signals),
        )

    # ----------------------------------------------------------- per-class

    def _dispatch_vibe(
        self, query: str, limit: int, scope_filter: ScopeFilter
    ) -> list[Hit]:
        return self._vector.search(
            query=query, k=limit, scope_filter=scope_filter,
            expand_via_graph=True, expansion_depth=1,
        )

    def _dispatch_fact(
        self, query: str, limit: int, scope_filter: ScopeFilter
    ) -> list[Hit]:
        # Use vector hits as seeds for PageRank — names mentioned in the query
        # are the most reliable seeds.
        seed_hits = self._vector.search(
            query=query, k=5, scope_filter=scope_filter, expand_via_graph=False,
        )
        if not seed_hits:
            return []
        # Filter out near-zero-score seeds: when the embedder finds only one
        # genuinely-matching entity in the KB, the remaining k-1 hits are
        # whatever sqlite-vec returned at distance ~max. Feeding them to PPR
        # as equal-weight seeds dilutes the personalization and pulls mass
        # toward unrelated central hubs.
        top_score = seed_hits[0].score
        threshold = max(0.05, top_score * 0.3)
        seed_ids = [h.entity.id for h in seed_hits if h.score >= threshold]
        if not seed_ids:
            seed_ids = [seed_hits[0].entity.id]
        # Choose global vs local PPR based on KB size. Above ~5k entities the
        # Andersen-Chung-Lang push variant is dramatically faster while
        # producing the same top-k ranking on the relevant subgraph.
        kb_size = len(self._backend.list_entity_ids(scope_filter))
        if kb_size >= 5000:
            scores = self._graph.local_personalized_pagerank(
                seeds=seed_ids, scope_filter=scope_filter, epsilon=1e-3,
            )
        else:
            scores = self._graph.personalized_pagerank(
                seeds=seed_ids, scope_filter=scope_filter,
            )
        ranked_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[Hit] = []
        for eid, score in ranked_ids[:limit]:
            ent = self._backend.get_entity(eid)
            if ent is None:
                continue
            out.append(Hit(
                entity=ent,
                score=float(score),
                layer="graph",
                confidence_class="source_grounded",
                expansion_path=[],
                source_refs=[],
            ))
        return out

    def _dispatch_change(
        self,
        query: str,
        limit: int,
        scope_filter: ScopeFilter,
        *,
        as_of: datetime | None,
        as_of_kind: Literal["valid", "learned"],
    ) -> list[Hit]:
        when = as_of or datetime.now(__import__("datetime").timezone.utc)
        entities = self._temporal.entities_valid_at(
            when, scope_filter=scope_filter, kind=as_of_kind,
        )
        # Index entities by canonical_name for token-lookup.
        entities_by_id = {e.id: e for e in entities}
        toks_q = {t.lower() for t in query.split() if t}

        # Score every temporally-valid entity by lexical overlap.
        scored: list[tuple[Hit, float]] = []
        for ent in entities:
            ent_toks = set(ent.canonical_name.split())
            inter = len(toks_q & ent_toks)
            score = (inter + 1) / (len(toks_q) + 1)
            scored.append((
                Hit(entity=ent, score=score, layer="temporal",
                    confidence_class="source_grounded",
                    expansion_path=[], source_refs=[]),
                score,
            ))

        # Also pull in the *objects* of edges anchored at lexically-matching
        # entities, BUT only edges whose validity window covers `when`. This
        # is how a CHANGE query like "what does Kendra currently prefer" gets
        # the right object (Nike at t=now, Adidas at t=-200d): we resolve the
        # subject lexically, then traverse the PREFERS edge that was valid at
        # the requested time.
        for ent in entities:
            ent_toks = set(ent.canonical_name.split())
            if not (toks_q & ent_toks):
                continue
            for edge in self._backend.edges_from(ent.id):
                if edge.relation_type == "CONTRADICTS":
                    continue
                if edge.valid_from > when:
                    continue
                if edge.valid_until is not None and edge.valid_until <= when:
                    continue
                obj = entities_by_id.get(edge.target_id) or self._backend.get_entity(edge.target_id)
                if obj is None:
                    continue
                # Object's own validity must also cover `when`.
                if obj.valid_from > when:
                    continue
                if obj.valid_until is not None and obj.valid_until <= when:
                    continue
                scored.append((
                    Hit(entity=obj, score=0.95 * edge.confidence_score,
                        layer="temporal", confidence_class="source_grounded",
                        expansion_path=[edge.id], source_refs=[edge.source_ref]),
                    0.95 * edge.confidence_score,
                ))

        # Dedup by entity.id keeping highest score.
        best: dict[str, Hit] = {}
        for hit, _ in scored:
            prev = best.get(hit.entity.id)
            if prev is None or hit.score > prev.score:
                best[hit.entity.id] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]

    def _dispatch_code(
        self, query: str, limit: int, scope_filter: ScopeFilter,
    ) -> list[Hit]:
        """CODE-class dispatch.

        Three-step:
        1. Resolve any explicit names in the query (e.g. ``authenticate_user``)
           to entity IDs via the storage layer.
        2. If we have at least one seed, run PageRank to surface
           callers + structural dependents.
        3. Fall back to vector search over signature/docstring embeddings
           when no explicit name resolves.
        """
        # Extract candidate identifiers from the query — anything that looks
        # like a snake_case or CamelCase name.
        import re
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query)
        # Filter out stopwords / common verbs.
        stop = {
            "what", "who", "where", "which", "how", "the", "a", "an", "is",
            "are", "in", "on", "of", "for", "to", "from", "and", "or",
            "callers", "calls", "callees", "impact", "function", "class",
            "method", "module", "import",
        }
        candidates = [c for c in candidates if c.lower() not in stop and len(c) >= 3]

        seed_ids: list[str] = []
        for name in candidates:
            eid = self._backend.find_code_entity(canonical_name=name)
            if eid is not None:
                seed_ids.append(eid)

        if seed_ids:
            scores = self._graph.personalized_pagerank(
                seeds=seed_ids, scope_filter=scope_filter,
            )
            out: list[Hit] = []
            for eid, score in sorted(
                scores.items(), key=lambda kv: kv[1], reverse=True
            ):
                ent = self._backend.get_entity(eid)
                if ent is None or ent.valid_until is not None:
                    continue
                if ent.type not in {"function", "method", "class", "module"}:
                    continue
                out.append(Hit(
                    entity=ent, score=float(score), layer="graph",
                    confidence_class="source_grounded",
                    expansion_path=[], source_refs=[],
                ))
                if len(out) >= limit:
                    break
            if out:
                return out

        # Vector fallback — semantic search over the code embeddings.
        return self._dispatch_vibe(query, limit, scope_filter)

    def _dispatch_hybrid(
        self,
        query: str,
        limit: int,
        scope_filter: ScopeFilter,
        *,
        as_of: datetime | None,
        as_of_kind: Literal["valid", "learned"],
    ) -> list[Hit]:
        vibe_hits = self._dispatch_vibe(query, limit, scope_filter)
        fact_hits = self._dispatch_fact(query, limit, scope_filter)
        change_hits = self._dispatch_change(
            query, limit, scope_filter, as_of=as_of, as_of_kind=as_of_kind,
        )
        # Merge, take max score per entity, keep best layer label.
        best: dict[str, Hit] = {}
        for h in (*vibe_hits, *fact_hits, *change_hits):
            prev = best.get(h.entity.id)
            if prev is None or h.score > prev.score:
                best[h.entity.id] = h
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ranked[:limit]

    def _collect_sources(self, hits: list[Hit]) -> list[SourceRef]:
        if not hits:
            return []
        rows = self._backend.sources_for_entities([h.entity.id for h in hits])
        return [
            SourceRef(
                id=r["id"],
                content_hash=r["content_hash"],
                ingested_at=datetime.fromisoformat(r["ingested_at"]),
            )
            for r in rows
        ]
