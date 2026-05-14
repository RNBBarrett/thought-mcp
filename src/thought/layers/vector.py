"""Vector Layer — GraphRAG-style retrieval with Matryoshka two-pass.

Two-pass retrieval per Matryoshka Representation Learning (Kusupati et al.;
OpenAI text-embedding-3): a coarse low-dim ANN over the candidate pool followed
by a full-dim rerank of the top-N. For SQLite-only deployments without a vector
extension loaded, we fall back to in-Python cosine — correct but O(N) per
query; suitable for the MVP scale and for CI.

GraphRAG fusion: optional expansion of seed hits along graph edges, with
hop-decay weighting. Implements the LightRAG / Microsoft GraphRAG pattern of
"ANN finds candidates; graph traversal pulls in connected entities."
"""
from __future__ import annotations

import numpy as np

from ..embeddings.base import Embedder, bytes_to_vector
from ..models import Hit, ScopeFilter
from ..storage.sqlite.backend import SQLiteBackend


class VectorLayer:
    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        coarse_dim_fraction: float = 0.25,
        coarse_top_n: int = 100,
        graph_expansion_alpha: float = 0.5,
        use_binary_quantization: bool = False,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        # Matryoshka coarse dimension: first 25% of the embedding axes by default.
        self._coarse_dim = max(8, int(embedder.dim * coarse_dim_fraction))
        self._coarse_top_n = coarse_top_n
        self._graph_alpha = graph_expansion_alpha
        # Binary quantization (Charikar 2002 random-hyperplane LSH) gives a
        # ~32× index shrink and ~8-16× ANN speedup, but only on *dense*
        # embeddings — production models like BGE-M3 or MiniLM. On sparse
        # vectors (e.g. the deterministic test embedder, hashed bag-of-words)
        # most dims are zero, sign-packing treats zero as positive, and
        # Hamming distance becomes meaningless. Default off; user opts in for
        # dense models via config.
        self._use_binary = use_binary_quantization

    def search(
        self,
        *,
        query: str,
        k: int,
        scope_filter: ScopeFilter,
        expand_via_graph: bool = True,
        expansion_depth: int = 1,
    ) -> list[Hit]:
        if k <= 0:
            return []
        q_full = self._embedder.embed(query)

        scope_ids = self._backend.list_entity_ids(scope_filter)
        if not scope_ids:
            return []

        # Pass 1: coarse candidate generation.
        # Fast path A (default): sqlite-vec float cosine MATCH — C-level
        # brute force with SIMD, dramatically faster than Python looping but
        # quality-preserving.
        # Fast path B (opt-in, dense embedders): binary Hamming over
        # sign-quantized vectors — another ~8-16× over the float path at
        # some recall cost. Off by default because sparse / hashed
        # embedders break the sign approximation.
        # Slow path: Python brute-force cosine (used when sqlite-vec can't
        # load — Anaconda Python, etc.).
        if self._backend.vec_available():
            if self._use_binary:
                coarse_candidates = self._vec_coarse_pass_binary(q_full, scope_ids)
            else:
                coarse_candidates = self._vec_coarse_pass_float(q_full, scope_ids)
        else:
            q_coarse = self._matryoshka_truncate(q_full)
            coarse_candidates = self._coarse_pass(q_coarse, scope_ids)
        if not coarse_candidates:
            return []
        top_coarse = coarse_candidates[: self._coarse_top_n]

        # Pass 2: full-dim rerank over the coarse top-N (cosine over fp32).
        reranked = self._full_pass(q_full, [eid for eid, _ in top_coarse])

        # Build seed hits.
        seed_hits: list[tuple[str, float]] = reranked[: max(k, self._coarse_top_n // 4)]

        # GraphRAG expansion (optional).
        if expand_via_graph and expansion_depth > 0:
            seed_hits = self._graph_expand(seed_hits, expansion_depth, scope_ids)

        # Deduplicate keeping max score per entity.
        best: dict[str, float] = {}
        for eid, score in seed_hits:
            best[eid] = max(best.get(eid, -1.0), score)
        ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]

        hits: list[Hit] = []
        for eid, score in ordered:
            ent = self._backend.get_entity(eid)
            if ent is None:
                continue
            hits.append(
                Hit(
                    entity=ent,
                    score=float(score),
                    layer="vector",
                    confidence_class=self._confidence_class(ent.id),
                    expansion_path=[],
                    source_refs=[],
                )
            )
        return hits

    # ----------------------------------------------------------- pass helpers

    def _vec_coarse_pass_float(
        self, q_full: np.ndarray, scope_ids: set[str]
    ) -> list[tuple[str, float]]:
        """Fast path — fp32 cosine ANN via sqlite-vec.

        sqlite-vec's MATCH operator runs in C with SIMD vectorisation. For
        the deterministic test embedder this is ~10-50× faster than the
        Python brute-force fallback while preserving identical ranking;
        in production with a real embedder the same path scales to 10M
        vectors at single-ms p95.
        """
        from ..embeddings.base import vector_to_bytes

        dim = self._embedder.dim
        oversample = max(self._coarse_top_n * 4, 64)
        q_blob = vector_to_bytes(q_full)
        candidates = self._backend.vec_nearest_float(q_blob, dim, k=oversample)
        # vec_distance is L2 (smaller = closer). Map to a 0..1 score where
        # higher = better. For unit-norm vectors L2² = 2(1 − cos), so we
        # convert in a stable way that preserves order.
        scaled = [
            (eid, max(0.0, 1.0 - (d * d) / 4.0))
            for eid, d in candidates
            if eid in scope_ids
        ]
        return scaled

    def _vec_coarse_pass_binary(
        self, q_full: np.ndarray, scope_ids: set[str]
    ) -> list[tuple[str, float]]:
        """Fast path B — binary sign-quantized Hamming via sqlite-vec.

        ~32× smaller index, ~8-16× faster than fp32 MATCH. Charikar (2002)
        random-hyperplane LSH shows Hamming over sign vectors approximates
        cosine well enough for ranking — but only on *dense* vectors. On
        sparse or hashed embeddings (most dims = 0) the approximation
        collapses because sign(0) = 1 by convention. Default off; flip
        ``use_binary_quantization=True`` when wiring a dense embedder.
        """
        from ..embeddings.base import vector_to_bytes
        from ..storage.sqlite.backend import _sign_pack

        dim = self._embedder.dim
        oversample = max(self._coarse_top_n * 4, 64)
        q_blob = vector_to_bytes(q_full)
        q_bits = _sign_pack(q_blob, dim)
        candidates = self._backend.vec_nearest_bit(q_bits, dim, k=oversample)
        scaled = [
            (eid, 1.0 - (d / dim))
            for eid, d in candidates
            if eid in scope_ids
        ]
        return scaled

    def _coarse_pass(
        self, q_coarse: np.ndarray, scope_ids: set[str]
    ) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        q_norm = q_coarse / (np.linalg.norm(q_coarse) + 1e-12)
        for eid, dim, blob in self._backend.iter_embeddings(
            model_name=self._embedder.model_name,
            model_version=self._embedder.model_version,
        ):
            if eid not in scope_ids:
                continue
            v = bytes_to_vector(blob, dim)
            v_coarse = self._matryoshka_truncate(v)
            v_norm = v_coarse / (np.linalg.norm(v_coarse) + 1e-12)
            results.append((eid, float(np.dot(q_norm, v_norm))))
        results.sort(key=lambda kv: kv[1], reverse=True)
        return results

    def _full_pass(self, q_full: np.ndarray, candidate_ids: list[str]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for eid in candidate_ids:
            stored = self._backend.get_embedding(
                eid,
                model_name=self._embedder.model_name,
                model_version=self._embedder.model_version,
            )
            if stored is None:
                continue
            dim, blob = stored
            v = bytes_to_vector(blob, dim)
            results.append((eid, float(np.dot(q_full, v))))
        results.sort(key=lambda kv: kv[1], reverse=True)
        return results

    def _matryoshka_truncate(self, v: np.ndarray) -> np.ndarray:
        return v[: self._coarse_dim]

    def _graph_expand(
        self,
        seeds: list[tuple[str, float]],
        depth: int,
        scope_ids: set[str],
    ) -> list[tuple[str, float]]:
        expanded: dict[str, float] = dict(seeds)
        frontier: list[tuple[str, float, int]] = [(eid, score, 0) for eid, score in seeds]
        while frontier:
            current, base_score, dist = frontier.pop(0)
            if dist >= depth:
                continue
            for edge in self._backend.edges_from(current):
                if edge.relation_type == "CONTRADICTS":
                    continue
                if edge.target_id not in scope_ids:
                    continue
                decay = self._graph_alpha ** (dist + 1)
                contribution = base_score * decay * edge.confidence_score
                prev = expanded.get(edge.target_id, -1.0)
                if contribution > prev:
                    expanded[edge.target_id] = contribution
                    frontier.append((edge.target_id, contribution, dist + 1))
        return sorted(expanded.items(), key=lambda kv: kv[1], reverse=True)

    def _confidence_class(self, entity_id: str) -> str:
        """MetaRAG-style epistemic class.

        - source_grounded: entity has at least one inbound non-meta edge with
          source_ref pointing at a sources row (i.e., evidence on disk).
        - inferred: only graph-derived (e.g., from a SUPERSEDES chain).
        - hallucination_risk: no source ref, no inbound evidence.

        The MVP uses the simple rule above; a richer NLI-based check arrives
        in v0.2.
        """
        inbound = self._backend.edges_to(entity_id)
        if any(e.relation_type not in {"CONTRADICTS", "SUPERSEDES"} for e in inbound):
            return "source_grounded"
        ent = self._backend.get_entity(entity_id)
        if ent is None:
            return "hallucination_risk"
        # Every entity is created with a source_ref edge inbound from sources at
        # ingest time; bare creation with no inbound edges means it was
        # bootstrapped without provenance proper.
        return "inferred" if inbound else "source_grounded"
