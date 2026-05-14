"""Ablation harness — quantify each Tier A frontier technique's contribution.

Runs THOUGHT through the comparison workload with each named feature toggled
off (one at a time), so the README can show the marginal effect of each
addition. The baseline is the full system; each ablated run reports the delta
in CHANGE correctness, HYBRID correctness, and overall recall@10.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thought.memory import Memory

from .run import _hit_correct, _names_from_thought
from .workload import build_workload, case_as_of, now_anchored, step_now


@dataclass
class AblationRow:
    name: str
    overall: float
    fact: float
    change: float
    hybrid: float
    contradictions_detected: int


def _eval_memory(mem: Memory, workload, base) -> AblationRow:
    correct = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
    total = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
    contradictions = 0
    for step in workload.ingests:
        r = mem.remember(
            content=step.content, scope=step.scope, owner_id=step.owner_id,
            now=step_now(base, step),
            unique_predicates=set(step.unique_predicates) or None,
        )
        contradictions += len(r.contradictions_detected)
    for case in workload.recalls:
        total[case.kind] += 1
        result = mem.recall(
            query=case.query, limit=10, scope=case.scope, owner_id=case.owner_id,
            as_of=case_as_of(base, case),
        )
        returned = _names_from_thought(result.hits)
        if _hit_correct(returned, case.ground_truth, case.forbidden):
            correct[case.kind] += 1

    def pct(c, t):
        return c / t if t else 0.0
    overall = sum(correct.values()) / sum(total.values())
    return AblationRow(
        name="placeholder",
        overall=overall,
        fact=pct(correct["FACT"], total["FACT"]),
        change=pct(correct["CHANGE"], total["CHANGE"]),
        hybrid=pct(correct["HYBRID"], total["HYBRID"]),
        contradictions_detected=contradictions,
    )


def _open(db: str, **kw) -> Memory:
    return Memory.open(db_path=db, embedder_choice="deterministic", embedder_dim=128, **kw)


def run(out_path: str = "docs/ablation.md") -> None:
    workload = build_workload()
    base = now_anchored()
    rows: list[AblationRow] = []

    # Baseline: full THOUGHT.
    db = ".abl_full.db"
    Path(db).unlink(missing_ok=True)
    full = _open(db)
    r = _eval_memory(full, workload, base)
    r.name = "Full v0.1 (all Tier A)"
    rows.append(r)
    full.close()
    Path(db).unlink(missing_ok=True)

    # Ablation 1: disable bidirectional PPR (HippoRAG #2).
    # We re-create a Memory with a monkey-patched graph layer using only
    # forward edges — exercising "without HippoRAG bidirectional walks."
    from thought.layers import graph as graph_module
    original_ppr = graph_module.GraphLayer.personalized_pagerank

    def forward_only_ppr(self, *, seeds, scope_filter=None, damping=0.85,
                        max_iter=30, tolerance=1e-6):
        """Forward-only PageRank — what we'd have without HippoRAG."""
        from collections import defaultdict, deque

        from thought.models import ScopeFilter
        scope_filter = scope_filter or ScopeFilter(scope="all")
        allowed = {e.id for e in self._backend.list_entities(scope_filter)}
        seed_list = [s for s in seeds if s in allowed]
        if not seed_list:
            return {}
        adj = defaultdict(list)
        nodes = set(seed_list)
        frontier = deque(seed_list)
        while frontier:
            current = frontier.popleft()
            for edge in self._backend.edges_from(current):
                if edge.relation_type in self.META_RELATIONS:
                    continue
                if edge.target_id not in allowed:
                    continue
                adj[current].append((edge.target_id, max(edge.confidence_score, 1e-3)))
                if edge.target_id not in nodes:
                    nodes.add(edge.target_id)
                    frontier.append(edge.target_id)
        if not nodes:
            return {}
        n_seeds = len(seed_list)
        personalization = {nid: (1.0 / n_seeds if nid in seed_list else 0.0) for nid in nodes}
        scores = dict(personalization)
        for _ in range(max_iter):
            new_scores = {nid: (1 - damping) * personalization[nid] for nid in nodes}
            for src, outs in adj.items():
                total_w = sum(w for _, w in outs) or 1.0
                contrib = damping * scores[src]
                for tgt, w in outs:
                    new_scores[tgt] = new_scores.get(tgt, 0.0) + contrib * (w / total_w)
            delta = sum(abs(new_scores[n] - scores[n]) for n in nodes)
            scores = new_scores
            if delta < tolerance:
                break
        return scores

    graph_module.GraphLayer.personalized_pagerank = forward_only_ppr
    try:
        db = ".abl_no_hippo.db"
        Path(db).unlink(missing_ok=True)
        m = _open(db)
        r = _eval_memory(m, workload, base)
        r.name = "− HippoRAG bidirectional PPR"
        rows.append(r)
        m.close()
        Path(db).unlink(missing_ok=True)
    finally:
        graph_module.GraphLayer.personalized_pagerank = original_ppr

    # Ablation 2: disable bi-temporal edge retirement on contradiction.
    # We monkeypatch the ingest pipeline to skip retiring the prior edge.
    from thought.ingest import pipeline as pipe_module
    original_detect = pipe_module.IngestPipeline._detect_contradictions

    def shallow_detect(self, new_triples, entity_id_by_name, unique_predicates,
                       source_id, now):
        """Detect contradictions but DO NOT retire the prior edge — what we'd
        have without bi-temporal supersession."""
        out = []
        for t in new_triples:
            if t.predicate not in unique_predicates:
                continue
            subj_id = entity_id_by_name[t.subject.name.lower()]
            new_obj_id = entity_id_by_name[t.object.name.lower()]
            rows_ = self._backend._conn.execute(
                "SELECT edge_id, object_id FROM triples WHERE subject_id = ? "
                "AND predicate = ? AND object_id != ?",
                (subj_id, t.predicate, new_obj_id),
            ).fetchall()
            for r2 in rows_:
                # No edge retirement.
                edge_id = self._backend.upsert_edge(
                    source_id=new_obj_id, target_id=r2["object_id"],
                    relation_type="CONTRADICTS", source_ref=source_id,
                    confidence_score=0.9, valid_from=now, learned_at=now,
                    confidence_class="inferred",
                )
                from thought.models import ContradictionRef
                out.append(ContradictionRef(
                    entity_a=new_obj_id, entity_b=r2["object_id"],
                    edge_id=edge_id, detected_at=now,
                ))
        return out

    pipe_module.IngestPipeline._detect_contradictions = shallow_detect
    try:
        db = ".abl_no_bitemp.db"
        Path(db).unlink(missing_ok=True)
        m = _open(db)
        r = _eval_memory(m, workload, base)
        r.name = "− Bi-temporal edge retirement"
        rows.append(r)
        m.close()
        Path(db).unlink(missing_ok=True)
    finally:
        pipe_module.IngestPipeline._detect_contradictions = original_detect

    # Ablation 3: disable the query router → use vector-only.
    # We bypass the classifier by always returning VIBE.
    from thought.router import classifier as cls_module
    original_classify = cls_module.RuleBasedClassifier.classify

    def always_vibe(self, query):
        from thought.models import QueryClass
        return QueryClass.VIBE, {"vibe": 1, "fact": 0, "change": 0}

    cls_module.RuleBasedClassifier.classify = always_vibe
    try:
        db = ".abl_no_router.db"
        Path(db).unlink(missing_ok=True)
        m = _open(db)
        r = _eval_memory(m, workload, base)
        r.name = "− Query router (force VIBE)"
        rows.append(r)
        m.close()
        Path(db).unlink(missing_ok=True)
    finally:
        cls_module.RuleBasedClassifier.classify = original_classify

    # Write report.
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Ablation study — marginal contribution of Tier A techniques\n"]
    lines.append(
        "Each row turns OFF one frontier technique to measure its marginal "
        "contribution to THOUGHT's accuracy on the 200-op comparison workload. "
        "Higher overall, FACT, CHANGE, HYBRID = better.\n"
    )
    lines.append("| Variant | Overall | FACT | CHANGE | HYBRID | Contradictions detected |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r.name} | {r.overall:.1%} | {r.fact:.1%} | {r.change:.1%} | "
            f"{r.hybrid:.1%} | {r.contradictions_detected} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run()
