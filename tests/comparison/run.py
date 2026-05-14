"""Comparison-harness runner.

Drives the same workload through THOUGHT, OB1-simulator, and Karpathy-wiki-
simulator and produces ``docs/comparison.md`` containing measured numbers:
relationship recall@10, temporal correctness, contradictions detected, p50/p95
latency, and a boolean qualitative-capability matrix.

This is not a microbenchmark — the deterministic embedder is intentionally
simple so the numbers reflect ARCHITECTURE quality, not embedding-model
quality.  Subbing in BGE-M3 or MiniLM improves all three competitors
proportionally — what matters here is the gap between architectures.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from thought.memory import Memory

from .karpathy_simulator import KarpathyWikiSimulator
from .ob1_simulator import OB1Simulator
from .workload import Workload, build_workload, case_as_of, now_anchored, step_now


@dataclass
class SystemResult:
    name: str
    correct_by_class: dict[str, int]
    total_by_class: dict[str, int]
    latencies_ms: list[float]
    contradictions_detected: int
    temporal_correct: int
    temporal_total: int

    def recall_at_10_overall(self) -> float:
        c = sum(self.correct_by_class.values())
        t = sum(self.total_by_class.values())
        return c / t if t else 0.0

    def recall_at_10_for(self, kind: str) -> float:
        c = self.correct_by_class.get(kind, 0)
        t = self.total_by_class.get(kind, 0)
        return c / t if t else 0.0

    def temporal_correctness(self) -> float:
        return self.temporal_correct / self.temporal_total if self.temporal_total else 0.0

    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = max(0, min(len(sorted_l) - 1, int(0.95 * len(sorted_l))))
        return sorted_l[idx]


def _hit_correct(
    returned: set[str], truth: tuple[str, ...], forbidden: tuple[str, ...] = ()
) -> bool:
    """Multi-element truth must be FULLY covered; single-element truth must
    appear; ``forbidden`` items in the returned set DISQUALIFY the answer
    (this is what catches OB1 / Karpathy returning contemporary values for
    historical queries — they don't know better and shouldn't get credit).
    """
    truth_set = {t.lower() for t in truth}
    forbidden_set = {f.lower() for f in forbidden}
    if forbidden_set and (returned & forbidden_set):
        return False
    if len(truth_set) <= 1:
        return bool(returned & truth_set)
    return truth_set.issubset(returned)


def _names_from_thought(hits) -> set[str]:
    out: set[str] = set()
    for h in hits:
        out.add(h.entity.name.lower())
        out.add(h.entity.canonical_name.lower())
    return out


def run_thought(workload: Workload, db_path: str) -> SystemResult:
    base = now_anchored()
    mem = Memory.open(db_path=db_path, embedder_choice="deterministic", embedder_dim=128)
    try:
        contradictions = 0
        for step in workload.ingests:
            r = mem.remember(
                content=step.content,
                scope=step.scope,
                owner_id=step.owner_id,
                now=step_now(base, step),
                unique_predicates=set(step.unique_predicates) or None,
            )
            contradictions += len(r.contradictions_detected)

        correct: dict[str, int] = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
        total: dict[str, int] = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
        latencies: list[float] = []
        temporal_correct = 0
        temporal_total = 0

        for case in workload.recalls:
            total[case.kind] += 1
            t0 = time.perf_counter()
            result = mem.recall(
                query=case.query,
                limit=10,
                scope=case.scope,
                owner_id=case.owner_id,
                as_of=case_as_of(base, case),
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            returned = _names_from_thought(result.hits)
            if _hit_correct(returned, case.ground_truth, case.forbidden):
                correct[case.kind] += 1
            if case.kind == "CHANGE":
                temporal_total += 1
                if _hit_correct(returned, case.ground_truth, case.forbidden):
                    temporal_correct += 1

        return SystemResult(
            name="THOUGHT",
            correct_by_class=correct, total_by_class=total,
            latencies_ms=latencies,
            contradictions_detected=contradictions,
            temporal_correct=temporal_correct, temporal_total=temporal_total,
        )
    finally:
        mem.close()


def run_simulator(workload: Workload, sim, name: str) -> SystemResult:
    base = now_anchored()
    for step in workload.ingests:
        sim.ingest(content=step.content, now=step_now(base, step))

    correct: dict[str, int] = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
    total: dict[str, int] = {"VIBE": 0, "FACT": 0, "CHANGE": 0, "HYBRID": 0}
    latencies: list[float] = []
    temporal_correct = 0
    temporal_total = 0

    for case in workload.recalls:
        total[case.kind] += 1
        t0 = time.perf_counter()
        returned = sim.names_in_top_k(case.query, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        if _hit_correct(returned, case.ground_truth, case.forbidden):
            correct[case.kind] += 1
        if case.kind == "CHANGE":
            temporal_total += 1
            # OB1 / Karpathy have NO temporal awareness — a CHANGE query is only
            # "correct" if the right historical answer surfaces AND the
            # forbidden contemporary answer does NOT.
            if _hit_correct(returned, case.ground_truth, case.forbidden):
                temporal_correct += 1

    return SystemResult(
        name=name,
        correct_by_class=correct, total_by_class=total,
        latencies_ms=latencies,
        # Neither simulator has a contradiction model.
        contradictions_detected=0,
        temporal_correct=temporal_correct, temporal_total=temporal_total,
    )


CAPABILITIES = [
    ("bi-temporal as_of", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("source-grounded confidence class", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("contradiction as typed edge", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("multi-user scope isolation", {"THOUGHT": True, "OB1": "partial (RLS)", "Karpathy": False}),
    ("append-only audit log", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("Personalized PageRank retrieval", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("Ebbinghaus decay scoring", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("CRAG-style low-confidence flag", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("Matryoshka 2-pass ANN", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("Anthropic Contextual Retrieval", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("query router (VIBE/FACT/CHANGE)", {"THOUGHT": True, "OB1": False, "Karpathy": False}),
    ("forecasting (TLogic) [v0.2]", {"THOUGHT": "planned", "OB1": False, "Karpathy": False}),
]


def write_markdown(results: list[SystemResult], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Comparison harness — measured results\n")
    lines.append(
        "Workload: 200 recall ops (50 each of VIBE / FACT / CHANGE / HYBRID) "
        "over a deterministic knowledge base of 24 ingest steps with at least "
        "two structured contradictions inserted via PREFERS predicate.\n"
    )
    lines.append(
        "Embedder: deterministic hashed bag-of-words (same model for all three "
        "systems so the gap reflects architecture, not embedding quality).\n"
    )

    lines.append("## Recall@10 by query class\n")
    lines.append("| System | VIBE | FACT | CHANGE | HYBRID | overall |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | "
            f"{r.recall_at_10_for('VIBE'):.1%} | "
            f"{r.recall_at_10_for('FACT'):.1%} | "
            f"{r.recall_at_10_for('CHANGE'):.1%} | "
            f"{r.recall_at_10_for('HYBRID'):.1%} | "
            f"{r.recall_at_10_overall():.1%} |"
        )

    lines.append("\n## Latency (recall path, ms)\n")
    lines.append("| System | p50 | p95 |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r.name} | {r.p50_ms():.2f} | {r.p95_ms():.2f} |")

    lines.append("\n## Structural capabilities\n")
    lines.append("| Capability | THOUGHT | OB1 | Karpathy wiki |")
    lines.append("|---|---|---|---|")
    def render(v):
        if v is True:
            return "✅"
        if v is False:
            return "✗"
        return str(v)
    for cap, status in CAPABILITIES:
        lines.append(f"| {cap} | {render(status['THOUGHT'])} | {render(status['OB1'])} | {render(status['Karpathy'])} |")

    lines.append("\n## Contradictions detected (write-time)\n")
    for r in results:
        lines.append(f"- **{r.name}**: {r.contradictions_detected}")

    lines.append("\n## Temporal correctness on CHANGE queries\n")
    lines.append("| System | correct / total | rate |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.temporal_correct}/{r.temporal_total} | "
            f"{r.temporal_correctness():.1%} |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_path: str = "docs/comparison.md", db_path: str = ".cmp.db") -> None:
    workload = build_workload()
    Path(db_path).unlink(missing_ok=True)
    thought = run_thought(workload, db_path=db_path)
    ob1 = run_simulator(workload, OB1Simulator(dim=128), name="OB1")
    wiki = run_simulator(workload, KarpathyWikiSimulator(), name="Karpathy wiki")
    write_markdown([thought, ob1, wiki], Path(out_path))
    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    run()
