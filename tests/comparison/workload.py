"""Synthetic workload for the comparison harness.

Generates a deterministic 200-operation mix (50 each of VIBE, FACT, CHANGE,
HYBRID-class queries) over a knowledge base of person/company/preference
facts with temporal evolution and at least one inserted contradiction. Each
``RecallCase`` has a ``ground_truth`` set of entity names that the system
ought to surface in its top-10.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal


@dataclass(frozen=True)
class IngestStep:
    content: str
    scope: Literal["shared", "private"] = "shared"
    owner_id: str | None = None
    offset_days: int = 0
    unique_predicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecallCase:
    kind: Literal["VIBE", "FACT", "CHANGE", "HYBRID"]
    query: str
    ground_truth: tuple[str, ...]
    forbidden: tuple[str, ...] = ()  # values that MUST NOT appear in top-3
    as_of_offset_days: int | None = None
    scope: Literal["shared", "private", "all"] = "all"
    owner_id: str | None = None


@dataclass
class Workload:
    ingests: list[IngestStep] = field(default_factory=list)
    recalls: list[RecallCase] = field(default_factory=list)


def _person_company_pairs():
    """Deterministic set of person-owns-company facts."""
    persons = [
        "Alice", "Bob", "Carol", "Diana", "Evan",
        "Farah", "Gita", "Hari", "Ivy", "Jules",
    ]
    companies = [
        "Acme", "Beacon", "Comet", "Dynamo", "Echo",
        "Fjord", "Glacier", "Helix", "Indigo", "Juno",
    ]
    return list(zip(persons, companies))


def build_workload() -> Workload:
    w = Workload()
    pairs = _person_company_pairs()

    # 1) Ownership facts (shared scope).
    for p, c in pairs:
        w.ingests.append(IngestStep(content=f"{p} owns {c} Corp."))

    # 2) Membership chain — companies are part of HoldCos.
    holdcos = ["NorthHold", "SouthHold"]
    for i, (_, c) in enumerate(pairs):
        h = holdcos[i % len(holdcos)]
        w.ingests.append(IngestStep(content=f"{c} is part of {h}."))

    # 3) Reports-to chain (deeper FACT graph).
    for i in range(len(pairs) - 1):
        p1 = pairs[i][0]
        p2 = pairs[i + 1][0]
        w.ingests.append(IngestStep(content=f"{p1} reports to {p2}."))

    # 4) Per-user preferences that EVOLVE — CHANGE queries depend on this.
    # Each user prefers brand A at t=-400d, then brand B at t=0, with the
    # PREFERS predicate marked unique so the second overwrite is a structured
    # contradiction.
    users = [("kendra", "Adidas", "Nike"), ("mike", "Pepsi", "Coke")]
    for owner, old, new in users:
        w.ingests.append(IngestStep(
            content=f"{owner.title()} prefers {old}.",
            scope="private", owner_id=owner, offset_days=-400,
        ))
        w.ingests.append(IngestStep(
            content=f"{owner.title()} prefers {new}.",
            scope="private", owner_id=owner, offset_days=0,
            unique_predicates=("PREFERS",),
        ))

    # 5) VIBE queries — fuzzy similarity. Ground truth = any of the relevant
    # company names appearing in the kb that share the vibe.
    w.recalls.extend([
        RecallCase("VIBE", "find something like Beacon", ("beacon",)),
        RecallCase("VIBE", "anything similar to Helix", ("helix",)),
        RecallCase("VIBE", "find something related to Comet", ("comet",)),
        RecallCase("VIBE", "the vibe of Echo", ("echo",)),
        RecallCase("VIBE", "anything reminds me of Juno", ("juno",)),
    ] * 10)  # 50 VIBE

    # 6) FACT queries — relational.
    fact_cases = []
    for p, c in pairs:
        fact_cases.append(
            RecallCase("FACT", f"who owns {c}", (p.lower(), c.lower()))
        )
    # five more relational queries about the holdco chain
    for p, c in pairs[:5]:
        fact_cases.append(
            RecallCase("FACT", f"what is {c} part of", (c.lower(),))
        )
    w.recalls.extend((fact_cases * 4)[:50])  # 50 FACT

    # 7) CHANGE queries — bi-temporal.
    # Historical queries set `forbidden` to the contemporary value: a system
    # that returns it has failed the temporal correctness test even if it
    # also returns the right answer.
    change_cases = []
    for owner, old, new in users:
        change_cases.extend([
            RecallCase("CHANGE", f"what did {owner.title()} prefer in 2024",
                       ground_truth=(old.lower(),),
                       forbidden=(new.lower(),),
                       as_of_offset_days=-200,
                       scope="all", owner_id=owner),
            RecallCase("CHANGE", f"what does {owner.title()} currently prefer",
                       ground_truth=(new.lower(),),
                       forbidden=(old.lower(),),
                       as_of_offset_days=0,
                       scope="all", owner_id=owner),
            RecallCase("CHANGE", f"history of {owner.title()}'s preferences",
                       ground_truth=(old.lower(), new.lower()),
                       scope="all", owner_id=owner),
        ])
    w.recalls.extend((change_cases * 9)[:50])  # 50 CHANGE

    # 8) HYBRID queries — mix temporal and FACT signals.
    hybrid_cases = [
        RecallCase("HYBRID", "when did Alice take over Acme",
                   ("alice", "acme"), as_of_offset_days=0),
        RecallCase("HYBRID", "previously who owned Beacon", ("beacon",),
                   as_of_offset_days=-200),
        RecallCase("HYBRID", "what was related to Comet historically",
                   ("comet",), as_of_offset_days=-200),
    ]
    w.recalls.extend((hybrid_cases * 17)[:50])  # 50 HYBRID

    assert len(w.recalls) == 200
    return w


def now_anchored(base: datetime | None = None) -> datetime:
    return base or datetime(2026, 5, 13, tzinfo=UTC)


def step_now(base: datetime, step: IngestStep) -> datetime:
    return base + timedelta(days=step.offset_days)


def case_as_of(base: datetime, case: RecallCase) -> datetime | None:
    if case.as_of_offset_days is None:
        return None
    return base + timedelta(days=case.as_of_offset_days)
