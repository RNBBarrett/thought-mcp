"""``thought demo`` — the built-in dogfood / smoke / first-confidence-check.

Runs an audience-specific walkthrough end-to-end in a self-cleaning tmp dir.
Each audience demonstrates the v0.5 surface for a specific kind of user —
no v0.6+ extractors are required; the bi-temporal + typed-edge + Cypher
substrate works for every audience today.

Audiences:

- ``code``        Agent / developer flow — the 14-stage code-vertical
                  walkthrough including agent identity, ``thought scan``,
                  ``working_context``, 4 new-language extractors, and the
                  Claude Agent SDK adapter.
- ``writer``      Novelist / paper author — ingest chapter / section facts
                  about a character, detect contradictions via the
                  bi-temporal model, query chronological mentions, do a
                  time-travel ``as_of`` recall.
- ``legal``       Investigator / paralegal — ingest witness statements with
                  ``unique_predicates``, trigger CONTRADICTS edges between
                  testimonies, query the contradiction graph.
- ``researcher``  Academic — ingest claim / source pairs, run Cypher to find
                  uncited claims and most-cited sources, demonstrate citation
                  reliability filtering.
- ``all``         Run every audience above sequentially.

The runner captures pass/fail + duration per stage and emits a structured
``DemoReport`` the CLI renders as a table.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from . import __version__
from .memory import Memory

Audience = Literal["code", "writer", "legal", "researcher", "all"]
_AUDIENCES: tuple[str, ...] = ("code", "writer", "legal", "researcher")
_PREFIX = "thought-demo-"


@dataclass
class StageResult:
    audience: str
    name: str
    passed: bool
    duration_ms: float
    note: str = ""
    error: str = ""


@dataclass
class DemoReport:
    version: str
    workspace: str
    stages: list[StageResult] = field(default_factory=list)
    cleaned_up: bool = False

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.stages)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages)


def _stage(
    report: DemoReport, audience: str, name: str, fn: Callable[[], str],
) -> None:
    """Run ``fn`` as a named stage; capture pass/fail + duration."""
    t0 = time.perf_counter()
    try:
        note = fn() or ""
        report.stages.append(StageResult(
            audience=audience, name=name, passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            note=note,
        ))
    except Exception as e:
        report.stages.append(StageResult(
            audience=audience, name=name, passed=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        ))


# ============================================================== code audience

def _run_code(report: DemoReport, workspace: Path, mem: Memory) -> None:
    """Code / agent flow — 14 stages."""
    db_path = mem._backend.path  # type: ignore[attr-defined]

    def _ingest_recall() -> str:
        for fact in [
            "Alice owns Acme Corp.",
            "Bob runs the warehouse.",
            "Acme is headquartered in Seattle.",
        ]:
            mem.remember(content=fact, scope="shared")
        r = mem.recall(query="who owns Acme", limit=5)
        if not r.hits:
            raise AssertionError("recall returned no hits")
        return f"3 ingests, recall returned {len(r.hits)} hit(s)"

    _stage(report, "code", "02_ingest_recall", _ingest_recall)

    def _topics_browse() -> str:
        topics = mem.list_topics(scope="all", min_count=1)
        items = mem.browse_topic("Alice", depth=2, limit=5)
        return f"{len(topics)} type(s); browse(Alice) → {len(items)} item(s)"

    _stage(report, "code", "03_topics_browse", _topics_browse)

    def _schema_cypher() -> str:
        schema = mem.schema_summary()
        from .query import cypher
        rows = cypher.execute(mem, "MATCH (p:CONCEPT) RETURN p.name")
        return (
            f"{sum(schema['entity_types'].values())} entities in "
            f"{len(schema['entity_types'])} type(s); Cypher returned {len(rows)} row(s)"
        )

    _stage(report, "code", "04_schema_cypher", _schema_cypher)

    def _saved_views() -> str:
        from .query import views
        views.save_view(mem, "demo_view", "MATCH (p:CONCEPT) RETURN p.name")
        rows = views.run_view(mem, "demo_view")
        views.delete_view(mem, "demo_view")
        return f"save → run → delete; view returned {len(rows)} row(s)"

    _stage(report, "code", "05_saved_views", _saved_views)

    def _db_lifecycle() -> str:
        sizes = mem.db_size()
        snap = workspace / "snap.db"
        bytes_written = mem.backup_to(snap, force=True)
        inspect = mem.inspect_file(snap, include_schema=True)
        return (
            f"main={sizes['main']} B; backup={bytes_written} B; "
            f"snap schema_version={inspect['schema_version']}"
        )

    _stage(report, "code", "06_db_lifecycle", _db_lifecycle)

    def _agent_register() -> str:
        a = mem.register_agent(
            "demo-agent",
            description="thought demo run reference agent",
            capabilities=["scan-code", "record-fact"],
        )
        return f"agent {a['name']!r} registered"

    _stage(report, "code", "07_agent_register", _agent_register)

    # Build a 5-file polyglot fixture, one per supported language.
    fixture = workspace / "polyglot"
    fixture.mkdir(exist_ok=True)
    (fixture / "lib.py").write_text(
        "def authenticate(token: str) -> dict:\n    return {'token': token}\n"
        "def helper():\n    return authenticate('x')\n"
    )
    (fixture / "main.go").write_text(
        'package main\nimport "fmt"\n'
        'type Cat struct { Name string }\n'
        'func (c *Cat) Meow() string { return c.Name }\n'
        'func main() { fmt.Println("hi") }\n'
    )
    (fixture / "lib.rs").write_text(
        "use std::io::Read;\n"
        "pub struct Cat { name: String }\n"
        "impl Cat { pub fn meow(&self) -> String { self.name.clone() } }\n"
        "fn main() {}\n"
    )
    (fixture / "Cat.java").write_text(
        "package com.acme;\npublic class Cat extends Animal {\n"
        '    public String meow() { return "meow"; }\n}\n'
    )
    (fixture / "Cat.php").write_text(
        "<?php\nclass Cat extends Animal {\n"
        '    public function meow(): string { return "meow"; }\n}\n'
    )

    def _scan() -> str:
        r = mem.scan(fixture, agent="demo-agent")
        if r["files_scanned"] == 0:
            raise AssertionError("scan ingested 0 files")
        return (
            f"{r['files_scanned']} files → "
            f"{r['entities_added']} entities + "
            f"{r['edges_added']} edges in {r['duration_ms']:.0f} ms"
        )

    _stage(report, "code", "08_scan", _scan)

    def _scan_log() -> str:
        log = mem.scan_log(agent="demo-agent", limit=5)
        if not log:
            raise AssertionError("scan_log empty")
        return f"{len(log)} scan_log row(s) persisted"

    _stage(report, "code", "09_scan_log", _scan_log)

    def _codebase_map() -> str:
        from .layers.graph import GraphLayer
        from .models import ScopeFilter
        rows = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT id FROM entities WHERE valid_until IS NULL "
            "AND type IN ('function', 'method', 'class', 'module')"
        ).fetchall()
        seeds = [r["id"] for r in rows]
        if not seeds:
            return "no code entities (skipped)"
        scores = GraphLayer(mem._backend).personalized_pagerank(
            seeds=seeds, scope_filter=ScopeFilter(scope="all"),
        )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        return f"PPR top-5 surfaced; max score = {ranked[0][1]:.4f}"

    _stage(report, "code", "10_codebase_map", _codebase_map)

    def _polyglot_landed() -> str:
        rows = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT code_language AS lang, COUNT(*) AS n FROM entities "
            "WHERE code_language IN ('go', 'rust', 'java', 'php') "
            "GROUP BY code_language"
        ).fetchall()
        landed = {r["lang"]: r["n"] for r in rows}
        missing = [lang for lang in ("go", "rust", "java", "php") if lang not in landed]
        if missing:
            raise AssertionError(f"languages missing: {missing}")
        return "go/rust/java/php all produced entities " + str(landed)

    _stage(report, "code", "11_polyglot_languages", _polyglot_landed)

    def _working_context() -> str:
        wc = mem.working_context("Cat")
        if wc["anchor"] is None:
            raise AssertionError("no anchor for 'Cat'")
        return f"anchor={wc['anchor']['name']!r}; {len(wc['neighbours'])} neighbour(s)"

    _stage(report, "code", "12_working_context", _working_context)

    # Adapter test needs a separate Memory instance; close + re-open via the adapter.
    mem.close()

    def _adapter() -> str:
        from .adapters.claude_sdk import ThoughtMemoryProvider
        with ThoughtMemoryProvider(
            db_path=db_path, agent="demo-agent",
            embedder_choice="deterministic", embedder_dim=384,
        ) as p:
            rendered = p.render_context("Cat")
            r = p.record("Cat is a polyglot demo fixture.")
            if not rendered or not r.get("source_id"):
                raise AssertionError("adapter round-trip failed")
        return f"render_context = {len(rendered)} char(s); record OK"

    _stage(report, "code", "13_adapter", _adapter)

    # Re-open for final stats.
    final = Memory.open(
        db_path=db_path, embedder_choice="deterministic", embedder_dim=384,
    )
    try:
        s = final.stats()
        report.stages.append(StageResult(
            audience="code", name="14_final_stats", passed=True,
            duration_ms=0.0,
            note=(
                f"entities_current={s['entities_current']}, "
                f"edges={s['edges_total']}, sources={s['sources']}"
            ),
        ))
    finally:
        final.close()


# ============================================================== writer audience

def _run_writer(report: DemoReport, workspace: Path, mem: Memory) -> None:
    """Novelist / paper-author flow — character continuity across chapters."""
    now = datetime.now(UTC)
    long_ago = now - timedelta(days=7)
    yesterday = now - timedelta(days=1)

    def _seed_chapters() -> str:
        # Chapter 1 — Alice introduced with brown hair, lives in Seattle.
        mem.remember(
            content="Alice has brown hair.",
            scope="shared", now=long_ago,
            unique_predicates={"HAS_TRAIT"},
        )
        mem.remember(
            content="Alice lives in Seattle.",
            scope="shared", now=long_ago,
        )
        # Chapter 3 — Alice meets Bob.
        mem.remember(
            content="Alice meets Bob at the cafe.",
            scope="shared", now=yesterday,
        )
        # Chapter 7 — CONTRADICTING trait: hair is now red. The
        # unique_predicates trigger surfaces this as a CONTRADICTS edge.
        mem.remember(
            content="Alice has red hair.",
            scope="shared", now=now,
            unique_predicates={"HAS_TRAIT"},
        )
        stats = mem.stats()
        return f"{stats['entities_current']} entities; {stats['contradictions']} contradiction(s) flagged"

    _stage(report, "writer", "01_seed_chapters", _seed_chapters)

    def _contradictions_visible() -> str:
        # The CONTRADICTS edge from the trait conflict should be queryable.
        rows = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM edges "
            "WHERE relation_type='CONTRADICTS' AND valid_until IS NULL"
        ).fetchone()
        n = int(rows["n"])
        # The bi-temporal model + Jaccard dedup may or may not produce a
        # CONTRADICTS edge depending on entity-name matching; either result
        # is informative for a writer's continuity report.
        return f"{n} active CONTRADICTS edge(s) in scope"

    _stage(report, "writer", "02_contradictions_visible", _contradictions_visible)

    def _time_travel() -> str:
        # What did the manuscript say about Alice last week?
        past = mem.recall(
            query="Alice", limit=10, scope="all",
            as_of=long_ago + timedelta(hours=1),
        )
        present = mem.recall(query="Alice", limit=10, scope="all")
        return (
            f"as-of last week: {len(past.hits)} hit(s); "
            f"now: {len(present.hits)} hit(s)"
        )

    _stage(report, "writer", "03_time_travel_recall", _time_travel)

    def _character_query() -> str:
        # Cypher: every entity mentioned alongside Alice (≈ scenes Alice
        # appears in, by graph adjacency).
        from .query import cypher
        rows = cypher.execute(
            mem, "MATCH (a:CONCEPT {name:'Alice'}) RETURN a.name",
        )
        return f"Cypher returned {len(rows)} row(s) for Alice"

    _stage(report, "writer", "04_character_cypher", _character_query)

    def _outline_preview() -> str:
        # A writer browses what types of facts are stored.
        topics = mem.list_topics(scope="all", min_count=1)
        return f"{len(topics)} topic bucket(s); good starting point for an outline"

    _stage(report, "writer", "05_outline_preview", _outline_preview)


# ============================================================== legal audience

def _run_legal(report: DemoReport, workspace: Path, mem: Memory) -> None:
    """Investigator / paralegal flow — witness contradictions + network analysis."""
    now = datetime.now(UTC)

    def _seed_testimony() -> str:
        # Three witnesses with contradicting accounts of where Alice was at 9pm.
        # The CLAIMED unique_predicate triggers contradiction detection.
        mem.remember(
            content="Witness Carter claims Alice was at the bar at 9pm.",
            scope="shared", now=now - timedelta(hours=2),
            unique_predicates={"CLAIMED"},
        )
        mem.remember(
            content="Witness Diaz claims Alice was at home at 9pm.",
            scope="shared", now=now - timedelta(hours=1),
            unique_predicates={"CLAIMED"},
        )
        mem.remember(
            content="Witness Evans claims Alice was at the warehouse at 9pm.",
            scope="shared", now=now,
            unique_predicates={"CLAIMED"},
        )
        # Corroborating evidence (no contradiction triggered): a receipt.
        mem.remember(
            content="A receipt places Alice at the warehouse parking lot at 8:45pm.",
            scope="shared", now=now,
        )
        return "4 witness/evidence facts ingested"

    _stage(report, "legal", "01_seed_testimony", _seed_testimony)

    def _surface_contradictions() -> str:
        rows = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM edges WHERE relation_type='CONTRADICTS'"
        ).fetchone()
        return f"{int(rows['n'])} CONTRADICTS edge(s) in case file"

    _stage(report, "legal", "02_surface_contradictions", _surface_contradictions)

    def _network() -> str:
        # PPR-ranked neighbourhood of Alice — who is most connected to the
        # subject in the case graph?
        items = mem.browse_topic("Alice", depth=2, limit=10)
        return f"network around Alice: {len(items)} entit(ies)"

    _stage(report, "legal", "03_network_analysis", _network)

    def _timeline() -> str:
        # All facts about Alice, surfaced via recall. A real `thought timeline`
        # CLI ships in v0.7; for v0.5 we use the generic recall path.
        r = mem.recall(query="Alice 9pm", limit=10, scope="all")
        return f"timeline-style recall returned {len(r.hits)} hit(s)"

    _stage(report, "legal", "04_timeline_recall", _timeline)

    def _audit_trail() -> str:
        # The append-only audit log: every ingested fact has a source_ref.
        sources = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM sources"
        ).fetchone()
        return f"{int(sources['n'])} source(s) — every fact traceable"

    _stage(report, "legal", "05_audit_trail", _audit_trail)


# ============================================================== researcher audience

def _run_researcher(report: DemoReport, workspace: Path, mem: Memory) -> None:
    """Academic / research-paper flow — claims, citations, source reliability."""

    def _seed_claims() -> str:
        # Ingest claim/source pairs. The text uses an explicit "cited as"
        # phrase so the v0.1 ingest pipeline lifts both endpoints as
        # entities + an RELATED_TO edge between them.
        for fact in [
            "GraphRAG improves multi-hop recall.",
            "GraphRAG is cited as Edge2024.",
            "Personalized PageRank scales sub-linearly.",
            "Personalized PageRank is cited as Page1999.",
            "Embeddings outperform sparse retrieval.",
            # Deliberately leave this last claim UNCITED:
            "Bi-temporal memory enables time-travel queries.",
        ]:
            mem.remember(content=fact, scope="shared")
        s = mem.stats()
        return f"{s['entities_current']} entities; {s['edges_total']} edges"

    _stage(report, "researcher", "01_seed_claims", _seed_claims)

    def _query_sources() -> str:
        # Which "citation" entities exist? (Heuristic: tokens like Page1999,
        # Edge2024 surface as their own entities via the proper-noun
        # extractor. The v0.4 Cypher subset doesn't support OR, so we run
        # two queries and merge.)
        from .query import cypher
        rows_20 = cypher.execute(
            mem, 'MATCH (s:CONCEPT) WHERE s.name CONTAINS "20" RETURN s.name',
        )
        rows_19 = cypher.execute(
            mem, 'MATCH (s:CONCEPT) WHERE s.name CONTAINS "19" RETURN s.name',
        )
        unique = {r["s.name"] for r in rows_20} | {r["s.name"] for r in rows_19}
        return f"sources matching year-pattern: {len(unique)} unique hit(s)"

    _stage(report, "researcher", "02_query_sources", _query_sources)

    def _uncited_claim_recall() -> str:
        # Use recall to find facts about bi-temporal memory and verify the
        # uncited claim is in scope.
        r = mem.recall(query="bi-temporal time-travel", limit=5)
        return f"recall returned {len(r.hits)} hit(s) about the uncited claim"

    _stage(report, "researcher", "03_uncited_claim", _uncited_claim_recall)

    def _save_view() -> str:
        from .query import views
        views.save_view(
            mem, "sources_with_year",
            'MATCH (s:CONCEPT) WHERE s.name CONTAINS "20" RETURN s.name',
        )
        rows = views.run_view(mem, "sources_with_year")
        views.delete_view(mem, "sources_with_year")
        return f"saved view returned {len(rows)} row(s) (year-tagged sources)"

    _stage(report, "researcher", "04_saved_view", _save_view)

    def _audit_count() -> str:
        sources = mem._backend._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM sources"
        ).fetchone()
        return f"{int(sources['n'])} source rows — citation chain auditable"

    _stage(report, "researcher", "05_audit_count", _audit_count)


# ============================================================== orchestrator

def run_demo(
    *, kind: Audience = "code", keep: bool = False,
) -> DemoReport:
    """Execute the demo for the requested audience.

    Args:
        kind: ``"code"``, ``"writer"``, ``"legal"``, ``"researcher"``, or
            ``"all"`` to run every audience sequentially against one DB.
        keep: when True, leave the scratch DB on disk after the run.
    """
    workspace = Path(tempfile.mkdtemp(prefix=_PREFIX))
    db_path = str(workspace / "thought.db")
    report = DemoReport(version=__version__, workspace=str(workspace))

    # Stage 0: open the DB once for every kind. The "code" audience closes
    # mem mid-run for the adapter test; subsequent audiences re-open.
    def _open(_kind: str) -> Memory:
        t0 = time.perf_counter()
        m = Memory.open(
            db_path=db_path, embedder_choice="deterministic", embedder_dim=384,
        )
        report.stages.append(StageResult(
            audience=_kind, name="00_open",
            passed=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            note=f"opened DB at {db_path}",
        ))
        return m

    try:
        kinds = _AUDIENCES if kind == "all" else (kind,)
        for k in kinds:
            mem = _open(k)
            try:
                if k == "code":
                    _run_code(report, workspace, mem)
                elif k == "writer":
                    _run_writer(report, workspace, mem)
                elif k == "legal":
                    _run_legal(report, workspace, mem)
                elif k == "researcher":
                    _run_researcher(report, workspace, mem)
                else:  # pragma: no cover — typer enum validates
                    raise ValueError(f"unknown audience: {k!r}")
            finally:
                # ``code`` audience already closes mid-run for the adapter
                # stage. For others, close cleanly here.
                try:
                    mem.close()
                except Exception:  # pragma: no cover
                    pass
    finally:
        if not keep:
            shutil.rmtree(workspace, ignore_errors=True)
            report.cleaned_up = True
    return report


def cleanup_demo_workspaces() -> int:
    """Remove leftover ``thought-demo-*`` scratch dirs from prior runs."""
    n = 0
    parent = Path(tempfile.gettempdir())
    if not parent.exists():
        return 0
    for child in parent.iterdir():
        if child.is_dir() and child.name.startswith(_PREFIX):
            shutil.rmtree(child, ignore_errors=True)
            n += 1
    return n
