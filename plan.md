THOUGHT
Temporal Hierarchical Object Union & Graph Hybrid Toolkit
It’s punchy, easy to remember, and sounds like something a senior engineer at a place like Thoughtworks or OpenAI would build.

Why the Name Fits the Architecture
Temporal: Captures your valid_from/until logic (Shortcoming 2 of OB1).

Hierarchical: Represents the Hot/Warm/Cold tiering and the scoping of Shared vs. Private data.

Object Union: Describes how the Router unifies different data types (Vectors, Nodes, raw Source) into a single coherent response.

Graph Hybrid: Explicitly calls out the logic layer that solves the "hallucinated relationship" problem.

The "THOUGHT" Pitch
If you were explaining this to a peer in the terminal or on GitHub, it sounds much more like a polished product:

"I'm building THOUGHT. It's a local MCP server that doesn't just store embeddings—it manages the lifecycle of your code context. It uses a temporal graph to ensure that when your codebase evolves, the AI doesn't get 'stuck' on deprecated versions of your functions. It effectively gives any LLM a persistent, auditable memory fabric that stays on your own machine." 

### A Next-Generation Context Architecture Built on the Shortcomings of What Came Before

---

## Part 1 — Where Existing Solutions Fall Short

Before describing what TGVH is, it is worth being precise about what existing systems fail to do and why those failures are structural — not fixable by adding features on top.

---

### OB1 (OpenBrain) — The Flat Vector Store

OB1 is built on Postgres + pgvector, exposed via an MCP server. Its architecture is a single `thoughts` table: store a piece of text, generate a vector embedding, retrieve by semantic similarity. The surrounding ecosystem — recipes, dashboards, integrations, Slack capture — is genuinely well-designed. But the core data layer has five structural limits that no amount of community extensions can fix.

---

**Shortcoming 1: No relationship logic.**

OB1 stores facts as isolated rows. There is no way to express that one thing belongs to, causes, contradicts, or depends on another. If you store "User A owns Company B" and "Company B is headquartered in Florida," those two facts share no structural connection — they are two unrelated vectors in the same table. An AI querying OB1 must infer relationships from semantic similarity alone, which means it can hallucinate connections that were never recorded as true.

*How TGVH overcomes this:* The Graph Layer stores entities as nodes and relationships as typed edges. The AI cannot invent a relationship that has no edge. Every connection is explicit, structural, and queryable — not inferred.

---

**Shortcoming 2: Facts are silently overwritten.**

OB1 has no concept of a fact's validity over time. If you store "Kendra prefers Adidas" in 2024 and "Kendra prefers Nike" in 2026, the system has two floating vectors with no relationship to each other. There is no indication that one superseded the other, no record of when the change happened, and no way to ask "what did we know about Kendra's preferences in 2024?"

*How TGVH overcomes this:* The Temporal Layer gives every node and edge a `valid_from` and `valid_until` window. Facts are never overwritten — they are retired. A `SUPERSEDES` edge links the new fact to the old one. Both remain queryable. The system can answer questions about any point in time.

---

**Shortcoming 3: No query routing — everything is always a vector search.**

OB1 routes every query through the same pipeline regardless of what is being asked. A question like "what did we decide about the API architecture?" (which has a definitive factual answer) gets the same fuzzy vector search as "find me something related to performance." One of those questions needs precision; the other needs intuition. Treating them identically means both get a worse answer than they deserve.

*How TGVH overcomes this:* The Router classifies every query before touching any data. VIBE questions go to the Vector Layer. FACT questions go to the Graph Layer. CHANGE questions go to the Temporal Layer. Each question is answered by the mathematical structure best suited to it.

---

**Shortcoming 4: No provenance — facts cannot be traced to their source.**

When OB1 returns a memory, there is no structural guarantee of where it came from. Source metadata can be added as a tag, but it is informal and not enforced. If a retrieved fact turns out to be wrong, there is no reliable way to trace it back to the original document, the session that produced it, or the time it was recorded.

*How TGVH overcomes this:* Every edge in the Graph Layer carries a mandatory `source_ref` — a pointer to the raw source that produced it. Every claim is permanently auditable. Tracing a fact back to its origin is a single graph traversal, not a manual investigation.

---

**Shortcoming 5: Multi-user is an afterthought.**

Row Level Security appears in OB1's repo as a *primitive* — a community-contributed extension listed alongside recipes and dashboards. It was not part of the original design. This means the system was built for a single user and adapted afterward, which produces a fundamentally weaker isolation model than one built with multi-tenancy from the start.

*How TGVH overcomes this:* Multi-user scoping is a first-class design principle. The Graph Layer has two native zones — a Shared Zone for organizational or universal facts, and a Private Zone for personal context — with `owner_id` on every private node and edge-level access control. It is not bolted on after the fact.

---

### Karpathy's LLM Wiki — The Compiling Pattern

Andrej Karpathy's gist (published April 2026, 5,000+ stars) describes a genuinely important insight: instead of re-deriving knowledge from raw documents on every query, have an LLM maintain a persistent, compounding wiki that grows richer with every source added. The knowledge is compiled once and kept current.

The pattern is correct. The implementation guidance is intentionally abstract — Karpathy himself writes: *"This document is intentionally abstract. It describes the idea, not a specific implementation."* The shortcomings are not with the insight; they are with what the pattern leaves unsolved, as documented extensively in the gist's own comment thread.

---

**Shortcoming 1: The wiki is lossy compression.**

When an LLM ingests a raw document and writes a wiki page, it summarises. Caveats get dropped. Exact wording changes. Minority views disappear. Dates become approximate. Edge cases are omitted. Once users query the wiki instead of the original sources, these summary errors become part of the knowledge base — and there is no way to detect them because the connection between wiki page and source document is informal.

As commenter `a-a-k` wrote directly on the gist: *"Once people start querying the wiki instead of the original material, summary errors become part of the knowledge base."*

*How TGVH overcomes this:* Raw sources are immutable and permanently linked to derived facts via `source_ref` edges. The Graph Layer does not replace raw sources — it indexes them. Any fact can be traced back to the exact source that produced it, and a re-ingestion can update derived facts without losing the original.

---

**Shortcoming 2: The context window is still a ceiling.**

The wiki's `index.md` file — the catalogue of all pages — grows linearly with the knowledge base. At small scale it works well. At several hundred pages, searching the index becomes a context dump of its own. Commenter `superimpactful` identified this precisely: *"Once the index grows large enough, you've traded one context dump for a slightly smaller one."*

The pattern has no structural answer to scale. Karpathy suggests adding a search tool like `qmd` as an optional addition, but this is acknowledged as supplementary infrastructure, not a core architectural solution.

*How TGVH overcomes this:* The Router eliminates the linear scan entirely. Every query is classified and dispatched to a bounded lookup — graph traversal, ANN search, or temporal window scan. The search surface stays constant as the knowledge base grows. There is no index file to scan.

---

**Shortcoming 3: Stale facts persist silently.**

The wiki has no temporal awareness. A page written in January reflects what was true in January. If the underlying reality changes in March, the page is stale — but nothing in the system flags this. The LLM is asked to notice staleness during a "lint" pass, but this relies on the LLM remembering to run the pass, correctly identifying what changed, and updating all affected pages. There is no structural enforcement.

*How TGVH overcomes this:* Every fact has a `valid_from` / `valid_until` window. The Temporal Layer's consolidation engine runs on a schedule, scanning the warm tier for facts whose validity windows have closed without a replacement. Staleness is detected structurally, not by asking an LLM to remember.

---

**Shortcoming 4: Contradictions are noted, not resolved.**

The Karpathy wiki asks the LLM to flag contradictions between pages. This is noted as a lint operation — periodic, manual, informal. There is no structural concept of a contradiction: no way to query "what facts about X are currently contested," no way to track when a contradiction was detected, and no enforcement that it gets resolved.

As `a-a-k` noted: *"'Ask the LLM to maintain it' is not an engineering solution unless there are validators, source hashes, span-level citations, regression tests, and human review."*

*How TGVH overcomes this:* Contradictions are a typed edge — `CONTRADICTS` — in the Graph Layer. When ingestion detects a new fact that conflicts with an existing one, both facts remain as nodes and a `CONTRADICTS` edge is created between them with a `detected_at` timestamp and `confidence_score`. Contradictions are queryable data, not LLM notes.

---

**Shortcoming 5: No semantic search — context injection is the only retrieval method.**

The wiki retrieves knowledge by reading pages into the context window. There is no vector search. This means queries that require fuzzy, associative thinking — "find something related to the mood of this project" or "what's similar to this approach we took before" — have no structural answer. The LLM must read everything and hope the relevant material is present.

*How TGVH overcomes this:* The Vector Layer handles all intuition and associative queries using approximate nearest neighbor search. The Router sends the right questions to the right layer. Fuzzy retrieval and precise retrieval coexist without compromise.

---

## Part 2 — The TGVH Solution

Having identified the specific failures of both systems, TGVH is designed to address each one structurally — not through workarounds, but through architecture.

The core principle: **a memory system should know what kind of question is being asked before it searches anything, store facts with their origin and validity, and never lose history in the act of updating.**

---

### The Three Layers

**Layer 1 — Vector (Intuition Memory)**

Stores every fact as a high-dimensional semantic coordinate. Handles fuzzy, associative queries using approximate nearest neighbor search. Every embedding carries a version tag — if the embedding model is upgraded, mismatched embeddings are detectable and correctable. This is the layer OB1 uses exclusively. In TGVH it is one tool among three.

**Layer 2 — Graph (Logic Memory)**

Stores entities as nodes and relationships as typed edges. Every edge carries `source_ref` (provenance), `confidence_score`, `relation_type`, `valid_from`, and `valid_until`. Special edge types — `CONTRADICTS`, `SUPERSEDES`, `DERIVED_FROM` — make the structure self-documenting. No relationship can be hallucinated; if an edge does not exist, the relationship is not known.

**Layer 3 — Temporal (Timeline Memory)**

Manages the lifecycle of every fact. Facts are never deleted — they are retired with `valid_until` and replaced with a `SUPERSEDES` link. The hot/warm/cold tier model controls retrieval priority. A background consolidation engine runs on the warm tier, detecting contradictions, merging duplicates, and flagging stale facts structurally rather than relying on prompted maintenance.

---

### The Router

The Router is the component that neither OB1 nor the Karpathy wiki has. It classifies every incoming query into one of four types before any data is searched:

| Query Type | Signal | Dispatched To |
|---|---|---|
| VIBE | Similarity, association, mood | Vector Layer |
| FACT | Ownership, membership, causation | Graph Layer |
| CHANGE | History, supersession, timeline | Temporal Layer |
| HYBRID | Crosses multiple concerns | All three, results merged |

Every response is bounded to ≤10 results regardless of knowledge base size. The context delivered to the AI is a precise map, not a wall of text.

---

### The Data Lifecycle

```
[ Raw Source ]  ←  immutable, always retained
      ↓
[ Ingestion ]
  ├── Embedding → Vector Layer
  ├── Entity + relation extraction → Graph Layer (with source_ref)
  └── Contradiction check → creates CONTRADICTS edge if conflict found
      ↓
[ HOT tier ]    ← current session, highest priority
      ↓ (48 hours)
[ WARM tier ]   ← consolidation engine runs here
      ↓ (30 days without access)
[ COLD tier ]   ← archival, still queryable
```

---

### Multi-User Design

The Graph Layer has two native zones:

- **Shared Zone** — facts visible to all users (public knowledge, org-level context)
- **Private Zone** — personal facts scoped by `owner_id`

Edges can cross zones (a personal preference can reference a shared entity). Contradiction detection runs within each scope. A private fact contradicting a shared one surfaces a `CONTRADICTS` edge visible only to that user. This is not Row Level Security bolted on afterward — it is the native structure of the graph.

---

## Part 3 — Summary of Advantages

| Failure | OB1 | Karpathy Wiki | TGVH |
|---|---|---|---|
| No relationship logic | ✗ flat rows | ✗ flat markdown | ✅ Graph Layer with typed edges |
| Facts silently overwritten | ✗ no history | ✗ stale pages | ✅ Temporal validity + SUPERSEDES |
| No query routing | ✗ always vector | ✗ always inject | ✅ Router classifies first |
| No provenance | ✗ informal tags | ✗ informal citations | ✅ source_ref on every edge |
| Multi-user afterthought | ✗ RLS primitive | ✗ single-user only | ✅ two-zone graph, native |
| Context window ceiling | ✗ at scale | ✗ index.md grows | ✅ bounded router, no ceiling |
| Contradiction unstructured | ✗ none | ✗ LLM lint only | ✅ CONTRADICTS typed edge |
| No semantic search | ✅ pgvector | ✗ none | ✅ ANN with version tracking |

---

## Part 4 — Implementation Phases

**Phase 1 — Graph + Temporal Foundation**
Build the graph schema with all typed edges including `CONTRADICTS`, `SUPERSEDES`, `DERIVED_FROM`. Implement temporal validity windows. This is the hardest work and the clearest differentiation from everything in the current ecosystem.

**Phase 2 — Vector Layer + Provenance**
Add vector embeddings with version tracking. Every ingested fact produces both a graph node and a vector embedding simultaneously, linked by shared ID. The `source_ref` edge is created at this stage.

**Phase 3 — Router**
Build the query classifier. Four classes to start. Tune against real queries from your target use case before exposing externally.

**Phase 4 — MCP Interface + Multi-User**
Expose via MCP with two operations: `remember()` and `recall()`. Implement the two-zone graph for multi-tenancy. Test with Claude Code as the first client.

**Phase 5 — Consolidation Engine**
Build the background warm-tier process: contradiction detection, duplicate merging, staleness flagging, cold-tier demotion. This is what transforms a database into a memory fabric that improves over time without human maintenance.

---

## The One-Sentence Pitch

OB1 stores your thoughts. Karpathy's wiki compiles your knowledge. TGVH **remembers with provenance, understands relationships, detects contradictions, and never forgets what used to be true** — and it routes every query to the right mathematical structure before touching a single byte of data.
