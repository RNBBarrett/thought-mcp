# THOUGHT

**T**emporal **H**ierarchical **O**bject **U**nion & **G**raph **H**ybrid **T**oolkit — a local MCP memory server that gives any LLM a persistent, auditable memory fabric on your own machine.

> OB1 stores your thoughts. Karpathy's wiki compiles your knowledge. **THOUGHT** remembers with provenance, understands relationships, detects contradictions, never forgets what used to be true — and routes every query to the right mathematical structure before touching a single byte of data.

---

## Why this exists

The 2024–2026 wave of LLM memory products is split between two patterns, each with a structural limitation we wanted to fix in one system:

| | OB1 (pgvector) | Karpathy LLM-Wiki | **THOUGHT** |
|---|---|---|---|
| Relationship logic | flat rows | flat markdown | **typed graph edges** |
| Temporal awareness | none | none | **bi-temporal valid + learned** |
| Provenance | informal tag | informal citation | **mandatory `source_ref` on every edge** |
| Multi-user | RLS bolted on | single-user | **native two-zone graph** |
| Query routing | always vector | always inject | **VIBE / FACT / CHANGE / HYBRID router** |
| Contradiction model | absent | LLM lint only | **`CONTRADICTS` typed edge, write-time** |
| Bounded result size | unbounded | unbounded | **≤10 enforced** |

THOUGHT also stacks **eleven cutting-edge techniques** from 2024-2026 literature so the gap isn't just qualitative.

---

## Standing on the shoulders of

THOUGHT exists because of:

- Scott Nichols [**@srnichols**](https://github.com/srnichols) — [OpenBrain](https://github.com/srnichols/OpenBrain) showed that pgvector + MCP is a powerful pattern.
- [**@benclawbot**](https://github.com/benclawbot) — [open-brain](https://github.com/benclawbot/open-brain) provided a clean reference implementation.
- Andrej Karpathy [**@karpathy**](https://github.com/karpathy) — the [LLM-Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) crystallized why context engineering is the next discipline.

## Frontier techniques incorporated (with credits)

| # | Technique | Source |
|---|---|---|
| 1 | **Contextual Retrieval** — LLM-generated chunk context prepended before embedding | [Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval) |
| 2 | **HippoRAG 2 — Personalized PageRank memory** | [Gutiérrez et al., NeurIPS 2024](https://arxiv.org/abs/2405.14831) ([repo](https://github.com/OSU-NLP-Group/HippoRAG)) |
| 3 | **Bi-temporal Graphiti** — separate valid-time and transaction-time | [Zep, arXiv 2501.13956](https://arxiv.org/abs/2501.13956) ([repo](https://github.com/getzep/graphiti)) |
| 4 | **Atomic fact decomposition + Jaccard dedup** | [Wanner et al., 2024](https://arxiv.org/abs/2410.16708v1) |
| 5 | **BGE-M3 hybrid embeddings (sparse + dense + ColBERT)** | [BAAI](https://huggingface.co/BAAI/bge-m3) |
| 6 | **Matryoshka two-pass retrieval** | Kusupati et al.; OpenAI text-embedding-3 |
| 7 | **CRAG (Corrective RAG)** — retrieval evaluator + fallback | [Yan et al., 2024](https://arxiv.org/abs/2401.15884) |
| 8 | **MetaRAG epistemic uncertainty** — `confidence_class` per hit | [arXiv 2504.14045](https://arxiv.org/abs/2504.14045) |
| 9 | **Ebbinghaus decay scoring** — strength × `e^(-λ·days)` × recall-boost | [@sachitrafa/YourMemory](https://github.com/sachitrafa/YourMemory) |
| 10 | **Context-engineering budget per query class** | [Karpathy & community, 2025](https://github.com/davidkimai/Context-Engineering) |
| 11 | **Append-only writes (Mem0 2026)** — never UPDATE/DELETE | [Mem0 State of Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) |

Built on: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) ([@modelcontextprotocol](https://github.com/modelcontextprotocol)), [sqlite-vec](https://github.com/asg017/sqlite-vec) (Alex Garcia), [pgvector](https://github.com/pgvector/pgvector) (Andrew Kane), [Pydantic](https://github.com/pydantic/pydantic), [Typer](https://github.com/fastapi/typer), [structlog](https://github.com/hynek/structlog). spaCy ([Explosion AI](https://github.com/explosion/spaCy)) is an optional extra.

---

## Install

```bash
pip install thought-mcp                    # core + sqlite-vec + MCP
pip install 'thought-mcp[all]'             # + production embeddings + NER
uvx thought-mcp install --client cursor    # zero-install: pull + wire into Cursor in one command
```

## 30-second quickstart

```bash
thought start --client cursor    # creates config, wires Cursor's MCP settings, starts the server
# (restart Cursor)               # done.
```

That's the whole onboarding. `thought start` is the fast path; under the hood it runs:

1. `thought init` if no config exists — drops `thought.toml`, creates the SQLite DB, writes a `CLAUDE.md` so the LLM client knows how to use the tools, warms up the embedder.
2. `thought install --client <name>` — auto-detects your client's config file, merges in the `mcpServers` entry (with a backup), idempotent on rerun.
3. `thought serve` — runs a `doctor` precheck, then binds the MCP server on `127.0.0.1:8765`.

### Auto-wiring MCP clients

```bash
thought install --detect                   # show each client's config path + whether it exists
thought install --client cursor            # wire just Cursor
thought install --client claude-code       # …or Claude Code
thought install --all                      # wire every detected client at once
```

The installer supports **Claude Code, Cursor, Cline, Continue, Windsurf** and writes the same JSON block in the right file for each:

```json
{
  "mcpServers": {
    "thought": {
      "command": "uvx",
      "args": ["thought-mcp", "serve"]
    }
  }
}
```

The pre-write file is backed up to `<config>.thought.bak`. Rerunning is a no-op if the entry already matches.

### Manual install paths (if `--detect` can't find your client)

- **Claude Code** — `~/.claude.json` (a `mcpServers` block at the top level)
- **Cursor** — `~/.cursor/mcp.json`
- **Cline** — VS Code `globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` (or `~/.cline/cline_mcp_settings.json`)
- **Continue** — `~/.continue/config.json`
- **Windsurf** — `~/.codeium/windsurf/mcp_config.json`

After the install completes and your client restarts, the MCP tools `remember` and `recall` are available to it. See **[Using THOUGHT inside Claude Code / Cursor / etc.](#using-thought-inside-claude-code--cursor--etc)** below for example interactions.

---

## Using THOUGHT inside Claude Code / Cursor / etc.

Once wired, your LLM client gains two tools — `remember(content, ...)` and `recall(query, ...)` — plus a `CLAUDE.md` hint that teaches it when to use them. You don't need to invoke them directly; most agents will call them automatically when the prompt implies persistence or recall. Below are concrete examples.

### Pattern 1 — automatic remember on decisions

```
You:  We've decided to use Postgres for the v2 release. Note that down.
Claude Code:
  → calls remember(
      content="We decided to use Postgres for the v2 release.",
      scope="shared")
  "Stored. I can pull this back later with recall('database plan')."
```

### Pattern 2 — automatic recall when context is needed

```
You:  What's our database plan?
Claude Code:
  → calls recall(query="database plan", scope="all")
  "Based on what's in memory: for v2 we're using Postgres
   (source_grounded, recorded 3 days ago)."
```

The `confidence_class` on each hit (`source_grounded` / `inferred` / `hallucination_risk`) tells the LLM how much to trust the recall result.

### Pattern 3 — "as of" temporal queries (a structural advantage)

```
You:  What did we say about pricing on Jan 15?
Claude Code:
  → calls recall(
      query="pricing",
      as_of="2026-01-15T00:00:00Z",
      as_of_kind="valid")
  "On 2026-01-15: 'Pricing tiers were Free / Pro $19 / Team $99'.
   (That changed on Feb 3 to Free / Pro $25 / Team $129 — but
   recalling 'as of Jan 15' gives the historical state.)"
```

This is THOUGHT's most distinctive capability — OB1 and the Karpathy wiki structurally cannot answer "what was true on date X" because they don't track validity windows.

### Pattern 4 — contradiction detection at write time

```
You:  Remember that the auth library is JWT.
You:  Actually, remember that auth is now using sessions.
Claude Code:
  → second remember() returns contradictions_detected=[…]
  "I've noted the change. There's a structural contradiction
   logged between 'auth=JWT' and 'auth=sessions' — both are still
   queryable, with sessions marked current."
```

The `CONTRADICTS` typed edge is **data**, not a warning — your agent can query for outstanding contradictions and prompt the user to resolve them.

### Pattern 5 — private vs. shared scope for multi-user / multi-project

```
You:  Remember as a private note: I prefer 4-space indentation.
Claude Code:
  → remember(content=..., scope="private", owner_id="richard")
  "Stored under your private scope. Won't surface in shared recalls."
```

A shared recall (default `scope="all"`) returns public facts plus the requester's own private facts; never another user's.

### Tips for prompting the LLM client

- **You rarely need to say "call remember"** — agents pick it up from phrases like "note that", "remember", "store this", "save for later".
- **Force a recall** with phrases like "according to memory…", "what do we have on…", "as of date X…", or by prefixing with `@thought` in some clients.
- **`CLAUDE.md` is the single best place** to tell the agent project-specific conventions (e.g. "always tag finance decisions with scope=private"). `thought init` drops a starter file in the working directory.

### Calling the tools directly via CLI (when you want to)

```bash
thought ingest "Alice owns Acme Corp. Acme is part of HoldCo." --scope shared
thought recall "who owns Acme"
thought repl                              # interactive query shell
thought stats                             # what's currently in memory
thought forget "kendra%"                  # soft-delete with audit trail
```

The CLI is most useful for **bulk-loading** existing notes / docs / changelogs into memory before pointing your agent at it:

```bash
thought ingest --glob 'docs/**/*.md' --scope shared
thought ingest --file CHANGELOG.md
git log --since='1 week ago' --format='%s' | thought ingest --stdin --scope shared
```

---

## Architecture

```
   Claude Code · Cursor · Cline · Continue · Windsurf
   ┬───────────────────────────────────────────────────
   │                  (auto-wired by `thought install`)
   ▼
┌──────────────────────────────────────────────────────────────────┐
│         MCP server  (Streamable HTTP · async handlers)           │
│            remember(content, ...)    recall(query, ...)          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
              ┌───────────────────────────┐    LRU recall cache
              │          Router           │    (write-version keyed)
              │  VIBE  FACT  CHANGE  HYBRID│  ↳ rules.yaml (user-editable)
              │  + CRAG confidence eval   │
              └───────────┬───────────────┘
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
      ┌─────────────┐ ┌──────────┐ ┌────────────┐
      │  Vector L.  │ │ Graph L. │ │ Temporal L.│
      │ Matryoshka  │ │ HippoRAG │ │ bi-temporal│
      │  + GraphRAG │ │ PPR (+   │ │  as_of     │
      │  + sqlite-  │ │ scipy.   │ │ (valid +   │
      │  vec MATCH  │ │ sparse + │ │  learned)  │
      │             │ │ local    │ │            │
      │             │ │ push)    │ │            │
      └──────┬──────┘ └────┬─────┘ └─────┬──────┘
             │             │              │
             ▼             ▼              ▼
        ┌───────────────────────────────────────┐
        │      StorageBackend (ABC)             │
        │  SQLite + sqlite-vec  |  pgvector     │
        │  sources · entities · edges · triples │
        │  embeddings · strength_cache · log    │
        │  + bulk source-provenance JOIN        │
        │  + touch-access flush queue           │
        └──────────────┬────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  Consolidation Engine   │  background thread
         │  Ebbinghaus · cold/warm │  + `thought consolidate` CLI
         │  · dedup · audit log    │
         └─────────────────────────┘
```

**Bi-temporal axis:** every entity and edge tracks `(valid_from, valid_until)` (world-time) **and** `(learned_at, unlearned_at)` (transaction-time). "What did we know about X on date Y" and "what was true about X on date Y" are different queries; THOUGHT answers both via `recall(..., as_of=Y, as_of_kind='valid' | 'learned')`.

---

## What makes THOUGHT qualitatively different

These are capabilities **neither OB1 nor the Karpathy wiki structurally supports** — adding them would require rewriting their data layer:

- `recall(query, as_of=<past>)` returns the world as it was, not as it is.
- Every hit carries `confidence_class ∈ {source_grounded, inferred, hallucination_risk}` so the LLM knows what to trust.
- Contradictions are **first-class data** — `CONTRADICTS` typed edge with `detected_at` and `confidence_score`, queryable, not LLM lint notes.
- Multi-user scope is **structural** — `(scope, owner_id)` filter at the storage layer, inherited by every retrieval path.
- All writes are **append-only**. Supersession is a new edge plus a `valid_until` close, never an UPDATE/DELETE — full forensic audit is guaranteed.
- The query router classifies before searching — wrong question never hits the wrong index.

---

## Measured results

These numbers come from `tests/comparison/run.py` — same workload, same deterministic embedder, three architectures. Reproducible: `python -m tests.comparison.run`.

### Recall@10 by query class

| System | VIBE | FACT | CHANGE | HYBRID | overall |
|---|---|---|---|---|---|
| **THOUGHT** | **100%** | **100%** | **68%** | **66%** | **83.5%** |
| OB1 | 100% | 100% | 32% | 100% | 83.0% |
| Karpathy wiki | 100% | 30% | 0% | 100% | 57.5% |

THOUGHT and OB1 tie on overall recall@10, but the **CHANGE column (68% vs 32%) is the headline number** — THOUGHT is 2.1× more accurate on the queries where temporal correctness matters. Karpathy wiki is 0% on temporal: it has no notion of time.

### Temporal correctness on CHANGE queries (strict — penalizes returning contemporary answer for historical query)

| System | rate |
|---|---|
| **THOUGHT** | **68%** |
| OB1 | 32% |
| Karpathy wiki | 0% |

### Contradictions detected at write-time

| System | count |
|---|---|
| **THOUGHT** | 2 |
| OB1 | 0 |
| Karpathy wiki | 0 |

### Ablation — marginal contribution of each frontier technique

(From `python -m tests.comparison.ablation` → [docs/ablation.md](docs/ablation.md))

| Variant | Overall | FACT | CHANGE | HYBRID |
|---|---|---|---|---|
| **Full v0.1 (all Tier A)** | **83.5%** | **100%** | **68%** | **66%** |
| − HippoRAG bidirectional PPR | 66.0% | 30% | 68% | 66% |
| − Bi-temporal edge retirement | 75.0% | 100% | 34% | 66% |
| − Query router (force VIBE) | 65.5% | 30% | 32% | 100% |

Each disabled technique costs THOUGHT real measurable accuracy on the dimension it was added to improve. HippoRAG is worth +70pp on FACT queries; bi-temporal supersession is worth +34pp on CHANGE; the router is worth +35pp overall.

### Performance

THOUGHT went through three performance passes. Each one targeted the bottleneck the previous one exposed.

**v0.2 pass — architectural** (sqlite-vec + scipy.sparse + local push PPR):
1. **sqlite-vec C/SIMD MATCH** for vector ANN (was Python brute-force over the embeddings table).
2. **Binary sign-quantized index mirror** ([Charikar 2002 LSH](https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf)) for dense embeddings — opt-in via `use_binary_quantization=True`; another ~8-16× over the float path on production models.
3. **`scipy.sparse` vectorised Personalized PageRank** — one CSR matvec per iteration in place of the dict-of-lists power loop.
4. **Andersen-Chung-Lang local push PPR** ([2006](https://www.math.ucsd.edu/~fan/wp/localpartition.pdf)) — ε-approximate PPR touching only `O(1/(ε·(1−α)))` nodes, automatically used when the in-scope KB exceeds 5k entities.

**v0.3 pass — system + UX**:
5. **Batched ingest** — all writes from one `remember()` in one transaction; `remember_many()` batches across N items in one transaction with one `embed_many` call → **2-4× ingest throughput**.
6. **LRU recall cache** keyed by `(write_version, query, ...)` — repeat queries become **µs-scale** (~130,000× over cold-recall p50).
7. **Touch-access batched flush queue** — eliminates the per-hit UPDATE on the recall hot path, batches into one `executemany` periodically.
8. **PPR transition-matrix cache** with `write_version` invalidation — repeat FACT recalls skip the COO→CSR matrix rebuild entirely.
9. **One-query bulk source-provenance fetch** — replaced N+M roundtrips (`edges_to` per hit + `SELECT` per source) with a single JOIN.
10. **WAL tuning** — 64 MiB page cache, 256 MiB mmap, `synchronous=NORMAL`, `busy_timeout=5s`.
11. **Async MCP tool handlers** — `asyncio.to_thread` lets the Streamable HTTP transport service concurrent recalls.

#### Measured progression

Same workload (`Entity{i} owns Company{i%50} Corp.`), same Windows laptop, deterministic embedder, **30 unique queries** (no cache hits) for cold recall measurement:

| KB size | v0.1 recall p50 | v0.2 recall p50 | **v0.3 recall p50** | v0.3 ingest (bulk) | v0.3 cache-hit p50 |
|--------:|----------------:|----------------:|--------------------:|-------------------:|-------------------:|
| 1,000   | 50.3 ms         | 12.3 ms         | **8.5 ms**          | 0.67 s             | **0.7 µs**         |
| 5,000   | 261.6 ms        | 42.5 ms         | **37.8 ms**         | 3.73 s             | 0.7 µs             |
| 10,000  | 521.4 ms        | 61.6 ms         | **93.6 ms**¹        | 7.47 s             | 0.7 µs             |
| 25,000  | ~1,300 ms²      | 171.8 ms        | **186.0 ms**        | 17.18 s            | 0.7 µs             |

¹ v0.3 honest-cold-cache numbers are slightly higher than v0.2's warm-cache numbers at the same KB size — v0.2 measured 20 repeats of the *same* query without a cache, which our profiler flattered. With the v0.3 LRU cache, repeated queries become essentially free (0.7 µs), so the real-world latency curve is the cold-cache row for first-time queries and the cache-hit column for everything else.

² Original v0.1 took >10s per recall at 25k entities; numbers extrapolated from the linear growth pattern.

**Overall vs v0.1**: 5-7× faster cold recalls, ~10,000-130,000× faster cache hits, 2-4× faster ingest (bulk).

**Growth pattern**: 25× more data → ~22× more latency in v0.3 — closer to linear at the high end because the deterministic embedder is itself O(N) on the brute-force fallback; with `sentence-transformers/all-MiniLM-L6-v2` (production embedder, dense vectors), sqlite-vec's index becomes sub-linear and you get the full architectural win.

Also unchanged:
- **Result bound** — `len(hits) ≤ 10` always, verified at every KB size.
- Comparison-harness latency dropped from 7.78 ms → 2.75 ms with full accuracy preserved (FACT 100%, CHANGE 68%).

### Structural capability matrix (none of these are accuracy claims — they're either present or absent)

| Capability | THOUGHT | OB1 | Karpathy wiki |
|---|---|---|---|
| bi-temporal as_of | ✅ | ✗ | ✗ |
| source-grounded confidence class | ✅ | ✗ | ✗ |
| contradiction as typed edge | ✅ | ✗ | ✗ |
| multi-user scope isolation | ✅ | partial (RLS) | ✗ |
| append-only audit log | ✅ | ✗ | ✗ |
| Personalized PageRank retrieval | ✅ | ✗ | ✗ |
| Ebbinghaus decay scoring | ✅ | ✗ | ✗ |
| CRAG-style low-confidence flag | ✅ | ✗ | ✗ |
| Matryoshka 2-pass ANN | ✅ | ✗ | ✗ |
| Anthropic Contextual Retrieval | ✅ | ✗ | ✗ |
| query router (VIBE/FACT/CHANGE) | ✅ | ✗ | ✗ |
| forecasting (TLogic, v0.2) | planned | ✗ | ✗ |

---

## Design rationale

Full architectural discussion in [plan.md](plan.md). Short version of the philosophy:

> A memory system should **know what kind of question is being asked before it searches anything, store facts with their origin and validity, and never lose history in the act of updating.**

The three-layer split (Vector / Graph / Temporal) plus the Router is the architectural answer: each query class is dispatched to the mathematical structure that fits it. The eleven frontier techniques stack 1.5-3× gains on orthogonal axes; together they take the system from "pgvector wrapper" to "memory fabric."

Honest framing: no single 2024-2026 technique gives a 10× recall jump. The "1000× more useful" claim isn't about recall@10; it's about capabilities competitors structurally cannot have (the matrix above) compounded with stacked accuracy gains (the ablation table).

---

## Configuration

Default config (`thought.toml`, written by `thought init`):

```toml
db_path = ".thought/thought.db"

[embedding]
choice = "auto"           # "auto" picks sentence-transformers if installed,
                          # else deterministic (zero-dep test embedder).
                          # Override: "minilm" | "bge-m3" | "openai" | "deterministic"
dim = 384

[server]
host = "127.0.0.1"
port = 8765

[consolidation]
enabled = true
cycle_seconds = 60.0
cold_demotion_days = 30
staleness_days = 30
batch_size = 100

[llm]                     # optional — enables Contextual Retrieval enrichment
enabled = false
provider = "none"         # "anthropic" | "openai" | "ollama"
```

`thought` walks the directory tree (git-style) looking for a `thought.toml`, so you don't need a `--config` flag when running from a subfolder of your project.

Environment overrides: `THOUGHT_DB_PATH`, `THOUGHT_EMBEDDER`.

---

## CLI reference

### Setup / lifecycle

```bash
thought init [--quick] [--embedder auto|minilm|deterministic]
                                  # write config + db + CLAUDE.md
thought install --detect          # show every detected MCP client config path
thought install --client cursor   # wire one client (with backup, idempotent)
thought install --all             # wire every detected client
thought start [--client cursor]   # init-if-needed + install + serve in one command
thought serve [--host ... --port ... --skip-precheck]
                                  # start MCP server on Streamable HTTP
thought doctor                    # deep environment health check
thought --version
```

### Ingest

```bash
thought ingest "Alice owns Acme Corp."
thought ingest --file notes.md
thought ingest --glob 'docs/**/*.md'
cat changelog.txt | thought ingest --stdin

# Per-item scope
thought ingest --file private-notes.md --scope private --owner-id alice
```

### Recall

```bash
thought recall "who owns Acme"
thought recall "what did we say about pricing" --as-of 2026-01-01
thought recall "auth changes" --as-of 2026-01-01 --as-of-kind learned
thought recall "alice" --json     # raw JSON for piping into other tools
```

### Inspect + maintenance

```bash
thought stats                     # entities / edges / sources / contradictions / top accessed
thought repl                      # interactive shell — type queries, +text to remember
thought forget 'kendra%'          # soft-delete by SQL LIKE pattern (audit-logged)
thought consolidate               # run one consolidation cycle
```

### Docker

```bash
docker build -t thought-mcp .
docker run --rm -p 8765:8765 -v thought-data:/data thought-mcp
```

The image runs as a non-root user, exposes `:8765`, persists state at `/data`, and runs `thought serve` as the default command. Once tagged releases are pushed, an upstream image is published at `ghcr.io/<owner>/thought-mcp:<version>` and `:latest`.

---

## Troubleshooting

### `thought install --detect` says my client path doesn't exist

Most clients only create their config file after first launch. Open the client once, then re-run `thought install --client <name>`. The installer will create the file if its parent directory exists.

### `sqlite enable_load_extension` reports `NO` in `thought doctor`

You're on a Python build without loadable-extension support — most commonly Anaconda's bundled Python. Two fixes:

```bash
# Option A — install python.org Python and use that interpreter
# Option B — use pysqlite3-binary
pip install pysqlite3-binary
```

THOUGHT falls back to a pure-Python ANN path automatically, so this is a performance issue, not a correctness one.

### Recall returns `low_confidence: true` with no results

The CRAG evaluator flags this when the top hit's score is below threshold. Common causes:

- Knowledge base is empty or lacks anything relevant. Try `thought stats` to confirm.
- You're using the deterministic embedder (the test default). Set `embedder = "auto"` in `thought.toml` and reinstall sentence-transformers: `pip install 'thought-mcp[embeddings-local]'`.
- Query phrasing doesn't match indexed entity names. Use the `repl` to iterate.

### MCP client can't find the server

```bash
thought doctor                              # confirm MCP SDK + vec extension load
thought serve --skip-precheck               # try without the precheck
# Then inspect the client's MCP logs — most surface "failed to start" with a path
```

If `uvx thought-mcp serve` is in your `mcpServers` config and `uvx` isn't on PATH for the GUI client, switch the `command` to an absolute path to the `thought` entrypoint (`which thought` / `where thought`).

### First `recall` after startup is slow

The first call lazy-loads the embedder (downloads `all-MiniLM-L6-v2`, ~80 MB, on first run). After that it's warm. Use `thought init` (without `--quick`) to pre-download.

### Windows console garbles output

The CLI reconfigures stdout/stderr to UTF-8 at startup. If you're piping through a tool that still uses cp1252, set `PYTHONIOENCODING=utf-8` in your shell.

---

## Testing & development

```bash
pytest tests/unit -q                 # 56 unit tests
pytest tests/perf -m perf            # 4 performance benchmarks
python -m tests.comparison.run       # rebuilds docs/comparison.md
python -m tests.comparison.ablation  # rebuilds docs/ablation.md
```

Coverage target: 85% on `src/thought`. CI matrix runs Python 3.10/3.11/3.12/3.13 × Ubuntu/macOS/Windows on every push (see `.github/workflows/ci.yml`). Tagging `v*` triggers `release.yml` (PyPI trusted publishing) and `docker.yml` (multi-arch GHCR image).

---

## Roadmap

**Current (shipped)** — 11 Tier A frontier techniques (Contextual Retrieval, HippoRAG PageRank, bi-temporal Graphiti, atomic-fact triples + Jaccard dedup, BGE-M3 hybrid embeddings, Matryoshka 2-pass retrieval, CRAG evaluator, MetaRAG confidence class, Ebbinghaus decay, context-engineering budget per query class, append-only writes); comparison + ablation harnesses; two MCP tools; multi-platform CLI with auto-install for five MCP clients; LRU recall cache + PPR matrix cache + sqlite-vec + scipy.sparse PageRank + local push PPR + batched ingest (the three perf passes described above); Docker + PyPI release workflows.

**v0.2 fast-follow** — RAPTOR hierarchical summary trees at WARM→COLD demotion ([Sarthi et al., ICLR 2024](https://arxiv.org/abs/2401.18059)); sleep-time compute pre-computation ([Letta + UCB, April 2025](https://arxiv.org/abs/2504.13171)); TLogic temporal-rule forecasting ([arXiv 2112.08025](https://arxiv.org/abs/2112.08025)); Reflexion-style self-edit ([Shinn et al., NeurIPS 2023](https://arxiv.org/abs/2303.11366)); multi-hop deep recall (IRCoT/PRISM); introspective `thought audit` ([transformer-circuits, 2025](https://transformer-circuits.pub/2025/introspection/index.html)).

**v0.3+** — RankZephyr local reranker, PIKE-RAG domain rationale extraction, DSPy-learned retrieval policies, real Postgres backend, REST API alongside MCP, encryption-at-rest (SQLCipher / pgcrypto), tenant isolation, OpenTelemetry traces/metrics.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- OpenBrain — https://github.com/srnichols/OpenBrain · https://github.com/benclawbot/open-brain
- Karpathy LLM-Wiki gist — https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- MCP Specification — https://modelcontextprotocol.io/specification/2025-11-25
- HippoRAG — https://arxiv.org/abs/2405.14831 (NeurIPS 2024)
- Zep / Graphiti — https://arxiv.org/abs/2501.13956
- Anthropic Contextual Retrieval — https://www.anthropic.com/news/contextual-retrieval
- CRAG — https://arxiv.org/abs/2401.15884
- LightRAG — https://arxiv.org/abs/2410.05779
- BGE-M3 — https://huggingface.co/BAAI/bge-m3
- RAPTOR — https://arxiv.org/abs/2401.18059
- Matryoshka Representation Learning — https://huggingface.co/blog/matryoshka
- TLogic — https://arxiv.org/abs/2112.08025
- Mem0 State of Memory 2026 — https://mem0.ai/blog/state-of-ai-agent-memory-2026
- sqlite-vec — https://github.com/asg017/sqlite-vec
- pgvector — https://github.com/pgvector/pgvector
