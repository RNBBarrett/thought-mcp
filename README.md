# THOUGHT

[![PyPI](https://img.shields.io/pypi/v/thought-mcp.svg)](https://pypi.org/project/thought-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/thought-mcp.svg)](https://pypi.org/project/thought-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/RNBBarrett/thought-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/RNBBarrett/thought-mcp/actions/workflows/ci.yml)
[![Docker](https://github.com/RNBBarrett/thought-mcp/actions/workflows/docker.yml/badge.svg)](https://github.com/RNBBarrett/thought-mcp/actions/workflows/docker.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-thought--mcp-blue?logo=docker)](https://github.com/RNBBarrett/thought-mcp/pkgs/container/thought-mcp)

**T**emporal **H**ierarchical **O**bject **U**nion & **G**raph **H**ybrid **T**oolkit — a local MCP memory server that gives any LLM a persistent, auditable memory fabric on your own machine.

> OB1 stores your thoughts. Karpathy's wiki compiles your knowledge. **THOUGHT** remembers with provenance, understands relationships, detects contradictions, never forgets what used to be true — and routes every query to the right mathematical structure before touching a single byte of data.

---

## ✨ New in v0.4 — Lifecycle commands + Local LLMs + Cypher + Ask

Four headline capabilities:

1. **`thought db`** — `size` / `flush` / `backup <file>` / `load <file>` / `inspect <file>`. With `--before` / `--since` / `--time-axis` to slice the KB by date. Always auto-backs-up before destructive operations.
2. **Local-LLM integration** — `embedder.choice = "ollama"` / `"lmstudio"` / `"openai-compat"`. `thought ollama-setup` and `thought lmstudio-setup` discover models and write your `thought.toml`. Same provider switch covers auto-write's `--mode extract` so durable-fact extraction runs locally too.
3. **Cypher query layer** — `thought query "MATCH (p:PERSON)-[:WORKS_AT]->(o) RETURN p.name, o.name"`. Saved views (`thought view save adidas_seattle "..."`) turn queries into named, re-evaluating memory constructs. `thought schema` shows what's queryable. `AS_OF` for time-travel.
4. **`thought ask` in English** — natural-language → Cypher via your configured LLM (Anthropic / Ollama / LM Studio / OpenAI-compat). Validates against the parser; bad translations degrade to plain `recall` so you always get something. `--save-as <name>` harvests good translations into saved views.

```bash
# Backup before doing something risky:
thought db backup ./snap.db && thought db flush --since 2026-01-01 --time-axis valid --yes

# Run with Ollama (no API keys, fully local):
ollama serve && ollama pull nomic-embed-text
thought ollama-setup --write

# Ask in English (uses whichever LLM is configured):
thought ask "who at Acme also prefers Adidas?" --explain --save-as adidas_acme

# Or write Cypher directly:
thought query "MATCH (p:PERSON)-[:PREFERS]->(:CONCEPT {name:'Adidas'}) RETURN p.name"
```

See [CHANGELOG.md](CHANGELOG.md#040--2026-05-15--db-lifecycle--local-llms--cypher--ask) for the full v0.4 list. v0.3's auto-memory hooks (below) and v0.2's code-vertical surface are unchanged — v0.4 is purely additive.

---

## ✨ Previously in v0.3 — Auto-memory + topic browsing

**Memory that captures + retrieves itself, plus a way to see what's in there.** Wire two Claude Code hooks into your project, restart, and from that point on every assistant turn is auto-ingested and every user prompt is pre-recalled against the KB — no explicit `remember` / `recall` calls needed. Discoverability: a new `thought topics` command lists what the KB *contains* (people, places, functions, concepts…) and `thought browse <topic>` drills into a specific area.

```bash
# One command wires both hooks into Claude Code (project-scoped settings.json).
thought hook install --both

# What's in my memory right now?
thought topics
#  type           count  examples
#  CONCEPT        89     Acme, Adidas, dessert
#  function       425    personalized_pagerank, recall, remember
#  PERSON         47     Alice, Bob, Dana
#  ORGANIZATION   12     Acme Corp, Beta, OpenAI

# Drill into a topic (or an entity name)
thought browse dessert --depth 2
#  #   type     entity   score
#  1   CONCEPT  donut    0.42
#  2   CONCEPT  cake     0.31
#  3   CONCEPT  pastry   0.18
```

The auto-recall hook uses the `additionalContext` field on Claude Code's `UserPromptSubmit` event (10k-char cap, low-confidence gated so it doesn't pollute context). The auto-write hook runs on `Stop`, picks the last user + last assistant turns out of the session transcript, and pipes them into the existing ingest pipeline — content-sha256 idempotency + Jaccard dedup keep replays from doubling the KB. Optional `--mode extract` routes each turn through Haiku first for higher-signal facts (~$0.001/turn).

Two new MCP tools are also exposed for agents that want the discoverability primitives directly: **`mcp__thought__list_topics`** and **`mcp__thought__browse_topic`**.

What's in the bi-temporal model already does the heavy lifting on contradictions — if you say "I prefer Adidas" on Monday and "I prefer Nike" on Friday, the second auto-write supersedes the first via a `CONTRADICTS` + `SUPERSEDES` edge pair, exactly like Zep's temporal validity windows from the recent literature.

```bash
# Per-hook install if you'd rather pick them à la carte:
thought hook install --recall          # auto-recall only
thought hook install --write           # auto-write only
thought hook install --write --scope user  # global to your home dir
```

See [CHANGELOG.md](CHANGELOG.md) for the full v0.3 list. v0.2's code-vertical surface (below) and v0.1's horizontal-memory surface are unchanged — v0.3 is purely additive.

---

## ✨ Previously in v0.2 — Memory for AI coding agents

v0.2 specialises the same architecture for the workflow with the strongest natural fit: **AI-assisted coding**. THOUGHT now parses your source with tree-sitter, builds a real function-call graph as typed edges, and stamps every fact with its git commit. The bi-temporal `as_of` queries you already had now answer *"what did the codebase look like at commit X?"* for free.

```bash
thought ingest-code src/                      # tree-sitter ingest, multi-language
thought ingest-git . --mode full              # stamp every commit
thought callers GraphLayer.personalized_pagerank
#  #  score    type    entity                              file
#  1  0.0132   method  Dispatcher._dispatch_code           Dispatcher
#  2  0.0130   method  Dispatcher._dispatch_fact           Dispatcher
#  3  0.0122   method  CodeLayer.impact_set                CodeLayer
#  4  0.0110   method  CodeLayer.callers_of                CodeLayer
thought impact authenticate_user              # what's affected if I change this?
thought diff --from v1.0 --to HEAD            # set diff between two commits
```

**Real measurement on this codebase**: 38 files → 425 entities → 575 CALLS edges in **<250 ms**. The killer-demo query *"who calls GraphLayer.personalized_pagerank"* returns the four real callers ranked by Personalized PageRank in **60 ms** on a 1086-edge graph.

What's new:

- **AST-aware ingest** via tree-sitter — Python + TypeScript / JavaScript out of the box, multi-language plugin shape for the rest.
- **Function-call-graph edges** — `CALLS`, `IMPORTS`, `INHERITS_FROM`, `OVERRIDES`, `DEFINES` as typed edges. The Graph Layer's HippoRAG-style PageRank then powers ranked callers / impact-set queries.
- **Git-history stamping** — every entity carries `code_commit_sha`. `thought diff --from <sha1> --to <sha2>` returns the set difference of functions between two commits.
- **New Router CODE class** — natural-language queries like *"who calls authenticate_user"* route through the call-graph machinery automatically.
- **5 new CLI commands**: `ingest-code`, `ingest-git`, `callers`, `impact`, `diff`.

See [CHANGELOG.md](CHANGELOG.md) for the full v0.2 list. The v0.1 horizontal-memory surface below is unchanged — v0.2 is purely additive.

---

## What is THOUGHT?

THOUGHT is a **memory server for LLMs**. You install it on your machine, wire it into your AI coding assistant (Claude Code, Cursor, Cline, Continue, Windsurf), and now your assistant has a brain that persists across conversations and across projects.

Everything runs **locally** — your memory is a single SQLite file on your laptop. No cloud, no account, no sync service, no API key.

### The problem it solves

Out of the box, AI coding assistants have goldfish memory. Every new conversation starts blank. If you told it last week to *"always use Postgres for v2 features,"* you'll be telling it again today. If you decided in March that *"the auth module is being rewritten,"* by April that context is gone.

Existing fixes don't really solve this — they trade one problem for another:

| Common workaround | What goes wrong |
|---|---|
| **Stuff context into your system prompt** | You hit token limits fast, and the model can't tell what's current vs. obsolete. |
| **Cloud memory** (ChatGPT, Claude Projects) | Locked to one vendor, no audit log, can't query *"as of last week,"* no contradiction handling. |
| **RAG over your notes** (mem0, Letta, …) | Stores facts as flat vectors. No relationships between facts, no time tracking, no provenance, no notion of "this used to be true." |
| **An LLM-maintained Markdown wiki** (Karpathy's gist) | Lossy by design (the LLM summarises everything), grows linearly, no semantic search, no temporal queries. |

THOUGHT fixes the structural issues, not just the symptoms.

### What you get

Once installed, your AI assistant gains two new tools it can call automatically when the conversation implies it:

- **`remember(content)`** — *"note that we decided X."* THOUGHT extracts the entities and relationships, embeds them for similarity search, and links everything to its source so you can audit later.
- **`recall(query)`** — *"what did we decide about X?"* THOUGHT figures out what kind of question you asked, routes it to the right retrieval strategy, and returns at most 10 hits — each tagged with how trustworthy it is.

You can also drive it from your terminal (CLI) or use the Python API directly.

### Why it's better than existing solutions

The TL;DR, in plain English:

- **It knows when facts changed.** Every fact carries two timestamps: when it was true in the world, and when the system learned it. *"What did we say about pricing on Jan 15?"* actually works — even if pricing changed on Feb 3.

- **It tracks how facts relate.** Functions, classes, people, projects, decisions — they're all entities in a typed graph (CALLS, OWNS, INHERITS_FROM, CONTRADICTS, …). Asking *"who calls `authenticate_user`?"* is a real graph query, not a fuzzy text match.

- **It refuses to hallucinate relationships.** Every edge has a mandatory pointer back to the source document that produced it. If a fact has no source, it doesn't exist. No more *"the model invented a connection that was never in the data."*

- **It surfaces contradictions instead of silently overwriting.** When you say *"auth is now using sessions"* after previously saying *"auth is JWT,"* both facts stay. A `CONTRADICTS` edge is created. `recall` can then answer *"what facts about auth are currently disputed?"*

- **It picks the right retrieval method per question.** Fuzzy associative queries hit vector similarity. Relationship queries hit graph traversal. Time-travel queries hit the temporal layer. The wrong question never hits the wrong index.

- **It bounds output.** No matter how big the knowledge base gets, `recall` returns at most 10 hits. Your context window doesn't get blown up by a runaway retrieval.

- **It's append-only.** Nothing is ever deleted. When facts go stale, they're retired (their validity window closes), not erased. Full forensic audit of every change.

- **It's natively multi-user.** `scope='shared'` for project-wide facts, `scope='private'` with `owner_id` for personal notes. Five devs on one repo each get their own private memory plus a shared common pool.

Plus **eleven cutting-edge retrieval techniques** from 2024–2026 literature (Anthropic Contextual Retrieval, HippoRAG-style PageRank, bi-temporal Graphiti, CRAG, MetaRAG confidence, …) stacked on top — see the [Frontier techniques](#frontier-techniques-incorporated-with-credits) section below for the full list with citations.

The technical capability matrix vs. the closest comparable systems:

| | OB1 (pgvector) | Karpathy LLM-Wiki | **THOUGHT** |
|---|---|---|---|
| Relationship logic | flat rows | flat markdown | **typed graph edges** |
| Temporal awareness | none | none | **bi-temporal (world-time + learned-time)** |
| Provenance | informal tag | informal citation | **mandatory `source_ref` on every edge** |
| Multi-user | RLS bolted on | single-user | **native two-zone graph** |
| Query routing | always vector | always inject | **VIBE / FACT / CHANGE / CODE / HYBRID router** |
| Contradiction model | absent | LLM lint only | **`CONTRADICTS` typed edge, write-time** |
| Bounded result size | unbounded | unbounded | **≤10 enforced** |

### What THOUGHT is **not**

- **Not a cloud service.** Everything runs locally. No data leaves your machine.
- **Not a vector DB replacement.** It uses one (sqlite-vec by default, pgvector optional), but adds the graph + temporal layers on top.
- **Not a fine-tuner.** It doesn't change your model. It changes what your model can *remember*.
- **Not retrieval-quality magic.** No single 10× win exists in 2024–2026 LLM-retrieval literature; THOUGHT compounds several 1.5-3× gains across orthogonal dimensions. Expect 2-3× better recall on questions that actually need the typed graph or temporal layer; expect roughly parity on pure-vibe semantic queries.

---

## How to use THOUGHT

This section walks through everything from install to advanced workflows, with explanations of *why* each step exists. If you just want the 30-second version, skip to **[Quickstart](#quickstart)**.

### Install

Three ways. Pick one:

```bash
# Option 1 (recommended) — full bundle, everything you'll use
pip install 'thought-mcp[all]'

# Option 2 — minimal: CLI + MCP server only (no production embeddings)
pip install thought-mcp

# Option 3 — zero install: uvx fetches it on demand
uvx thought-mcp install --client cursor
```

`uvx` is what the MCP client configs use internally, so option 3 is fine if you don't want a global install. After install, verify with:

```bash
thought doctor
```

You should see all green. Any red items will print the exact command to fix them.

### Quickstart

The one-line happy path for connecting THOUGHT to your AI client:

```bash
thought start --client cursor   # or claude-code, cline, continue, windsurf
```

Then **restart your AI client** (close every window, reopen). Done. The next conversation will have the `remember` and `recall` tools available.

If you're not sure which client to pick, run `thought install --detect` first — it shows every supported client's config path and whether it's installed on your machine.

### What `thought start` actually does

Knowing what changed makes troubleshooting easier later:

1. **Creates the SQLite database** at `.thought/thought.db` in your current directory. This is your memory. Back it up like any database.
2. **Writes `CLAUDE.md`** in your current directory. This tells your AI assistant how the memory tools work and when to use them. You can edit it to add project-specific conventions like *"always tag finance decisions with scope=private."*
3. **Writes `thought.toml`** with sensible defaults. Most people never need to touch it.
4. **Updates your AI client's MCP config** to register `thought` as a server. The previous config is backed up to `<config>.thought.bak`.
5. **Starts the MCP server** in the background, listening on `127.0.0.1:8765`.

After your AI client restarts, it discovers `thought` and gains the two new tools.

### Day-to-day usage — letting your AI use the memory

Once wired up, your AI assistant calls `remember` and `recall` *automatically* when the conversation implies it. You usually don't need to invoke them explicitly. Here's what that looks like:

**Telling it to remember something:**

```
You:  We've decided to use Postgres for the v2 release.
AI:   Got it. [calls remember internally]
      Stored. We can recall this with "database plan" or similar later.
```

The AI parses out the facts (v2 → Postgres), links them to the conversation as the source, and writes them to memory. You don't see the tool call.

**Asking about something it remembered:**

```
You:  What's our database plan?
AI:   [calls recall internally]
      Based on what's in memory: for v2 we're using Postgres
      (source_grounded, recorded 3 days ago).
```

The AI also sees how trustworthy each hit is via its `confidence_class` (`source_grounded`, `inferred`, or `hallucination_risk`). If a hit comes back as hallucination_risk, a well-prompted AI will flag the uncertainty rather than presenting it as fact.

**Time travel — what was true at some past date:**

```
You:  What was our pricing on January 15?
AI:   [calls recall with as_of="2026-01-15"]
      On Jan 15: Free / Pro $19 / Team $99.
      Pricing changed on Feb 3 to Free / Pro $25 / Team $129, so the
      current prices are different.
```

This is THOUGHT's most distinctive capability. Existing memory tools structurally can't answer this — they overwrite old facts. THOUGHT keeps both, marks the older one retired, and `as_of` queries see the world as it was.

**Catching contradictions automatically:**

```
You:  Remember that auth is now using sessions.
AI:   Noted. [calls remember]
      Heads up: a previously-stored fact says "auth is JWT."
      Both are now in memory, linked by a CONTRADICTS edge.
      Want me to retire the JWT fact?
```

The AI sees the conflict at write time and prompts you. The JWT fact isn't deleted — it's marked as superseded but still queryable for audit.

**Private vs. shared scope (multi-user / multi-project):**

```
You:  Remember as a private note: I prefer 4-space indentation.
AI:   Stored in your private scope. Won't surface in shared recalls.
```

Use `scope='private'` for personal preferences. Use `scope='shared'` for project decisions everyone on the team should see. A shared recall returns public facts plus the requester's own private facts; never another user's.

### How to nudge the AI when it doesn't reach for memory

If your AI is being lazy and skipping `recall`, try phrases like:

- *"According to memory..."*
- *"What do we have on..."*
- *"As of last week, ..."*
- *"Check memory for..."*
- *"@thought what's our..."* (in clients that support tool-prefix syntax)

To insist on storing something:

- *"Note this down: ..."*
- *"Remember that..."*
- *"Store this for later: ..."*
- *"Add to memory: ..."*

The single highest-leverage thing is the **`CLAUDE.md`** file that `thought init` drops in your project. Edit it to add project-specific conventions. The AI reads it on every session start, so rules like *"always remember architectural decisions, never remember code snippets"* are honored consistently.

### Auto-memory: capture + recall without thinking about it

By default, THOUGHT only writes when the agent calls `remember` and only reads when it calls `recall`. That works, but it puts the burden on the agent to *remember to remember*. In v0.3 you can wire two Claude Code hooks that do it for you:

```bash
# In your project root:
thought hook install --both           # installs into ./.claude/settings.json
# (or --recall / --write to pick one)

# Restart Claude Code.
```

#### What you'll experience after `--both` is installed

| Trigger | What happens | What you'll notice |
|---|---|---|
| Every user prompt | `recall(prompt)` runs; top-5 hits inject into the next turn's context (≤8 KB). **Gated on `low_confidence`** — if there's nothing relevant, nothing gets injected. | The agent "already knows" facts from earlier sessions without you mentioning them. Silent miss when truly unrelated — no context pollution. |
| Every assistant turn ends | Last user + last assistant turn ingested via the pipeline: **sha256 idempotency** + **Jaccard dedup** absorb replays, so the same conversation can run twice without bloating the KB. | `thought stats` slowly grows entity/edge counts as you work. |
| Contradictions appear | When auto-write sees a fact that conflicts with one already in the KB on a `unique_predicate` (PREFERS / WORKS_AT / OWNS / REPORTS_TO by default), it writes the new fact and adds **`CONTRADICTS` + `SUPERSEDES`** edges. The new fact "wins"; the old one is preserved with `valid_until` set. | Updated preferences and facts work cleanly — you can say "I now prefer Nike" and the KB knows Adidas is historical. |
| Time passes | The **consolidation engine** runs on a background thread: Ebbinghaus decay scoring, cold-tier demotion of unused entities, duplicate merging. | The KB stays focused; rarely-accessed facts gracefully fade in scoring without being deleted. |

#### Day-1 onboarding tip — seed the KB so you'll *feel* it working

The recall hook is silent when there's nothing relevant. On day 1 of a fresh KB that means *every* turn looks like nothing's happening. Seed it with a handful of facts and you'll feel the difference within minutes:

```bash
# At a terminal in your project:
thought ingest "I prefer hand-rolled SQL over ORMs for performance-sensitive code."
thought ingest "This project uses Python 3.12, ruff, pytest, mypy strict."
thought ingest "We deploy to AWS Lambda behind API Gateway; cold-start is the perf bottleneck."
# ...4-5 things that summarise your work context.

thought stats        # confirm entities grew
```

Now restart Claude Code. The next time you ask something even tangentially related (*"how should I optimise this query?"*), the recall hook will surface the "hand-rolled SQL" preference into context and the agent will weigh it.

#### Quality upgrade: `--mode extract` (highly recommended if you have any LLM configured)

The default `--mode raw` ingests transcripts wholesale and leans on the ingest pipeline's fact-extractor. It works, but the KB accumulates some noise (in-progress reasoning, hedging, "let me think" filler). The fix:

```toml
# In thought.toml, configure any LLM provider once:
[llm]
provider = "ollama"            # or "anthropic" / "lmstudio" / "openai-compat"
model = "mistral"              # or "claude-haiku-4-5-20251001" / etc.
base_url = "http://localhost:11434"
```

Then in `.claude/settings.json` change the Stop hook command:

```json
"command": "thought hook write --mode extract"
```

Each assistant turn is now routed through your chosen LLM first to distill *durable facts* before ingest. Costs are small:

| Provider | Approx. per-turn cost |
|---|---|
| Anthropic Haiku | ~$0.001/turn |
| **Ollama / LM Studio (local)** | **$0 — fully local** |
| OpenAI gpt-4o-mini | ~$0.0005/turn |
| OpenAI-compat (vLLM/llama.cpp local) | $0 |

The KB stays much cleaner — only third-person factual statements survive, not conversational filler. Strongly recommended once you've decided which LLM you're committing to.

#### Project scope vs. user scope — which one do you want?

`thought hook install` has two scopes, and the right pick depends on whether you want a per-project memory or one memory shared across everything:

```bash
thought hook install --both                   # default: --scope project
thought hook install --both --scope user      # global: every Claude Code session
```

| Scope | Settings file | What it covers | When to pick it |
|---|---|---|---|
| `project` (default) | `./.claude/settings.json` | Only Claude Code sessions launched in *this* directory | You want a project-local KB; you have a `thought.toml` here and you don't want this project's memory mixing with others. |
| `user` | `~/.claude/settings.json` | *Every* Claude Code session this user runs, anywhere | You want a single lifelong KB that follows you across projects, and you're OK with all projects writing into the same memory. |

Honest caveat for `--scope user`: the hooks resolve `thought.toml` (and therefore the SQLite db path) relative to whatever cwd Claude Code is launched in. If you start a session in a directory without a `thought.toml`, the hook still runs — but the ingest pipeline will silently fail to land anything because there's no configured db. The hook returns exit-code 0 so it doesn't break your turn (just emits a one-line warning to Claude Code's MCP log).

The clean "one shared KB across every project" recipe:

```powershell
# 1. Pick an absolute path for your lifelong memory.
$env:THOUGHT_DB_PATH = "$env:USERPROFILE\.thought\global.db"
# (Put that line in your $PROFILE on Windows or ~/.zshrc on macOS.)

# 2. Initialize once at that path.
thought init --db-path "$env:THOUGHT_DB_PATH" --no-claude-md

# 3. Install hooks globally.
thought hook install --both --scope user
```

Now every Claude Code session — in any directory — auto-reads + auto-writes the same global KB. Pair it with `thought stats` from any terminal to see what's accumulated.

To remove hooks later: delete the relevant block from `.claude/settings.json` (project or user), or rerun `thought hook install` with different flags after first removing the existing entries (the installer is additive — it doesn't strip stale entries).

#### Multi-user safety + opt-out

Auto-write defaults to `scope=private` so multi-user deployments don't accidentally cross-pollinate user-specific facts. To opt out for a single turn, just don't trigger the hooks (e.g. with a `--no-hooks` flag in Claude Code), or uninstall the hooks entirely by deleting the block from `.claude/settings.json`.

### Browsing what's in your memory

Two new commands answer the "what does the agent actually know?" question without writing a query:

```bash
# What kinds of facts are stored?
thought topics
#  type           count  examples
#  CONCEPT        89     Acme, Adidas, dessert
#  function       425    personalized_pagerank, recall, remember
#  PERSON         47     Alice, Bob, Dana
#  ORGANIZATION   12     Acme Corp, Beta, OpenAI

# Drill into a type (everything of that kind)
thought browse CONCEPT --limit 20

# Drill into a specific anchor (PPR-ranked neighbourhood)
thought browse Acme --depth 2

# JSON output for scripting
thought topics --json
thought browse Alice --json
```

The agent itself can use the same surface via `mcp__thought__list_topics` and `mcp__thought__browse_topic` — useful in prompts like *"first survey what we already know about authentication, then suggest changes"*.

### Managing your KB (v0.4)

THOUGHT now has first-class lifecycle handles for the KB. Three walkthroughs:

**"I just want to start over."**
```bash
thought db backup ./before.db        # snapshot before
thought db flush --yes               # wipe
# Test some new workflow...
thought db load ./before.db --yes    # roll back if it went poorly
```

**"Trim old facts I don't need anymore."**
```bash
# See what'd be affected first (peek at a date-bounded snapshot):
thought db backup ./old.db --before 2026-01-01 --time-axis valid
thought db inspect ./old.db --schema
# Confirmed it's what we want; actually flush:
thought db flush --before 2026-01-01 --time-axis valid --yes
```

The three time axes:
- `--time-axis created` (default): when the row was *inserted* (transaction time)
- `--time-axis valid` *(usually what you want)*: when the fact became true (world time)
- `--time-axis learned` : when the system *learned* it (alias of created in most setups)

**"Peek inside a backup before loading it."**
```bash
thought db inspect ./snap.db --schema
thought db query ./snap.db "MATCH (p:PERSON) RETURN count(p)"    # coming via the Cypher layer
thought db load ./snap.db --merge --since 2026-01-01 --time-axis valid
```

`db load --merge` is non-destructive — INSERT-OR-IGNORE based on the existing entity identity, so re-running is a no-op. Default `db load` (no `--merge`) replaces the active DB, auto-backing up the current one to `<db>.bak.<timestamp>` first.

### Running with local models — Ollama / LM Studio / any OpenAI-compatible server (v0.4)

Zero API cost, fully offline. Same provider switch covers embeddings *and* the `--mode extract` path for auto-write, so your durable-fact extraction also runs locally.

#### Ollama — step by step

**1. Install Ollama** if you don't have it. Download from [ollama.com/download](https://ollama.com/download) (macOS / Linux one-line installer / Windows MSI).

**2. Start the daemon** (it auto-starts on most installs, but in a fresh shell):

```bash
ollama serve                                  # blocks; leave it running in its own terminal
# Test it's up:
curl http://localhost:11434/api/tags
# {"models":[]}   ← daemon healthy, no models pulled yet
```

**3. Pull an embedding model.** Pick *one* of these (matters because the `dim` must match in your config):

| Model | Dim | Size | When to pick |
|---|---|---|---|
| `nomic-embed-text` | **768** | 274 MB | **Default — good balance of quality + speed** |
| `mxbai-embed-large` | 1024 | 670 MB | Higher quality, slower; English-centric |
| `all-minilm` | 384 | 45 MB | Smallest; matches sentence-transformers' default dim so you can A/B swap |
| `bge-m3` | 1024 | 1.2 GB | Multilingual; pick if your KB has non-English content |

```bash
ollama pull nomic-embed-text                  # ~30 s on a fast connection
```

**4. (Optional) Pull a chat model** if you want auto-write's `--mode extract` path to work locally:

| Chat model | Size | Speed on CPU | Good for `--mode extract`? |
|---|---|---|---|
| `mistral:7b` | 4.4 GB | ~5 tok/s | ✅ Reliable, fast enough |
| `llama3.2:3b` | 2.0 GB | ~10 tok/s | ✅ Smaller, recommended for low-RAM boxes |
| `qwen2.5:7b` | 4.7 GB | ~5 tok/s | ✅ Stronger reasoning than mistral |

```bash
ollama pull llama3.2:3b                       # for --mode extract
```

**5. Wire it into THOUGHT** — one command writes `thought.toml`:

```bash
thought ollama-setup --write                  # auto-detects the best embedding model
```

That writes a config like:

```toml
db_path = ".thought/thought.db"

[embedding]
choice = "ollama"
dim = 768
ollama_host = "http://localhost:11434"
ollama_model = "nomic-embed-text"

[llm]
enabled = true
provider = "ollama"
model = "llama3.2:3b"
base_url = "http://localhost:11434"
```

Edit `model` under `[llm]` if you pulled a different chat model.

**6. Verify the round-trip:**

```bash
thought init --quick --no-claude-md
thought ingest "Alice owns Acme Corp."        # uses Ollama for embedding
thought recall "alice"                        # should return Alice as hit #1
```

**7. Switch auto-write to extract mode** (optional, recommended once it's working):

In `.claude/settings.json` change the Stop hook command:

```json
"command": "thought hook write --mode extract"
```

Each assistant turn is now routed through `llama3.2:3b` (or whichever chat model you set) to distill durable facts before ingest. Zero API cost.

**Troubleshooting:**

- *"Ollama daemon unreachable at http://localhost:11434"* → daemon isn't running. `ollama serve` in another terminal.
- *"model `X` returned 1024-d embeddings but [embedding] dim = 768"* → you pulled a different model than the config expects. Fix one or the other: either edit `dim = 1024` and `ollama_model = "mxbai-embed-large"` in `thought.toml`, or `ollama pull nomic-embed-text` to match the existing config.
- *Recall returns nothing even after ingest* → the deterministic / hash embedding might be picking up. Verify your config: `cat thought.toml | grep choice` should show `"ollama"`. If it shows `"auto"` and sentence-transformers is installed, that wins by default — change to explicit `"ollama"`.
- *Slow recall the first time* → Ollama loads the model into memory on first call (~2-5 s). Subsequent calls are fast. Pre-warm with `curl -d '{"model":"nomic-embed-text","input":"hi"}' http://localhost:11434/api/embed`.

#### LM Studio — step by step

**1. Install LM Studio** if you don't have it. Download from [lmstudio.ai](https://lmstudio.ai/) (macOS / Windows / Linux). Open the app.

**2. Pull an embedding model from inside LM Studio:**

- Click the **Search** icon (left sidebar, looks like a magnifying glass)
- Search for `nomic-embed-text` (or one of the alternatives below)
- Click **Download** on the model card

| Model in LM Studio catalog | Dim | When to pick |
|---|---|---|
| `nomic-embed-text-v1.5` | **768** | **Default — same as Ollama's choice** |
| `mxbai-embed-large-v1` | 1024 | Higher quality |
| `bge-m3` | 1024 | Multilingual |

**3. Start LM Studio's local server:**

- Click the **Developer / Local Server** tab (left sidebar, looks like ⌘)
- Click the **green Start Server** button at the top
- Confirm the address shown is `http://localhost:1234/v1` (default)
- In the "Embedding Model" dropdown, pick the model you downloaded
- Optionally also load a chat model (for `--mode extract`) in the "Model to Load" dropdown — `mistral-7b-instruct` or `phi-3.5-mini-instruct` are good defaults

**4. Verify the server is up:**

```bash
curl http://localhost:1234/v1/models
# Should return JSON listing nomic-embed-text-v1.5 (and any chat model you loaded)
```

**5. Wire it into THOUGHT:**

```bash
thought lmstudio-setup --write
```

That writes a config like:

```toml
db_path = ".thought/thought.db"

[embedding]
choice = "lmstudio"
dim = 768
lmstudio_url = "http://localhost:1234/v1"
lmstudio_model = "nomic-embed-text-v1.5"

[llm]
enabled = true
provider = "lmstudio"
model = "mistral-7b-instruct"     # edit to whatever chat model you loaded
base_url = "http://localhost:1234/v1"
```

**6. Verify the round-trip:**

```bash
thought init --quick --no-claude-md
thought ingest "Bob runs the warehouse."
thought recall "bob"
```

**7. Switch auto-write to extract mode** (optional):

Edit `.claude/settings.json`, change the Stop hook to `"command": "thought hook write --mode extract"`. Each turn now routes through LM Studio's loaded chat model. Zero API cost.

**Troubleshooting:**

- *"LM Studio unreachable at http://localhost:1234/v1"* → the local server isn't running. Open LM Studio, **Developer** tab, click **Start Server**.
- *"no models loaded"* → you downloaded a model but didn't *load* it. In the Developer tab, pick it from the dropdown.
- *Dim mismatch error* → you loaded a different model than `lmstudio_model` says in your config. Fix one side or the other.
- *`thought ask` returns junk Cypher* → the chat model loaded is too small or not instruction-tuned. Try `mistral-7b-instruct` or larger for the extract / ask paths.

#### Any OpenAI-compatible server (vLLM, llama.cpp `--api`, text-generation-webui)

Same shape as LM Studio but pointed at a different URL:

```toml
[embedding]
choice = "openai-compat"
dim = 1024
openai_compat_url = "http://localhost:8000/v1"
openai_compat_model = "your-model-id"
openai_compat_api_key = ""        # blank for local; set for OpenAI cloud

[llm]
provider = "openai-compat"
model = "your-chat-model"
base_url = "http://localhost:8000/v1"
api_key = ""
```

#### Migrating an existing KB to a new embedder

If you've been using `deterministic` or `minilm` and want to upgrade to Ollama / LM Studio without re-ingesting from source:

```bash
# Edit thought.toml to point at the new embedder first (or run *-setup --write).
thought reembed --to ollama --dim 768         # re-embeds every entity in place
```

This walks every currently-valid entity, embeds its `name + canonical_name + attrs` through the new embedder, swaps the stored vector. Entities, edges, sources, and saved views are all untouched.

#### Performance honesty

Local embeddings via Ollama / LM Studio are 5-20 ms per call vs ~0.2 ms for in-process sentence-transformers. For most users that's fine; the `auto` choice still picks sentence-transformers when installed. Pick local LLMs for privacy / no-API-cost, not for raw speed.

### Querying your memory with Cypher (v0.4)

A documented Cypher subset that compiles to parameterised SQL against our typed-edge graph. Read-only in v0.4 — writes still go through `remember`.

**Discover what's queryable first:**
```bash
thought schema
#  Entity types        Relation types
#  PERSON          47  WORKS_AT      32
#  ORGANIZATION    12  PREFERS       18
#  CONCEPT         89  LIVES_IN      24
#  function       425  CALLS        575
```

**Worked examples:**
```bash
# Pattern match
thought query "MATCH (p:PERSON)-[:WORKS_AT]->(o:ORGANIZATION) RETURN p.name, o.name"

# Property + WHERE
thought query "MATCH (p:PERSON {name:'Alice'}) WHERE p.tier = 'hot' RETURN p"

# Time-travel: what did the KB believe yesterday?
thought query --as-of 2026-05-14 "MATCH (a:PERSON)-[:PREFERS]->(c) RETURN a.name, c.name"

# See the SQL we emit:
thought query --explain "MATCH (p:PERSON) RETURN p.name LIMIT 5"
```

**Save a query as a memory construct:**
```bash
thought view save adidas_seattle '
  MATCH (p:PERSON)-[:PREFERS]->(:CONCEPT {name:"Adidas"})
  MATCH (p)-[:LIVES_IN]->(:CITY {name:"Seattle"})
  RETURN p.name'
thought view run adidas_seattle    # re-evaluates against the live KB every time
thought view list
```

This is the *"join disparate facts into a new construct"* primitive — the saved view is pull-evaluated, so any future fact that satisfies it surfaces automatically. Views survive `db flush` (they describe queries, not data).

**Supported v0.4 subset:**

| Cypher feature | Supported? | Notes |
|---|---|---|
| `MATCH (n:Type {prop:value})` | ✅ | Entity match by type + property |
| `(a)-[:REL]->(b)` typed edge | ✅ | Forward + reverse `<-[:REL]-` |
| `WHERE expr AND expr` | ✅ | `=`, `<>`, `<`, `>`, `<=`, `>=`, `CONTAINS`, `STARTS WITH`, `IN` |
| `RETURN id, id.prop, id AS alias` | ✅ | Property projection + JSON objects |
| `LIMIT N`, `SKIP N` | ✅ | |
| `AS_OF "iso-date"` | ✅ | Bi-temporal time-travel |
| `OPTIONAL MATCH` | ❌ | Use multiple MATCH clauses |
| `MERGE` / `CREATE` / `DELETE` / `SET` | ❌ | Writes go through `remember` |
| Variable-length paths `-[:R*1..N]->` | ❌ | Use explicit multi-step patterns |
| `WITH` chaining | ❌ | Save intermediate as a view |
| Aggregations (`count`, `collect`) | ❌ | Coming in v0.5 |

Anything outside the subset raises a clear `UnsupportedCypher` error pointing at this table — no half-working execution.

### Asking in English — `thought ask` (v0.4)

Same query layer, but you type the question and your configured LLM translates it. With Ollama / LM Studio configured, this is zero-API-cost.

```bash
# Setup: just configure [llm] in thought.toml — any provider works.
# (Or use thought ollama-setup --write / lmstudio-setup --write.)

thought ask "who at Acme also prefers Adidas?"
#  fell back to recall: no LLM provider configured
#  (or: actual Cypher translation if [llm] is set)

thought ask "what shoes do people in Seattle prefer?" --explain
#  Cypher: MATCH (p:PERSON)-[:LIVES_IN]->(:CITY {name:"Seattle"})
#          MATCH (p)-[:PREFERS]->(c:CONCEPT) RETURN p.name, c.name
#  ... result table ...

# Harvest a good translation into a durable saved view:
thought ask "people who work at Acme" --save-as acme_people --explain
```

**Honest tradeoffs:**
- Text-to-Cypher SOTA accuracy is ~72%; local models lag by ~10-20 points. v0.4 *validates* the LLM's output against the parser before executing — bad translations fall back to plain `recall(question)` so you always get something.
- `--explain` makes the translation auditable. `--save-as` lets you curate the good ones.
- `--no-fallback` makes scripted workflows fail loudly instead of falling back silently.

### Working with code — the v0.2 capabilities

If you're using THOUGHT for AI-assisted coding (the v0.2 specialisation), there's a separate ingest path that parses your source files via tree-sitter and builds a real function-call graph:

```bash
# Ingest a codebase — entities are functions / classes / methods / modules
thought ingest-code src/

# Ingest with full git history so as_of queries work for code
thought ingest-git . --mode full

# Ask who calls a function (ranked by importance)
thought callers authenticate_user

# Ask what's affected if you change a function
thought impact authenticate_user

# Show the set difference of entities between two commits
thought diff --from v1.0 --to HEAD
```

After ingestion, the AI's regular `recall` tool also gains code awareness. Natural-language questions like *"who calls authenticate_user?"* route through the call-graph machinery automatically.

### Using the CLI directly (no AI involved)

THOUGHT works fine from a terminal without an AI assistant. The CLI is most useful for bulk operations and inspection:

```bash
# Add a single fact
thought ingest "Alice owns Acme Corp. Acme is part of HoldCo."

# Bulk-ingest a directory of Markdown notes
thought ingest --glob 'docs/**/*.md'

# Pipe in from any tool that emits one fact per line
git log --since='1 week ago' --format='%s' | thought ingest --stdin

# Query directly
thought recall "who owns Acme"

# Open an interactive REPL — type queries, type +text to add facts
thought repl

# See what's currently in memory
thought stats

# Soft-delete entities matching a SQL LIKE pattern (audit-logged, not destroyed)
thought forget "kendra%"
```

### Upgrading

When a new version of THOUGHT ships:

```bash
pip install --upgrade thought-mcp     # pull the new package
thought upgrade --all                 # re-pin every MCP client config to the new version
# Restart your AI client to pick up the new server.
```

`thought upgrade --all` solves the *"uvx is still using its cached old version"* problem by re-pinning your MCP client configs to the exact version you just installed (with the required extras included).

### MCP client config paths (manual install if `--detect` can't find your client)

If `thought install --detect` doesn't find a client you have installed, the JSON block to add manually is:

```json
{
  "mcpServers": {
    "thought": {
      "command": "uvx",
      "args": ["--from", "thought-mcp[mcp,sqlite-vec]", "thought", "serve"]
    }
  }
}
```

Per-client locations:

- **Claude Code** — `~/.claude.json` (top-level `mcpServers` block)
- **Cursor** — `~/.cursor/mcp.json`
- **Cline** — VS Code `globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` (or `~/.cline/cline_mcp_settings.json`)
- **Continue** — `~/.continue/config.json`
- **Windsurf** — `~/.codeium/windsurf/mcp_config.json`

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
thought serve [--transport stdio|streamable-http] [--host ... --port ...]
                                  # start MCP server (stdio by default; HTTP optional)
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

### DB lifecycle + local LLMs + query + ask (v0.4)

```bash
# Manage the DB:
thought db size [--json]                           # disk + entity/edge counts
thought db flush [--yes] [--before X] [--since X] [--time-axis valid|learned|created]
thought db backup <file> [--force] [--before X] [--since X] [--time-axis ...]
thought db load <file> [--yes] [--merge] [--before X] [--since X] [--time-axis ...]
thought db inspect <file> [--schema] [--json]      # peek before loading

# Local LLMs:
thought ollama-setup [--host URL] [--model M] [--write]
thought lmstudio-setup [--base-url URL] [--model M] [--write]
thought reembed --to <ollama|lmstudio|minilm|...>  # migrate embedder w/o re-ingesting

# Query:
thought schema [--json]                            # entity + relation types
thought query "<cypher>" [--as-of DATE] [--explain] [--json]
thought view save <name> "<cypher>" [--replace]
thought view list [--json]
thought view show <name>
thought view run <name> [--json]
thought view delete <name>

# Ask in English:
thought ask "<question>" [--explain] [--no-fallback] [--save-as <name>]
```

### Auto-memory + topic browsing (v0.3)

```bash
thought hook install --recall            # auto-recall: UserPromptSubmit → recall + inject
thought hook install --write             # auto-write: Stop → ingest the last turn
thought hook install --both              # both, idempotent
thought hook install --both --scope user # globally for all projects, not just this one

# The hook subcommands themselves (called by Claude Code, not by you):
thought hook recall                      # stdin: hook payload → stdout: additionalContext
thought hook write [--mode raw|extract]  # stdin: hook payload → ingests transcript

# Browsing
thought topics [--scope all|shared|private] [--min-count N] [--json]
                                         # entity-type aggregations with examples
thought browse <name> [--depth 1] [--limit 20] [--json]
                                         # drill into a type or an entity name
```

### Code-vertical commands (v0.2)

```bash
thought ingest-code <path> [--glob '**/*.py'] [--lang python|typescript|auto]
                                  # tree-sitter ingest — functions / classes / methods as entities
thought ingest-git <repo> [--mode snapshot|full] [--paths '*.py,*.ts']
                                  # commit-stamped ingest; --mode full walks every commit
thought callers <name> [--file path] [--limit 10]
                                  # direct callers ranked by HippoRAG PageRank
thought impact  <name> [--file path] [--limit 20]
                                  # transitive impact set: what's affected if you change <name>
thought diff   --from <sha1> --to <sha2> [--file path]
                                  # set diff of entities between two ingested commits
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

Coverage target: 85% on `src/thought`. CI matrix runs Python 3.11/3.12/3.13 × Ubuntu/macOS/Windows on every push (see `.github/workflows/ci.yml`). Tagging `v*` triggers `release.yml` (PyPI trusted publishing) and `docker.yml` (multi-arch GHCR image).

---

## Roadmap

**Current (shipped)** — 11 Tier A frontier techniques (Contextual Retrieval, HippoRAG PageRank, bi-temporal Graphiti, atomic-fact triples + Jaccard dedup, BGE-M3 hybrid embeddings, Matryoshka 2-pass retrieval, CRAG evaluator, MetaRAG confidence class, Ebbinghaus decay, context-engineering budget per query class, append-only writes); comparison + ablation harnesses; two MCP tools; multi-platform CLI with auto-install for five MCP clients; LRU recall cache + PPR matrix cache + sqlite-vec + scipy.sparse PageRank + local push PPR + batched ingest (the three perf passes described above); Docker + PyPI release workflows.

**v0.2 fast-follow** — RAPTOR hierarchical summary trees at WARM→COLD demotion ([Sarthi et al., ICLR 2024](https://arxiv.org/abs/2401.18059)); sleep-time compute pre-computation ([Letta + UCB, April 2025](https://arxiv.org/abs/2504.13171)); TLogic temporal-rule forecasting ([arXiv 2112.08025](https://arxiv.org/abs/2112.08025)); Reflexion-style self-edit ([Shinn et al., NeurIPS 2023](https://arxiv.org/abs/2303.11366)); multi-hop deep recall (IRCoT/PRISM); introspective `thought audit` ([transformer-circuits, 2025](https://transformer-circuits.pub/2025/introspection/index.html)).

**v0.3+** — RankZephyr local reranker, PIKE-RAG domain rationale extraction, DSPy-learned retrieval policies, real Postgres backend, REST API alongside MCP, encryption-at-rest (SQLCipher / pgcrypto), tenant isolation, OpenTelemetry traces/metrics.

---

## Star history

If this is useful, ⭐ the repo so others can find it.

[![Star History Chart](https://api.star-history.com/svg?repos=RNBBarrett/thought-mcp&type=Date)](https://star-history.com/#RNBBarrett/thought-mcp&Date)

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
