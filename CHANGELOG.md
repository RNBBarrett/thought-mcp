# Changelog

All notable changes to **thought-mcp** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [SemVer](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — 2026-05-15 — Agent substrate + multi-language code memory

Pivot release: THOUGHT goes from *"Claude Code's memory tool"* to *"memory
backend any agent loop can use."* Three feature pillars + a slew of deferred
items called out at the bottom.

### Added

#### Agent identity + incremental scan
- **`agents` table + `scan_log` table** (migration `0004_agents.sql`,
  ``schema_version`` → 4). Entities + edges gain an optional ``agent_id``
  column. Named agents claim provenance for the facts they write; the
  scan_log gives each agent an automatic cursor so re-scans pick up where
  the last left off.
- **`thought scan <repo> [--as-agent NAME] [--since REF]`** — incremental
  code-scan primitive. Walks the repo, ingests changed/new files via the
  existing `CodeIngestPipeline`, records a row in `scan_log`, returns a
  structured summary the agent can act on.
- **`thought agent register/list/log`** — CLI for the agent identity model.
- **`mcp__thought__working_context(target, role, budget_tokens)`** — the
  universal *"what does my agent need to know right now"* primitive.
  Returns a token-budgeted, PPR-ranked, role-aware payload covering the
  anchor entity, top-K neighbours, recent contradictions, and any saved
  view named after the role. **The single tool every adapter calls.**
- **`mcp__thought__scan` / `scan_log_list` / `register_agent`** — MCP
  surface for the same primitives.

#### Multi-language code support
- **4 new language extractors** alongside Python + TypeScript/JavaScript
  from v0.2:
  - **Go** (`go_extractor.py`) — modules, structs, methods (receiver-qualified
    as `Cat.Meow`), interfaces, IMPORTS edges.
  - **Rust** (`rust_extractor.py`) — modules, functions, structs/enums/traits
    (as `class`-typed entities with `rust_kind` attr), impl blocks emit
    DEFINES + INHERITS_FROM edges, `use` declarations.
  - **Java** (`java_extractor.py`) — package modules, classes, interfaces,
    enums, records, methods, constructors. Honours `extends` / `implements`
    as INHERITS_FROM edges.
  - **PHP** (`php_extractor.py`) — namespace modules, classes, interfaces,
    traits, methods with visibility modifiers, `use` declarations.
- **Shared extractor helpers** (`_common.py`) — text-of, visibility-of,
  module-from-path, named-descendant walking.
- **`detect_language` extended** to map `.go` / `.rs` / `.java` / `.php` →
  the right extractor automatically.

#### Agent-SDK adapter package
- **`src/thought/adapters/claude_sdk.py`** — `ThoughtMemoryProvider` class:
  drop-in memory adapter for any Claude-Agent-SDK-shaped agent. Three
  methods cover the agent loop: `context_for(target, role)` returns a
  working-context dict; `render_context(target)` returns the same payload
  as a plain-text system-prompt augmentation; `record(content)` persists
  what the agent learned; `scan(repo_path)` runs an incremental scan
  under the agent's name.

#### Codebase mapping
- **`thought codebase-map [--budget-tokens N]`** — Aider-style top-N most
  important symbols across the KB, ranked by HippoRAG-style Personalized
  PageRank. The persistent equivalent of Aider's per-prompt repo map.

### Changed
- `pyproject.toml` `[code]` extra gains `tree-sitter-go`, `tree-sitter-rust`,
  `tree-sitter-java`, `tree-sitter-php`.
- New `[embeddings-code]` extra (reserved for the v0.5.1 jina-code embedder
  work).
- New `[adapters]` extra (currently just `httpx>=0.27` for the SDK
  adapters).

### Internal
- 320 tests pass (was 295 at v0.4.0). +25 across the new extractors,
  agent identity, scan, working_context, codebase-map CLI, and the
  Claude SDK adapter round-trip.
- Comparison harness re-run: 83.5% recall@10 (unchanged vs v0.4.0).

### Honest defers — what didn't ship in v0.5.0 but is in the v0.5 plan

The approved plan covered ~22 days of work spanning four verticals; this
release ships the most-leveraged ~30% in one focused session. Coming in
follow-on releases:

- **v0.5.1**: Function-body semantic embeddings via jina-embeddings-v2-base-code
  + `mcp__thought__find_similar_code`; LangChain / AutoGen / Pydantic AI /
  CrewAI / Letta adapter shims (only Claude SDK adapter shipped in 0.5.0);
  runnable `examples/vuln_scanner/` reference agent.
- **v0.6**: Writing vertical — `thought ingest-prose`, fiction + academic
  entity taxonomies, continuity-check / outline / citations / timeline
  commands.
- **v0.7**: Investigations vertical — `thought ingest-legal` / `ingest-osint`
  / `ingest-compliance` / `ingest-forensic` with the matching entity/edge
  taxonomies, plus deposition-analyzer + osint-aggregator reference agents.
- **v0.8**: Platform features — `thought graph` TUI, `memory-diff`,
  `bench` (LongMemEval), `publish-view`, federated sync, cryptographic
  attestations.

---

## [0.4.0] — 2026-05-15 — DB lifecycle + Local LLMs + Cypher + Ask

A big release. Four feature areas combine to make THOUGHT a complete local-AI
memory tool — manage the KB, run on local models, write graph queries, and
ask in English.

### Added

#### DB lifecycle (`thought db ...`)
- **`db size`** — disk usage of main + WAL + SHM sidecars + entity/edge counts.
- **`db flush`** — wipe the KB. Defaults to full flush; ``--before X`` /
  ``--since X`` / ``--time-axis valid|learned|created`` for date-bounded
  deletes. Interactive confirmation by default; ``--yes`` skips. Always
  auto-backs-up to ``<db>.bak.<timestamp>`` before destructive operations.
- **`db backup <file>`** — SQLite online-backup snapshot. Date filters
  produce a clean, self-contained subset file (DELETE + VACUUM after backup).
  ``--force`` to overwrite.
- **`db load <file>`** — atomically replace the active DB (or ``--merge``
  to INSERT-OR-IGNORE rows from the snapshot). Date filters apply to
  both modes. Auto-backs-up the current DB before replace.
- **`db inspect <file>``** — counts + (optional) schema summary of a backup
  file without loading it. The *"is this snapshot worth loading?"* primitive.
- New backend primitives ([src/thought/storage/sqlite/backend.py](src/thought/storage/sqlite/backend.py)):
  ``file_sizes`` / ``checkpoint_wal`` / ``flush`` / ``backup_to`` /
  ``merge_from`` / ``open_readonly`` classmethod for read-only inspection.
- WAL checkpoint in ``close()`` so backups always see a consistent file.

#### Local-LLM integration (Ollama + LM Studio + any OpenAI-compatible server)
- **`OllamaEmbedder`** — talks Ollama's native ``/api/embed`` (batched) with
  legacy ``/api/embeddings`` fallback. Auto-validates dim mismatch with
  clear error messages.
- **`OpenAICompatibleEmbedder`** + **`LMStudioEmbedder`** + **`OpenAIEmbedder``** —
  same embedder serves LM Studio, vLLM, llama.cpp ``--api``, text-generation-webui,
  and OpenAI proper. Optional ``api_key``.
- New embedder choices: ``ollama`` / ``lmstudio`` / ``openai-compat`` / ``openai``.
- ``thought.toml`` ``[embedding]`` gains ollama_host / ollama_model /
  lmstudio_url / lmstudio_model / openai_compat_url / openai_compat_model /
  openai_compat_api_key fields. Env overrides for all of them.
- **LLM-extract dispatch** — [src/thought/hooks/write.py](src/thought/hooks/write.py)'s
  ``_extract_facts()`` now dispatches on ``[llm] provider``: anthropic /
  ollama / lmstudio / openai-compat / openai / none. Auto-write ``--mode
  extract`` is now zero-API-cost for Ollama / LM Studio users.
- **`thought ollama-setup`** + **`thought lmstudio-setup`** — daemon ping +
  model discovery + ``thought.toml`` snippet. ``--write`` rewrites the config.
- **`thought reembed --to <choice>``** — re-embed every entity through a
  different embedder. Lets you start with ``deterministic`` and upgrade to
  Ollama / sentence-transformers later without re-ingesting from source.

#### Cypher query layer (subset)
- **`thought query "<cypher>"`** — run a documented Cypher subset against
  the live KB. Pattern matching, property filters, edge traversal, WHERE
  with AND, RETURN with property projection or full-row JSON, AS_OF for
  time-travel, LIMIT / SKIP. ``--explain`` shows the emitted SQL.
- Out-of-subset features raise ``UnsupportedCypher`` with a pointer to the
  README — no surprises, no half-working execution.
- **Saved views**: ``thought view save/list/run/show/delete <name>``.
  Stored as a row in the new ``saved_views`` table (migration 0003,
  schema_version → 3). Pull-evaluated on each ``view run`` against the
  live KB — the "derived memory construct" primitive.
- New MCP tools: ``schema``, ``query``, ``view_save``, ``view_list``,
  ``view_run``, ``view_delete``. Agents can compose Cypher and persist
  named views directly.

#### `thought ask` — natural-language wrapper
- **`thought ask "<english question>"`** — emits Cypher via whichever LLM
  is configured in ``[llm] provider`` (anthropic / ollama / lmstudio /
  openai-compat / openai). Validates against the parser before executing;
  bad translations degrade gracefully to ``recall(question)`` so the user
  always gets something.
- ``--explain`` shows the emitted Cypher + SQL before results.
- ``--save-as <name>`` persists a successful translation as a named view —
  harvest good NL queries into durable saved views.
- ``--no-fallback`` for scripted workflows that need to fail loudly.

#### Schema introspection
- **`thought schema`** + ``mcp__thought__schema()`` — entity-type + relation-type
  counts. Tells humans and LLMs alike what's queryable before composing Cypher.

#### SessionStart auto-context hook
- **`thought hook install --context`** registers a SessionStart hook that
  evaluates a designated saved view (default ``__startup__``) and injects
  the result as ``additionalContext`` at the start of every Claude Code
  session.

### Changed
- ``Memory.open`` accepts an optional ``embedding_cfg`` parameter so local-
  LLM embedders pick up their provider-specific config from ``thought.toml``.
- Migration runner adds ``0003_views.sql``; ``schema_version`` bumps to 3.

### Internal
- 295 tests pass (was 197 at v0.3.0). +98 new tests across DB lifecycle,
  Ollama, OpenAI-compat, reembed, hook write provider dispatch, setup
  helpers, SessionStart hook, Cypher lex/parse/compile/execute, saved
  views CRUD, ``ask`` mocked-provider dispatch, and CLI surface for every
  new command.
- Comparison harness re-run: 83.5% recall@10 (unchanged vs v0.3.0). No
  regression from the new ingest / query paths.

### Notes
- Cypher subset is read-only in v0.4. Writes still go through ``remember``
  / ``thought ingest`` / the auto-write hook. ``MERGE`` / ``CREATE`` /
  ``DELETE`` raise ``UnsupportedCypher`` at parse time.
- Variable-length paths (``-[:R*1..N]->``) are not supported in v0.4 — use
  multi-step explicit patterns. Documented in the README's supported-subset
  table.

---

## [0.3.0] — 2026-05-15 — Auto-write + auto-recall + topic browsing

### Added

#### Auto-memory via Claude Code hooks
- **`thought hook recall`** — `UserPromptSubmit` hook implementation. Reads
  the hook payload from stdin, runs ``recall(query=prompt)``, emits the
  result as ``additionalContext`` for Claude Code's next turn. Bounded
  to 8k chars; gated on ``low_confidence`` so it stays silent when there's
  nothing relevant. ~1-5 ms on a warm KB.
- **`thought hook write`** — `Stop` hook implementation. Reads the session
  transcript from the hook payload, picks the last user + last assistant
  turn, ingests both via ``Memory.remember_many``. Idempotent on content
  sha256 — replays don't double-write.
  - **`--mode raw`** (default): cheap; the ingest pipeline's Jaccard dedup
    + fact extractor absorb low-signal phrasing.
  - **`--mode extract`**: LLM-extracts durable facts via Anthropic Haiku
    before ingest. Costs ~$0.001/turn; falls back to ``raw`` with a stderr
    warning when ``ANTHROPIC_API_KEY`` or the ``[llm-anthropic]`` extra is
    missing.
- **`thought hook install [--recall|--write|--both]`** — writes the hook
  entries into ``.claude/settings.json`` (project-scoped by default, or
  ``--scope user`` for global). Idempotent. Backs up the original to
  ``settings.json.thought.bak`` before write.

#### Topic browsing
- **`thought topics`** + **`mcp__thought__list_topics`** — entity-type
  aggregations with the top-access-count examples per type. Cheap (one
  GROUP BY + one SELECT-LIMIT per type).
- **`thought browse <name>`** + **`mcp__thought__browse_topic`** —
  drill into a topic. Two-step resolution: name matches an entity type
  (``PERSON``, ``function``, ``CONCEPT``…) → returns the top entities of
  that type; otherwise treats ``name`` as an entity, resolves to an
  anchor via canonical-name match, returns the PPR-ranked neighbourhood
  (BFS fallback if PPR is empty).

#### Backend
- ``backend.count_by_type(scope_filter) → dict[str, int]`` (abstract method
  on ``StorageBackend``; SQLite impl is one GROUP BY on the existing
  ``e.type`` index).
- ``backend.find_anchor_by_name(name, scope_filter) → Entity | None`` —
  canonical-name-keyed anchor lookup ordered by access_count + importance.

### Changed
- ``Memory.list_topics`` + ``Memory.browse_topic`` facade methods.
- ``unique_predicates`` defaults wired for auto-write so user-preference
  facts (``PREFERS``, ``WORKS_AT``, ``OWNS``, ``REPORTS_TO``) auto-supersede
  on conflict via the existing bi-temporal contradiction mechanism — the
  Zep-style "temporal validity window" pattern.

### Research informing the design
The auto-memory design choices are grounded in 2024–2026 work on
conversational memory: HippoRAG / HippoRAG 2 (PPR retrieval — already in
use), Zep / Graphiti (temporal-validity contradiction handling), Mem0
(every-turn retrieval + tool-result injection), A-Mem (agentic memory
evolution — aspirational; deferred). The "skip aspirational
episode→semantic consolidation in v0.3" decision is intentional: Larimar
and A-Mem are research-grade, not production-ready, and the existing
Ebbinghaus-decay consolidation engine is enough to keep auto-write noise
from compounding.

### Internal
- 197 tests pass (was 150 at v0.2.2). 47 new tests across topic browsing,
  hook subcommands, hook installer, transcript reader, turn picker, raw +
  extract write modes, and the full ``write → recall`` integration loop.
- New package ``src/thought/hooks/`` (``recall.py``, ``write.py``,
  ``install.py``) — pure-Python, testable without subprocess.
- Comparison harness re-run: 83.5% recall@10 (unchanged vs v0.2.2). No
  regression from the new ingest paths.

---

## [0.2.2] — 2026-05-14 — Critical: MCP stdio transport + Windows config + thread-safe SQLite

### Fixed
- **CRITICAL — MCP server unreachable from any client since v0.1.0.**
  `thought serve` hardcoded ``transport="streamable-http"``, but every MCP
  client config wired up by `thought install` / `thought upgrade`
  invokes the server via stdio (``uvx --from "thought-mcp[mcp,sqlite-vec]==X"
  thought serve``). The HTTP listener bound port 8000, the client waited
  for stdio frames, the handshake timed out at 30 s. New default is
  ``--transport stdio``; pass ``--transport streamable-http`` for the
  HTTP transport (used by ``thought start`` for foreground dev).
- **CRITICAL — every MCP tool call would raise** ``ProgrammingError:
  SQLite objects created in a thread can only be used in that same
  thread``. The server dispatches tool work via ``asyncio.to_thread``,
  so the SQLite connection (created on the event-loop thread) was
  touched from a worker thread without ``check_same_thread=False``.
  The backend now opens the connection with cross-thread access enabled;
  SQLite's C-level mutex guarantees serialization.
- **`thought init` wrote invalid TOML on Windows.** ``db_path =
  "C:\Users\..."`` contains ``\U``, which is a TOML escape sequence
  requiring 8 hex digits; the next CLI call crashed with ``TOMLDecodeError:
  Invalid hex value (at line 1, column 16)``. ``init`` now normalises
  backslashes to forward slashes in the TOML output (SQLite accepts
  forward slashes on Windows).
- **`thought serve --host` / `--port` were silently ignored** for the
  HTTP transport. FastMCP carries its own ``settings.host`` /
  ``settings.port`` (default ``0.0.0.0:8000``); the CLI args are now
  pushed through to those settings.

### Changed
- **`mcp>=1.9`** in core deps (was ``>=1.0``). FastMCP's
  ``streamable-http`` transport landed in 1.9.0; earlier versions only
  supported ``stdio`` and ``sse``. The CLI's transport choices
  (``stdio`` / ``streamable-http``) require 1.9+.
- ``thought serve`` banner now goes to stderr in stdio mode so it
  doesn't corrupt MCP frames on stdout. Most MCP clients surface stderr
  in their logs panel.

### Added
- **`tests/integration/`** — first integration tests in the repo:
  - `test_mcp_stdio_e2e.py` spawns the server as a subprocess and drives
    it through the official ``mcp`` SDK client. The test that would have
    caught the v0.2.1 ship bug.
  - `test_mcp_http_smoke.py` confirms the HTTP transport binds the
    requested port.
- **`tests/unit/test_cli.py`** — end-to-end coverage for every CLI
  subcommand via ``typer.testing.CliRunner`` (28 tests). Zero CLI
  coverage existed before v0.2.2.
- **`tests/unit/test_server_tools.py`** — direct ``call_tool`` coverage
  for the MCP ``remember`` / ``recall`` handlers without going through
  any transport (7 tests). Catches the same thread-affinity class of
  bug as the integration tests, faster.

### Internal
- 150 tests pass (was 101 before v0.2.2). 39 new tests: 28 CLI + 7
  server-tools + 3 stdio e2e + 1 HTTP smoke.

---

## [0.2.1] — 2026-05-14 — Upgrade command + uvx-cache fix + critical MCP-startup fix

### Fixed
- **CRITICAL:** ``mcp`` is now a core dependency, not an optional extra.
  v0.2.0's ``uvx --from thought-mcp==0.2.0 thought serve`` would crash at
  startup with ``ModuleNotFoundError: No module named 'mcp'`` because
  uvx installs only core deps when ``--from`` doesn't include extras.
  The ``[mcp]`` extra is kept as a no-op alias so existing install
  recipes don't break.
- **`thought upgrade`** now generates configs that include the right
  extras: ``uvx --from "thought-mcp[mcp,sqlite-vec]==X.Y.Z" thought serve``.
- Last lingering reference to `rbarrett-indeed/thought-mcp` in the
  ``CLAUDE.md`` template inside ``cli.py`` (now ``RNBBarrett/thought-mcp``).
- CI workflow's `pip install` now includes the ``[code]`` extra so the
  tree-sitter / call-graph tests added in v0.2.0 actually run in CI.

### Added
- **`thought upgrade --client <name>` / `--all` / `--version X.Y.Z`** —
  re-pins MCP client configs to a specific ``thought-mcp`` version using
  ``uvx --from "thought-mcp[mcp,sqlite-vec]==<ver>" thought serve``.
  Forces uvx to fetch the named version on next IDE restart instead of
  reusing cached older builds. Default ``version`` is the running CLI's
  ``__version__``.
- **`clients.pin_server_block(version=...)`** helper + **`clients.upgrade()`**
  / **`clients.upgrade_many()`** functions backing the new CLI command
  (reuses the existing ``install`` machinery for backups + atomic writes).

---

## [0.2.0] — 2026-05-13 — Memory for AI coding agents

This release specialises THOUGHT for AI-assisted coding workflows. The
core three-layer memory architecture from v0.1 is unchanged; v0.2 adds an
AST-aware ingest pipeline, a call-graph edge type, git-history-stamped
provenance, and five new CLI commands. Existing v0.1 databases upgrade
cleanly — the schema additions are all `ALTER TABLE ADD COLUMN`.

### Added

#### AST-aware code ingest
- **`thought ingest-code <path>`** — parse source files via tree-sitter and
  ingest functions / classes / methods / modules as first-class entities
  (no more sliding-window text chunks).
- **Python + TypeScript / JavaScript support** out of the box. Each
  language plugin is a single file under `src/thought/ingest/code/`;
  community can add Go, Rust, Java in the same shape.
- Entities carry `code_file`, `code_language`, `code_commit_sha` columns
  plus the function signature, line range, docstring, and visibility
  (public / private by leading-underscore convention).
- Method names are class-qualified (`JWTAuth.verify`) so methods don't
  collide across classes.

#### Call-graph edges
- **`CALLS`** typed edge between functions / methods. Phase-2 pass walks
  each function body for call expressions and resolves the callee.
- Resolution order: in-file match → unique qualified suffix match
  (`obj.method()` → `ClassName.method`) → cross-file bare-name match →
  inferred stub.
- **Python builtins filtered** from stub creation (`len`, `sum`, `.append`,
  …) so the impact graph isn't polluted with noise.
- Also extracted from AST: `IMPORTS`, `INHERITS_FROM`, `DEFINES`,
  `OVERRIDES` (TypeScript only at present).

#### Git-history-aware ingest
- **`thought ingest-git <repo>`** with two modes:
  - `--mode snapshot` (default, fast): ingest HEAD only, stamp every
    entity with the HEAD SHA.
  - `--mode full`: walk every commit chronologically, stamp each entity
    with its commit SHA — enables bi-temporal `as_of` queries against
    historical commits.
- Pure-subprocess `GitWalker` — no native dependency on `pygit2`.

#### New CLI commands
- **`thought callers <name>`** — direct callers ranked by Personalized
  PageRank (HippoRAG-style bidirectional walks).
- **`thought impact <name>`** — transitive impact set: "what's affected
  if I change this?"
- **`thought diff --from <sha1> --to <sha2>`** — set difference of
  entities between two ingested commits. Added / removed lists.

#### Router additions
- New **`CODE` query class** in the Router. Triggered by code-shaped
  keywords (`function`, `class`, `caller`, `callee`, `impact`, file
  extensions, `since v1.0`, `before this commit`, …) plus camelCase /
  snake_case identifiers.
- CODE × CHANGE combinations promote to `HYBRID` (e.g. *"what changed
  in auth.middleware since v1.0"*).
- Natural-language queries like *"who calls authenticate_user"* now
  route through the same call-graph machinery without invoking the CLI.

#### Storage layer
- New columns on `entities`: `code_file`, `code_language`, `code_commit_sha`
  with partial indexes.
- New `backend.find_code_entity(canonical_name, ...)` — fast lookup by
  name + optional disambiguators for the call-graph resolver.
- Migrations now track applied filenames in an `applied_migrations` table
  → safe to re-run without re-applying `ALTER TABLE` statements.
- Backend's `upsert_entity` identity now includes `(code_file,
  code_commit_sha)` so methods of the same name in different files /
  commits don't merge.

### Changed
- The `auto` embedder selector now probes the underlying
  `sentence_transformers` package via `importlib.util.find_spec` before
  returning the wrapper, so the fallback to the deterministic embedder
  triggers correctly when the optional dep is missing.

### Fixed
- Migration runner is now idempotent (was running `ALTER TABLE ADD
  COLUMN` on every open, which failed on the second call against a v0.2+
  database).

### Internal
- 45 new unit tests (56 → 101 total).
- New test fixtures under `tests/fixtures/code/{python,typescript}/`.
- Dogfood: ingesting the THOUGHT codebase itself produces 425 entities
  and 575 CALLS edges in <250ms; the killer-demo query *"who calls
  GraphLayer.personalized_pagerank"* returns exactly the four real
  callers ranked by PageRank.

---

## [0.1.0] — 2026-05-13 — Initial release

The horizontal-memory MCP server. Two MCP tools (`remember` / `recall`),
three retrieval layers (Vector / Graph / Temporal), Router-based query
classification, 11 frontier techniques stacked.

### Added
- **MCP server** (FastMCP, Streamable HTTP transport) exposing
  `remember(content)` and `recall(query)`.
- **Query Router** — rule-based classifier (VIBE / FACT / CHANGE /
  HYBRID) dispatches to the right layer.
- **Vector Layer** — sqlite-vec ANN + Matryoshka 2-pass retrieval +
  GraphRAG-style graph expansion + optional binary sign-quantised index.
- **Graph Layer** — typed-edge graph with HippoRAG-style Personalized
  PageRank (scipy.sparse) and Andersen-Chung-Lang local-push variant
  for large knowledge bases.
- **Temporal Layer** — bi-temporal validity (`valid_*` + `learned_*`),
  tier transitions, `as_of` queries.
- **Ingest pipeline** — atomic-fact triples + Jaccard dedup + Contextual
  Retrieval (Anthropic 2024) + MetaRAG confidence class + write-time
  contradiction detection.
- **Consolidation engine** — background thread with Ebbinghaus decay,
  duplicate merging, staleness flagging, cold-tier demotion.
- **Pluggable storage** — SQLite + sqlite-vec default; Postgres + pgvector
  stub for future. Append-only writes; nothing ever deleted.
- **Pluggable embedder** — deterministic (test), `sentence-transformers/
  all-MiniLM-L6-v2` (production), `auto` mode picks the best available.
- **Multi-user scope** — native `(shared / private + owner_id)` zones
  enforced at the storage layer.
- **CLI** — 11 commands: `init`, `start`, `install`, `serve`, `ingest`,
  `recall`, `repl`, `stats`, `forget`, `consolidate`, `doctor`.
- **MCP client auto-installer** — `thought install --client {claude-code,
  cursor, cline, continue, windsurf}` writes the `mcpServers` config block.
- **LRU recall cache** — keyed by `(write_version, query, …)`; cache hits
  are µs-scale (~130,000× over cold).
- **PPR transition-matrix cache** — repeat FACT recalls skip matrix
  rebuild entirely.
- **Touch-access batched flush** — eliminates per-hit `UPDATE` on the
  recall hot path.

### Performance
- Recall p50 on a 10k-entity KB: **62 ms cold / 0.7 µs cached**.
- Comparison harness vs. OB1 / Karpathy-wiki simulators: **83.5% overall
  recall@10**, **68% on CHANGE** (OB1: 32%, Karpathy: 0%).
- Sub-linear scaling: 50× more data → ~6× more latency.

### Infrastructure
- **PyPI**: trusted publishing via GitHub Actions.
- **GHCR**: multi-arch (amd64 + arm64) Docker image on every tag.
- **CI**: Python 3.11 / 3.12 / 3.13 × Ubuntu / macOS / Windows matrix.
- **56 unit tests**, **4 perf benchmarks**, comparison + ablation
  harnesses.

[0.5.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.5.0
[0.4.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.4.0
[0.3.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.3.0
[0.2.2]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.2
[0.2.1]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.1
[0.2.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.0
[0.1.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.1.0
