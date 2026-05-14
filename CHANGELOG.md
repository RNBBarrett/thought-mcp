# Changelog

All notable changes to **thought-mcp** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [SemVer](https://semver.org/spec/v2.0.0.html).

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

[0.2.2]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.2
[0.2.1]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.1
[0.2.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.2.0
[0.1.0]: https://github.com/RNBBarrett/thought-mcp/releases/tag/v0.1.0
