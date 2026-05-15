# Contributing to thought-mcp

Thanks for picking this up. The project is small enough that one focused
PR per topic is the right shape.

## Dev setup

```bash
# Clone and install in editable mode with everything you'll need:
git clone https://github.com/RNBBarrett/thought-mcp
cd thought-mcp
pip install -e ".[dev,mcp,code,sqlite-vec]"
```

Python 3.11+ is required (we use `from datetime import UTC` and `typing.Self`).

## Running tests

```bash
# Full unit + integration + perf:
pytest tests/ -q

# Unit only (fast — under 10 s on a modern laptop):
pytest tests/unit/ -q

# A specific test file:
pytest tests/unit/test_query_cypher.py -q
```

The integration tests under `tests/integration/` spawn a real MCP server
over stdio. They run in the default `pytest tests/` invocation thanks to
the `-m ""` configuration; CI runs them too.

The comparison harness (`tests/comparison/`) is not a pytest run — it's a
benchmarking script:

```bash
python -m tests.comparison.run    # writes docs/comparison.md
```

## Lint + type-check

```bash
ruff check src tests              # must be clean before PR
ruff format src tests             # if you've reformatted
```

We don't run `mypy` in CI today (the strict config is noisy on third-party
imports), but new code should be cleanly type-annotated.

## PR norms

- One topic per PR. If you have a bug fix and a feature, two PRs.
- Tests required for new features and for bug fixes (regression test).
- CHANGELOG entry under "Unreleased" (or the upcoming version's section)
  for anything user-visible.
- README update if you're adding a CLI command or changing user-facing
  behavior.
- Lint clean (`ruff check src tests` returns 0).
- All tests pass locally.

## Code style

- Prefer **explicit > clever**. The codebase reads like prose for a reason.
- Comments are for the *why*, not the *what*. The diff already shows the
  what.
- No mocking the database in tests — we got burned in prior projects when
  mocked tests passed and prod migrations broke. Use real SQLite, real
  ingest pipeline. The whole test suite uses real in-tmp-dir DBs.
- Avoid feature flags and backwards-compatibility shims when you can just
  change the code. Migrations are additive (`ALTER TABLE ADD COLUMN`);
  semantic changes ship in a major version bump.

## Architecture-level changes

Open an issue first to align on direction. Examples of changes that warrant
a pre-discussion:

- New entity types or edge relation types
- Changes to the bi-temporal model (`valid_*` / `learned_*`)
- New embedder backend (we already have 5+ — please see if you can extend
  `OpenAICompatibleEmbedder` rather than add a new class)
- New storage backend (Postgres adapter would be welcome; please coordinate)
- Schema migrations (the runner is `applied_migrations`-tracked; follow
  the pattern in `src/thought/storage/sqlite/migrations/0003_views.sql`)

## Releasing (for maintainers)

1. Bump `version` in `pyproject.toml` + `src/thought/__init__.py`
2. Update `CHANGELOG.md` (move "Unreleased" → version section, link the tag)
3. Update the README's "✨ New in vX.Y" headline; demote prior version to
   "Previously in vX.Y"
4. `git commit -m "feat(vX.Y.Z): ..."` and `git tag -a vX.Y.Z -m "..."`
5. `git push origin main && git push origin vX.Y.Z` — triggers PyPI + GHCR
6. Wait for the GitHub Action to publish, then:
   `uvx --refresh --from "thought-mcp[mcp,sqlite-vec]==X.Y.Z" thought --version`
   to prime the uvx cache.
7. `thought upgrade --all -V X.Y.Z` to re-pin local MCP clients.

## Getting help

- File an issue at https://github.com/RNBBarrett/thought-mcp/issues
- For security disclosures, please use a GitHub Security Advisory (private)
  rather than a public issue.
