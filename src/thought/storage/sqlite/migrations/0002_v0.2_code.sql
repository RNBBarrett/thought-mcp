-- THOUGHT v0.2 schema additions for the AI-coding-agent vertical.
--
-- Additive only — v0.1 databases upgrade cleanly. No backfill; the new columns
-- are nullable and only populated for code-typed entities.
--
-- New entity attrs (hot fields lifted out of attrs_json for indexable lookup):
--   code_file        — source path relative to the ingested root, e.g. "src/auth/middleware.py"
--   code_language    — "python" | "typescript" | "javascript" | "go" | "rust" | ...
--   code_commit_sha  — the commit SHA at which we last observed this entity
--
-- No changes to ``edges`` — new relation types (CALLS, IMPORTS, INHERITS_FROM,
-- OVERRIDES, DEFINES, INTRODUCED_BY, MODIFIED_IN) are just new string values.

ALTER TABLE entities ADD COLUMN code_file TEXT;
ALTER TABLE entities ADD COLUMN code_language TEXT;
ALTER TABLE entities ADD COLUMN code_commit_sha TEXT;

-- Partial indexes — only built over rows that are actually code entities, so
-- they cost ~nothing on a v0.1 database that never ingests code.
CREATE INDEX IF NOT EXISTS idx_entities_code_file
    ON entities(code_file) WHERE code_file IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_code_commit
    ON entities(code_commit_sha) WHERE code_commit_sha IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_code_lang_type
    ON entities(code_language, type) WHERE code_language IS NOT NULL;

UPDATE schema_version SET version = 2 WHERE version = 1;
INSERT OR IGNORE INTO schema_version (version) VALUES (2);
