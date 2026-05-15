-- v0.4 saved views: named Cypher queries persisted as first-class memory constructs.
-- A saved view is pull-evaluated — running it re-executes the stored cypher
-- against the live KB. Survives `db flush` because it describes a query, not data.

CREATE TABLE IF NOT EXISTS saved_views (
    name           TEXT PRIMARY KEY,
    cypher_source  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    last_run_at    TEXT,
    last_run_ms    REAL,
    last_run_count INTEGER,
    scope          TEXT NOT NULL DEFAULT 'all',
    owner_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_saved_views_name ON saved_views(name);

INSERT OR IGNORE INTO schema_version (version) VALUES (3);
