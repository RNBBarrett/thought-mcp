-- v0.5 agent identity model + incremental-scan log.
-- A named agent (vulnerability scanner, writing assistant, custom workflow) can
-- claim provenance for the facts it wrote. The `scan_log` table is a per-agent
-- history of incremental-scan calls so the next scan picks up where the last
-- left off without manual state tracking.

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    capabilities  TEXT,                  -- JSON list of strings
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT
);

-- Entities + edges get an optional agent_id stamp. ON DELETE SET NULL because
-- removing an agent should not retroactively delete its facts.
ALTER TABLE entities ADD COLUMN agent_id TEXT REFERENCES agents(id) ON DELETE SET NULL;
ALTER TABLE edges    ADD COLUMN agent_id TEXT REFERENCES agents(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_entities_agent ON entities(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_edges_agent    ON edges(agent_id)    WHERE agent_id IS NOT NULL;

-- Per-agent scan history. One row per `thought scan` invocation.
CREATE TABLE IF NOT EXISTS scan_log (
    id                TEXT PRIMARY KEY,
    agent_id          TEXT REFERENCES agents(id) ON DELETE SET NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT NOT NULL,
    repo_path         TEXT NOT NULL,
    since             TEXT,              -- "HEAD~5", date, sha — the input cursor
    head_sha          TEXT,              -- sha at the time of this scan
    files_scanned     INTEGER NOT NULL DEFAULT 0,
    files_changed     INTEGER NOT NULL DEFAULT 0,
    entities_added    INTEGER NOT NULL DEFAULT 0,
    entities_retired  INTEGER NOT NULL DEFAULT 0,
    edges_added       INTEGER NOT NULL DEFAULT 0,
    edges_retired     INTEGER NOT NULL DEFAULT 0,
    duration_ms       REAL NOT NULL DEFAULT 0,
    note              TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_log_agent_time ON scan_log(agent_id, started_at DESC);

INSERT OR IGNORE INTO schema_version (version) VALUES (4);
