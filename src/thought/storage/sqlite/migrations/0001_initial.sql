-- THOUGHT v0.1 initial schema
-- Canonical schema. Postgres adapter mirrors with type substitutions.
-- All writes are append-only; retirement is a new edge, never an UPDATE/DELETE.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Immutable raw sources. Dedupe by sha256(content).
-- context_summary holds the Anthropic-Contextual-Retrieval pre-embedding context.
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    mime_type       TEXT NOT NULL DEFAULT 'text/plain',
    ingested_at     TEXT NOT NULL,
    context_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);

-- Entities (nodes). Bi-temporal: world-time + transaction-time.
CREATE TABLE IF NOT EXISTS entities (
    id                TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    name              TEXT NOT NULL,
    canonical_name    TEXT NOT NULL,
    owner_id          TEXT,
    scope             TEXT NOT NULL CHECK (scope IN ('shared','private')),
    tier              TEXT NOT NULL DEFAULT 'hot' CHECK (tier IN ('hot','warm','cold')),
    importance        REAL NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    valid_from        TEXT NOT NULL,
    valid_until       TEXT,
    learned_at        TEXT NOT NULL,
    unlearned_at      TEXT,
    created_at        TEXT NOT NULL,
    last_accessed_at  TEXT NOT NULL,
    access_count      INTEGER NOT NULL DEFAULT 0,
    attrs_json        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entities_canon_scope ON entities(canonical_name, scope, owner_id);
CREATE INDEX IF NOT EXISTS idx_entities_validity   ON entities(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_entities_learned    ON entities(learned_at);
CREATE INDEX IF NOT EXISTS idx_entities_tier_acc   ON entities(tier, last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC);

-- Edges (typed relations). Mandatory source_ref enforces provenance.
-- Special types: CONTRADICTS, SUPERSEDES, DERIVED_FROM, plus user-defined.
CREATE TABLE IF NOT EXISTS edges (
    id                TEXT PRIMARY KEY,
    source_id         TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id         TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type     TEXT NOT NULL,
    source_ref        TEXT NOT NULL REFERENCES sources(id),
    confidence_score  REAL NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    confidence_class  TEXT NOT NULL DEFAULT 'source_grounded'
        CHECK (confidence_class IN ('source_grounded','inferred','hallucination_risk')),
    valid_from        TEXT NOT NULL,
    valid_until       TEXT,
    learned_at        TEXT NOT NULL,
    unlearned_at      TEXT,
    detected_at       TEXT NOT NULL,
    cross_scope       INTEGER NOT NULL DEFAULT 0,
    attrs_json        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_source   ON edges(source_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_target   ON edges(target_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type, valid_from, valid_until);

-- Atomic-fact triples projection. Dedupe at write-time via Jaccard.
CREATE TABLE IF NOT EXISTS triples (
    id            TEXT PRIMARY KEY,
    subject_id    TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate     TEXT NOT NULL,
    object_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    edge_id       TEXT NOT NULL REFERENCES edges(id) ON DELETE CASCADE,
    fingerprint   TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    UNIQUE (subject_id, predicate, object_id, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_triples_subject_pred ON triples(subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_triples_fingerprint  ON triples(fingerprint);

-- Embeddings: model-versioned vectors. Full-dim canonical storage; ANN index is derived.
CREATE TABLE IF NOT EXISTS embeddings (
    entity_id      TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    dim            INTEGER NOT NULL,
    vector         BLOB NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (entity_id, model_name, model_version)
);

-- Strength cache for Ebbinghaus decay scoring.
CREATE TABLE IF NOT EXISTS strength_cache (
    entity_id         TEXT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    strength          REAL NOT NULL,
    last_computed_at  TEXT NOT NULL
);

-- Audit log for consolidation operations.
CREATE TABLE IF NOT EXISTS consolidation_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    op            TEXT NOT NULL,
    target_kind   TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    before_json   TEXT,
    after_json    TEXT,
    occurred_at   TEXT NOT NULL,
    actor         TEXT NOT NULL DEFAULT 'consolidator'
);
CREATE INDEX IF NOT EXISTS idx_consolidation_target ON consolidation_log(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_run    ON consolidation_log(run_id);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);
