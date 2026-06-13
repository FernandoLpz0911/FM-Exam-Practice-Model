CREATE TABLE IF NOT EXISTS concept (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    category          TEXT NOT NULL,
    exam_weight_tier  INTEGER NOT NULL,
    summary           TEXT,
    generator_json    TEXT,
    theory_md         TEXT
);

CREATE TABLE IF NOT EXISTS concept_prereq (
    concept_id  TEXT NOT NULL REFERENCES concept(id),
    prereq_id   TEXT NOT NULL REFERENCES concept(id),
    PRIMARY KEY (concept_id, prereq_id)
);

CREATE TABLE IF NOT EXISTS card_state (
    concept_id  TEXT PRIMARY KEY REFERENCES concept(id),
    stability   REAL,
    difficulty  REAL,
    last_review TEXT,
    due         TEXT,
    reps        INTEGER NOT NULL DEFAULT 0,
    lapses      INTEGER NOT NULL DEFAULT 0,
    step        INTEGER,
    state       TEXT NOT NULL DEFAULT 'learning'
);

CREATE TABLE IF NOT EXISTS session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS interaction (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES session(id),
    concept_id      TEXT NOT NULL REFERENCES concept(id),
    seed            INTEGER NOT NULL,
    problem_kind    TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    correct_answer  TEXT NOT NULL,
    user_answer     TEXT,
    is_correct      INTEGER,
    grade           INTEGER,
    elapsed_ms      INTEGER,
    shown_at        TEXT NOT NULL,
    answered_at     TEXT
);

CREATE TABLE IF NOT EXISTS readiness_snapshot (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at      TEXT NOT NULL,
    score         REAL NOT NULL,
    detail_json   TEXT
);

CREATE TABLE IF NOT EXISTS sample_exam (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at      TEXT NOT NULL,
    score         REAL NOT NULL,        -- fraction correct, 0–1
    n_questions   INTEGER,
    passing_score REAL DEFAULT 0.70,    -- SOA passing threshold fraction
    predicted     REAL,                 -- readiness score at time of exam
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_interaction_concept ON interaction(concept_id);
CREATE INDEX IF NOT EXISTS idx_interaction_session ON interaction(session_id);
