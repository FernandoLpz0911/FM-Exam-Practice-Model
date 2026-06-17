"""Tests for engine/db/connection.py and engine/db/dao.py."""
from __future__ import annotations

import sqlite3

import pytest

# connection.py


def test_init_db_is_idempotent(isolated_db):
    from engine.db.connection import init_db
    init_db()  # second call must not raise
    init_db()  # third call too


def test_get_connection_returns_row_factory(isolated_db):
    from engine.db.connection import get_connection
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO session (started_at) VALUES (?)", ("2025-01-01T00:00:00",)
        )
        row = conn.execute("SELECT * FROM session").fetchone()
    assert row["started_at"] == "2025-01-01T00:00:00"


def test_migrate_adds_theory_md_when_missing(tmp_path, monkeypatch):
    """Migration must ALTER TABLE when theory_md column is absent."""
    db_file = tmp_path / "old.db"
    monkeypatch.setenv("DB_PATH", str(db_file))

    # Create concept table WITHOUT theory_md (simulates old schema)
    raw = sqlite3.connect(str(db_file))
    raw.execute(
        "CREATE TABLE concept "
        "(id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, "
        "exam_weight_tier INTEGER NOT NULL, summary TEXT, generator_json TEXT)"
    )
    raw.commit()
    raw.close()

    from engine.db.connection import _migrate, get_connection
    with get_connection() as conn:
        _migrate(conn)

    raw2 = sqlite3.connect(str(db_file))
    cols = {row[1] for row in raw2.execute("PRAGMA table_info(concept)").fetchall()}
    raw2.close()
    assert "theory_md" in cols


def test_migrate_noop_when_column_present(isolated_db):
    """Migration must not fail when theory_md is already there."""
    from engine.db.connection import _migrate, get_connection
    with get_connection() as conn:
        _migrate(conn)   # column already exists — must not raise


# dao.py — concept queries


def test_get_all_concepts_empty(isolated_db):
    from engine.db import dao
    assert dao.get_all_concepts() == []


def test_get_all_concepts_with_data(seeded_db):
    from engine.db import dao
    concepts = dao.get_all_concepts()
    ids = {c.id for c in concepts}
    assert "tvm" in ids
    assert "annuity" in ids


def test_get_concept_found(seeded_db):
    from engine.db import dao
    concept = dao.get_concept("tvm")
    assert concept is not None
    assert concept.name == "Time value of money"
    assert concept.category == "interest"
    assert concept.exam_weight_tier == 3
    assert concept.generator is not None


def test_get_concept_with_prerequisites(seeded_db):
    from engine.db import dao
    concept = dao.get_concept("annuity")
    assert concept is not None
    assert "tvm" in concept.prerequisites


def test_get_concept_not_found(seeded_db):
    from engine.db import dao
    assert dao.get_concept("nonexistent") is None


def test_get_concept_no_generator(seeded_db):
    from engine.db import dao
    concept = dao.get_concept("theory_only")
    assert concept is not None
    assert concept.generator is None


# dao.py — session lifecycle


def test_create_session(isolated_db):
    from engine.db import dao
    sid, started_at = dao.create_session()
    assert isinstance(sid, int)
    assert sid > 0
    assert "T" in started_at or "-" in started_at


def test_close_session(isolated_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    ended = dao.close_session(sid)
    assert ended is not None


# dao.py — interaction log


def test_log_shown_and_retrieve(seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(
        session_id=sid,
        concept_id="tvm",
        seed=123,
        problem_kind="interest_tvm:future_value",
        params_json='{"i": 0.05}',
        correct_answer="1276.281563",
    )
    assert isinstance(item_id, int)

    assert dao.get_interaction_concept(item_id) == "tvm"
    assert dao.get_interaction_correct_answer(item_id) == "1276.281563"
    details = dao.get_interaction_details(item_id)
    assert details["problem_kind"] == "interest_tvm:future_value"
    assert details["params"]["i"] == pytest.approx(0.05)


def test_log_shown_defaults(seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(session_id=sid, concept_id="tvm")
    assert dao.get_interaction_correct_answer(item_id) == "N/A"


def test_log_answered(seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(
        session_id=sid,
        concept_id="tvm",
        problem_kind="interest_tvm:future_value",
        params_json="{}",
        correct_answer="1000.0",
    )
    dao.log_answered(
        item_id=item_id,
        user_answer="1000.0",
        is_correct=True,
        grade=4,
        elapsed_ms=5000,
    )
    # Should not raise; smoke-check interaction history
    history = dao.get_interaction_history()
    assert any(cid == "tvm" and correct for cid, correct in history)


def test_get_interaction_concept_missing(isolated_db):
    from engine.db import dao
    assert dao.get_interaction_concept(99999) is None


def test_get_interaction_correct_answer_missing(isolated_db):
    from engine.db import dao
    assert dao.get_interaction_correct_answer(99999) is None


def test_get_interaction_details_missing(isolated_db):
    from engine.db import dao
    assert dao.get_interaction_details(99999) is None


def test_get_interaction_history_empty(isolated_db):
    from engine.db import dao
    assert dao.get_interaction_history() == []


def test_get_interaction_history_ordered(seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    for i in range(3):
        iid = dao.log_shown(session_id=sid, concept_id="tvm", problem_kind="k:a", params_json="{}", correct_answer="1.0")
        dao.log_answered(iid, "1.0", True, 4, 0)
    hist = dao.get_interaction_history()
    assert len(hist) == 3
    assert all(cid == "tvm" for cid, _ in hist)


# dao.py — sample exam + journal


def test_log_sample_exam(isolated_db):
    from engine.db import dao
    row_id = dao.log_sample_exam(score=0.75, n_questions=30, passing_score=0.70, notes="test")
    assert row_id > 0


def test_get_sample_exams(isolated_db):
    from engine.db import dao
    dao.log_sample_exam(score=0.60)
    dao.log_sample_exam(score=0.80, n_questions=20, predicted=0.75)
    exams = dao.get_sample_exams()
    assert len(exams) == 2
    assert exams[0]["score"] == pytest.approx(0.60)


def test_get_journal_stats_empty(isolated_db):
    from engine.db import dao
    stats = dao.get_journal_stats()
    assert stats["total_answered"] == 0
    assert stats["accuracy"] == pytest.approx(0.0)
    assert stats["total_hours"] == pytest.approx(0.0)


def test_get_journal_stats_with_data(seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    iid = dao.log_shown(session_id=sid, concept_id="tvm", problem_kind="k:a", params_json="{}", correct_answer="1.0")
    dao.log_answered(iid, "1.0", True, 4, 3_600_000)
    dao.close_session(sid)
    stats = dao.get_journal_stats()
    assert stats["total_answered"] == 1
    assert stats["total_correct"] == 1
    assert stats["accuracy"] == pytest.approx(1.0)
    assert stats["total_hours"] == pytest.approx(1.0)
    assert stats["sessions_completed"] == 1
    assert stats["concepts_touched"] == 1


def test_count_answered_interactions(seeded_db):
    from engine.db import dao
    assert dao.count_answered_interactions() == 0
    sid, _ = dao.create_session()
    iid = dao.log_shown(session_id=sid, concept_id="tvm", problem_kind="k:a", params_json="{}", correct_answer="1.0")
    dao.log_answered(iid, "1.0", True, 4, 0)
    assert dao.count_answered_interactions() == 1


def test_init_db_no_directory(monkeypatch, tmp_path):
    """Branch: when db_dir is empty string, makedirs is skipped."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_PATH", "nodirtest.db")
    from engine.db.connection import init_db
    init_db()
    assert (tmp_path / "nodirtest.db").exists()
