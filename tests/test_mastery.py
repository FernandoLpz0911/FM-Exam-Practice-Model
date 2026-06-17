"""Tests for engine/scheduler/mastery.py."""
from __future__ import annotations

import sqlite3


def _set_card_state(db_path, concept_id, reps, state, stability):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT OR REPLACE INTO card_state
           (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step)
           VALUES (?, ?, 0, ?, ?, NULL, NULL, NULL, NULL)""",
        (concept_id, reps, state, stability),
    )
    conn.commit()
    conn.close()


class TestIsMastered:
    def test_fresh_concept_not_mastered(self, seeded_db):
        from engine.scheduler.mastery import is_mastered
        assert not is_mastered("tvm")

    def test_mastered_when_state_review_and_enough_reps(self, seeded_db):
        from engine.config import MASTERY_MIN_REPS, MASTERY_STABILITY_DAYS
        from engine.scheduler.mastery import is_mastered
        _set_card_state(seeded_db, "tvm", MASTERY_MIN_REPS, "review", MASTERY_STABILITY_DAYS)
        assert is_mastered("tvm")

    def test_not_mastered_when_stability_too_low(self, seeded_db):
        from engine.config import MASTERY_MIN_REPS
        from engine.scheduler.mastery import is_mastered
        _set_card_state(seeded_db, "tvm", MASTERY_MIN_REPS, "review", 1.0)
        assert not is_mastered("tvm")

    def test_not_mastered_when_reps_too_few(self, seeded_db):
        from engine.config import MASTERY_STABILITY_DAYS
        from engine.scheduler.mastery import is_mastered
        _set_card_state(seeded_db, "tvm", 1, "review", MASTERY_STABILITY_DAYS)
        assert not is_mastered("tvm")

    def test_not_mastered_when_state_learning(self, seeded_db):
        from engine.config import MASTERY_MIN_REPS, MASTERY_STABILITY_DAYS
        from engine.scheduler.mastery import is_mastered
        _set_card_state(seeded_db, "tvm", MASTERY_MIN_REPS, "learning", MASTERY_STABILITY_DAYS)
        assert not is_mastered("tvm")


class TestIsIntroduced:
    def test_not_introduced_on_reps_zero(self, seeded_db):
        from engine.scheduler.mastery import is_introduced
        assert not is_introduced("tvm")

    def test_introduced_after_one_rep(self, seeded_db):
        from engine.scheduler.mastery import is_introduced
        _set_card_state(seeded_db, "tvm", 1, "learning", 1.0)
        assert is_introduced("tvm")


class TestUnlockedConcepts:
    def test_concepts_without_prerequisites_always_unlocked(self, seeded_db):
        from engine.db.dao import get_all_concepts
        from engine.scheduler.mastery import unlocked_concepts
        concepts = get_all_concepts()
        unlocked = unlocked_concepts(concepts)
        ids = {c.id for c in unlocked}
        assert "tvm" in ids
        assert "theory_only" in ids

    def test_concept_with_unmet_prereq_locked(self, seeded_db):
        from engine.db.dao import get_all_concepts
        from engine.scheduler.mastery import unlocked_concepts
        concepts = get_all_concepts()
        unlocked = unlocked_concepts(concepts)
        ids = {c.id for c in unlocked}
        # annuity requires tvm, which hasn't been introduced
        assert "annuity" not in ids

    def test_concept_unlocked_after_prereq_introduced(self, seeded_db):
        from engine.db.dao import get_all_concepts
        from engine.scheduler.mastery import unlocked_concepts
        _set_card_state(seeded_db, "tvm", 1, "learning", 1.0)
        concepts = get_all_concepts()
        unlocked = unlocked_concepts(concepts)
        ids = {c.id for c in unlocked}
        assert "annuity" in ids
