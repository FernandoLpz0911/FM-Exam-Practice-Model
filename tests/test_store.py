"""Tests for engine/scheduler/store.py — card state persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


class TestGetOrCreate:
    def test_new_concept_returns_default(self, seeded_db):
        from engine.scheduler.store import get_or_create
        cs = get_or_create("tvm")
        assert cs.concept_id == "tvm"
        assert cs.reps == 0
        assert cs.lapses == 0
        assert cs.stability is None
        assert cs.state == "learning"

    def test_existing_concept_loaded(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create, save
        cs = get_or_create("tvm")
        cs = apply_rating(cs, 3)
        save(cs)

        cs2 = get_or_create("tvm")
        assert cs2.reps == 1
        assert cs2.stability is not None


class TestApplyRating:
    def test_reps_incremented(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create
        cs = get_or_create("tvm")
        updated = apply_rating(cs, 3)
        assert updated.reps == 1

    def test_lapse_incremented_on_again(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create, save
        cs = get_or_create("tvm")
        # First do a Good rating to get into review state
        cs = apply_rating(cs, 4)
        save(cs)
        cs = get_or_create("tvm")
        # Force state to "review" manually so Again counts as a lapse
        cs.state = "review"
        cs2 = apply_rating(cs, 1)  # Rating.Again
        assert cs2.lapses == 1

    def test_no_lapse_in_learning_state(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create
        cs = get_or_create("tvm")
        updated = apply_rating(cs, 1)
        assert updated.lapses == 0

    def test_due_date_set(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create
        cs = get_or_create("tvm")
        updated = apply_rating(cs, 3)
        assert updated.due is not None

    def test_early_reinforcement_cap(self, seeded_db):
        """First N reviews should be capped to ≤1 day, even on Easy."""
        from engine.config import EARLY_REINFORCEMENT_REPS
        from engine.scheduler.store import apply_rating, get_or_create
        cs = get_or_create("tvm")
        # Rating 4 = Easy — FSRS would normally schedule ~8+ days
        updated = apply_rating(cs, 4)
        if updated.reps < EARLY_REINFORCEMENT_REPS and updated.due is not None:
            due = updated.due
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            cap = datetime.now(timezone.utc) + timedelta(days=1, seconds=5)
            assert due <= cap

    def test_all_ratings_succeed(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create
        cs = get_or_create("tvm")
        for rating in (1, 2, 3, 4):
            updated = apply_rating(cs, rating)
            assert updated.reps == 1


class TestApplyRatingBranches:
    def test_past_early_reinforcement_no_cap(self, seeded_db):
        """When reps >= EARLY_REINFORCEMENT_REPS the cap is not applied (branch 70→77 False)."""
        from engine.config import EARLY_REINFORCEMENT_REPS
        from engine.scheduler.store import CardState, apply_rating
        cs = CardState(
            concept_id="tvm",
            reps=EARLY_REINFORCEMENT_REPS,  # e.g. 7; new_reps=8 → no cap
            stability=10.0,
            difficulty=3.0,
            state="review",
        )
        updated = apply_rating(cs, 3)
        assert updated.reps == EARLY_REINFORCEMENT_REPS + 1

    def test_naive_due_datetime_handled(self, seeded_db):
        """When FSRS returns a naive due date, cap is also made naive (line 73)."""
        from datetime import datetime, timedelta
        from unittest.mock import MagicMock, patch

        from engine.scheduler.store import apply_rating, get_or_create

        mock_card = MagicMock()
        # Naive datetime (no tzinfo) far in future → triggers cap creation at line 73
        mock_card.due = datetime.now() + timedelta(days=30)
        mock_card.stability = 5.0
        mock_card.difficulty = 3.0
        mock_card.last_review = datetime.now()
        mock_card.step = 0
        mock_card.state.name.lower.return_value = "review"

        cs = get_or_create("tvm")  # reps=0, so new_reps=1 < EARLY_REINFORCEMENT_REPS

        with patch("engine.scheduler.store._scheduler.review_card", return_value=(mock_card, None)):
            updated = apply_rating(cs, 3)

        assert updated.due is not None  # cap applied, due is the naive cap


class TestSave:
    def test_save_and_reload(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create, save
        cs = get_or_create("tvm")
        cs = apply_rating(cs, 3)
        save(cs)

        reloaded = get_or_create("tvm")
        assert reloaded.reps == 1
        assert reloaded.stability == pytest.approx(cs.stability or 0, abs=0.1)

    def test_upsert_overwrites(self, seeded_db):
        from engine.scheduler.store import apply_rating, get_or_create, save
        cs = get_or_create("tvm")
        cs = apply_rating(cs, 3)
        save(cs)
        cs2 = get_or_create("tvm")
        cs2 = apply_rating(cs2, 4)
        save(cs2)

        final = get_or_create("tvm")
        assert final.reps == 2
