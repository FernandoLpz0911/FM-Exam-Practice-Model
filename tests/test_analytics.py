"""Tests for engine/analytics/readiness.py and engine/analytics/charts.py."""
from __future__ import annotations

import pytest


def _make_concepts(seeded_db):
    from engine.db.dao import get_all_concepts
    return get_all_concepts()


# readiness.py


class TestFsrsPCorrect:
    """_fsrs_p_correct is private but exercised via compute_readiness fallback."""

    def test_cold_start_returns_prior(self, seeded_db):
        from engine.analytics.readiness import _fsrs_p_correct
        # No card state → reps=0 → cold-start prior 0.3
        p = _fsrs_p_correct("tvm")
        assert p == pytest.approx(0.3)

    def test_concept_with_stability_none(self, seeded_db):
        import sqlite3

        from engine.analytics.readiness import _fsrs_p_correct
        conn = sqlite3.connect(str(seeded_db))
        conn.execute(
            "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 1, 0, 'learning', NULL, NULL, NULL, NULL, NULL)",
            ("tvm",)
        )
        conn.commit()
        conn.close()
        p = _fsrs_p_correct("tvm")
        assert p == pytest.approx(0.3)  # stability <= 0 branch

    def test_concept_with_due_none(self, seeded_db):
        import sqlite3

        from engine.analytics.readiness import _fsrs_p_correct
        conn = sqlite3.connect(str(seeded_db))
        conn.execute(
            "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 3, 0, 'review', 10.0, NULL, NULL, NULL, NULL)",
            ("tvm",)
        )
        conn.commit()
        conn.close()
        p = _fsrs_p_correct("tvm")
        assert p == pytest.approx(0.5)  # due is None branch

    def test_concept_with_future_due(self, seeded_db):
        import sqlite3
        from datetime import datetime, timedelta, timezone

        from engine.analytics.readiness import _fsrs_p_correct
        future_due = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        conn = sqlite3.connect(str(seeded_db))
        conn.execute(
            "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 3, 0, 'review', 10.0, NULL, NULL, ?, NULL)",
            ("tvm", future_due)
        )
        conn.commit()
        conn.close()
        p = _fsrs_p_correct("tvm")
        assert 0.0 < p <= 1.0


class TestComputeReadiness:
    def test_empty_concepts_returns_zero(self, isolated_db):
        from engine.analytics.readiness import compute_readiness
        score, detail = compute_readiness([])
        assert score == pytest.approx(0.0)
        assert detail == {}

    def test_with_p_correct_dict(self, seeded_db):
        from engine.analytics.readiness import compute_readiness
        concepts = _make_concepts(seeded_db)
        p_correct = {c.id: 0.8 for c in concepts}
        score, detail = compute_readiness(concepts, p_correct)
        assert 0.0 <= score <= 1.0
        assert isinstance(detail, dict)

    def test_without_p_correct_falls_back_to_fsrs(self, seeded_db):
        from engine.analytics.readiness import compute_readiness
        concepts = _make_concepts(seeded_db)
        score, detail = compute_readiness(concepts, p_correct=None)
        assert 0.0 <= score <= 1.0

    def test_category_weights_in_detail(self, seeded_db):
        from engine.analytics.readiness import compute_readiness
        concepts = _make_concepts(seeded_db)
        _, detail = compute_readiness(concepts, {c.id: 0.5 for c in concepts})
        for cat, info in detail.items():
            assert "score" in info
            assert "band_weight" in info
            assert "concepts" in info

    def test_unknown_category_skipped(self, seeded_db):
        from engine.analytics.readiness import compute_readiness
        from engine.db.dao import Concept
        concepts = _make_concepts(seeded_db)
        # Add a concept with unknown category (weight=0 → skipped)
        fake = Concept(
            id="fake", name="Fake", category="unknown_cat",
            exam_weight_tier=2, summary=None, generator=None, prerequisites=[]
        )
        score, detail = compute_readiness(concepts + [fake], {c.id: 0.5 for c in concepts + [fake]})
        assert "unknown_cat" not in detail

    def test_zero_tier_sum_skipped(self, seeded_db):
        """Category with all tier=0 concepts should be skipped."""
        from engine.analytics.readiness import compute_readiness
        from engine.db.dao import Concept
        zero_tier = Concept(
            id="z1", name="Z", category="interest",
            exam_weight_tier=0, summary=None, generator=None, prerequisites=[]
        )
        score, detail = compute_readiness([zero_tier], {"z1": 0.5})
        # interest has band_weight but all tiers are 0 → skipped → score=0 or no detail for interest
        assert score == pytest.approx(0.0) or "interest" not in detail


class TestSnapshots:
    def test_save_and_get_snapshots(self, isolated_db):
        from engine.analytics.readiness import get_snapshots, save_snapshot
        save_snapshot(0.65, {"interest": {"score": 0.65, "band_weight": 0.5, "concepts": []}})
        snaps = get_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["score"] == pytest.approx(0.65)

    def test_snapshots_ordered_oldest_first(self, isolated_db):
        from engine.analytics.readiness import get_snapshots, save_snapshot
        save_snapshot(0.4, {})
        save_snapshot(0.6, {})
        save_snapshot(0.8, {})
        snaps = get_snapshots()
        scores = [s["score"] for s in snaps]
        assert scores == sorted(scores)

    def test_snapshots_limit(self, isolated_db):
        from engine.analytics.readiness import get_snapshots, save_snapshot
        for i in range(10):
            save_snapshot(float(i) / 10, {})
        snaps = get_snapshots(limit=5)
        assert len(snaps) == 5


# charts.py


class TestChartsReadinessOverTime:
    def test_empty_snapshots_returns_png(self):
        from engine.analytics.charts import readiness_over_time
        png = readiness_over_time([])
        assert png[:4] == b"\x89PNG"

    def test_with_data_returns_png(self):
        from engine.analytics.charts import readiness_over_time
        snaps = [
            {"taken_at": "2025-01-01T00:00:00+00:00", "score": 0.4},
            {"taken_at": "2025-01-15T00:00:00+00:00", "score": 0.6},
            {"taken_at": "2025-02-01T00:00:00+00:00", "score": 0.75},
        ]
        png = readiness_over_time(snaps)
        assert png[:4] == b"\x89PNG"


class TestChartsCategoryMastery:
    def test_empty_detail_returns_png(self):
        from engine.analytics.charts import category_mastery
        png = category_mastery({})
        assert png[:4] == b"\x89PNG"

    def test_with_data_returns_png(self):
        from engine.analytics.charts import category_mastery
        detail = {
            "interest": {"score": 0.8, "band_weight": 0.3, "concepts": []},
            "annuity": {"score": 0.5, "band_weight": 0.3, "concepts": []},
            "derivatives": {"score": 0.3, "band_weight": 0.4, "concepts": []},
        }
        png = category_mastery(detail)
        assert png[:4] == b"\x89PNG"


class TestChartsAblation:
    def test_empty_logs_returns_png(self):
        from engine.analytics.charts import ablation_comparison
        png = ablation_comparison([], [])
        assert png[:4] == b"\x89PNG"

    def test_with_data_returns_png(self):
        from engine.analytics.charts import ablation_comparison
        fsrs = [{"step": i, "readiness": i / 10} for i in range(5)]
        dkt = [{"step": i, "readiness": i / 10 + 0.05} for i in range(5)]
        png = ablation_comparison(fsrs, dkt)
        assert png[:4] == b"\x89PNG"


class TestChartsPredictedVsActual:
    def test_no_valid_data_returns_png(self):
        from engine.analytics.charts import predicted_vs_actual
        png = predicted_vs_actual([{"score": 0.7}])  # predicted is None
        assert png[:4] == b"\x89PNG"

    def test_with_data_returns_png(self):
        from engine.analytics.charts import predicted_vs_actual
        exams = [
            {"predicted": 0.6, "score": 0.65},
            {"predicted": 0.75, "score": 0.70},
        ]
        png = predicted_vs_actual(exams)
        assert png[:4] == b"\x89PNG"
