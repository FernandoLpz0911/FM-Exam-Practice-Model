"""Tests for engine/analytics/ablation.py."""
from __future__ import annotations

import random

import pytest
import torch

from engine.analytics.ablation import (
    _dkt_predict_batch,
    _dkt_score,
    _fresh_skill,
    _fsrs_score,
    _simulate,
    run_ablation,
)
from engine.db.dao import Concept
from engine.tracing.model import DKT


def _concept(cid: str, category: str = "interest", tier: int = 2) -> Concept:
    return Concept(
        id=cid,
        name=cid,
        category=category,
        exam_weight_tier=tier,
        summary=None,
        generator=None,
        prerequisites=[],
    )


_CONCEPTS = [_concept("a"), _concept("b"), _concept("c")]
_INDEX = {"a": 0, "b": 1, "c": 2}


class TestFreshSkill:
    def test_all_keys_set_to_base(self):
        skill = _fresh_skill(["x", "y", "z"], 0.5)
        assert skill == {"x": 0.5, "y": 0.5, "z": 0.5}


class TestFsrsScore:
    def test_higher_tier_scores_higher(self):
        reps = {"a": 0, "b": 0}
        c_high = _concept("a", tier=3)
        c_low = _concept("b", tier=1)
        assert _fsrs_score(c_high, reps) > _fsrs_score(c_low, reps)

    def test_more_reps_scores_lower(self):
        c = _concept("a", tier=2)
        assert _fsrs_score(c, {"a": 0}) > _fsrs_score(c, {"a": 10})


class TestDktScore:
    def test_low_p_correct_scores_higher(self):
        reps = {"a": 0}
        c = _concept("a", tier=2)
        score_weak = _dkt_score(c, {"a": 0.1}, reps)
        score_strong = _dkt_score(c, {"a": 0.9}, reps)
        assert score_weak > score_strong

    def test_missing_concept_uses_default(self):
        reps = {"a": 0}
        c = _concept("a", tier=2)
        score = _dkt_score(c, {}, reps)
        assert score == pytest.approx((1.0 - 0.5) * 2)


class TestDktPredictBatch:
    def test_returns_all_concept_indices(self):
        M = 3
        model = DKT(n_concepts=M, hidden=8, layers=1)
        model.eval()
        history = [(0, 1), (1, 0), (2, 1)]
        result = _dkt_predict_batch(model, history, M, torch.device("cpu"))
        assert set(result.keys()) == {0, 1, 2}
        assert all(0.0 <= v <= 1.0 for v in result.values())


class TestSimulate:
    def test_fsrs_policy_returns_correct_step_count(self):
        log = _simulate("fsrs", _CONCEPTS, None, _INDEX, 5, random.Random(42))
        assert len(log) == 5
        assert all("step" in e and "readiness" in e and "concept_id" in e for e in log)

    def test_dkt_policy_with_model(self):
        M = 3
        model = DKT(n_concepts=M, hidden=8, layers=1)
        model.eval()
        log = _simulate("dkt", _CONCEPTS, model, _INDEX, 12, random.Random(42))
        assert len(log) == 12

    def test_dkt_policy_no_model_falls_back_to_fsrs(self):
        log = _simulate("dkt", _CONCEPTS, None, _INDEX, 5, random.Random(42))
        assert len(log) == 5

    def test_device_explicitly_provided(self):
        device = torch.device("cpu")
        M = 3
        model = DKT(n_concepts=M, hidden=8, layers=1)
        model.eval()
        log = _simulate("dkt", _CONCEPTS, model, _INDEX, 3, random.Random(1), device=device)
        assert len(log) == 3

    def test_readiness_non_negative(self):
        log = _simulate("fsrs", _CONCEPTS, None, _INDEX, 4, random.Random(0))
        assert all(e["readiness"] >= 0.0 for e in log)


class TestRunAblation:
    def test_returns_expected_keys(self):
        result = run_ablation(
            n_warmup_sessions=5,
            n_warmup_steps=8,
            n_trial_steps=6,
            hidden=8,
            n_epochs=1,
            seed=42,
        )
        assert set(result.keys()) >= {"fsrs_only", "dkt_hybrid", "n_warmup_interactions"}
        assert len(result["fsrs_only"]) == 6
        assert len(result["dkt_hybrid"]) == 6

    def test_warmup_interaction_count(self):
        result = run_ablation(
            n_warmup_sessions=3,
            n_warmup_steps=5,
            n_trial_steps=4,
            hidden=8,
            n_epochs=1,
            seed=7,
        )
        assert result["n_warmup_interactions"] == 3 * 5

    def test_restores_preexisting_db_path(self, monkeypatch, tmp_path):
        """DB_PATH set before the call must be restored afterward (not popped)."""
        original_db = str(tmp_path / "outer.db")
        monkeypatch.setenv("DB_PATH", original_db)
        run_ablation(
            n_warmup_sessions=2,
            n_warmup_steps=4,
            n_trial_steps=2,
            hidden=8,
            n_epochs=1,
            seed=3,
        )
        import os
        assert os.environ["DB_PATH"] == original_db
