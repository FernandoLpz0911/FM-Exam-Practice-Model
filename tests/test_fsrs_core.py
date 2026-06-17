"""Tests for engine/scheduler/fsrs_core.py — pure FSRS math, no DB."""
from __future__ import annotations

import pytest

from engine.scheduler.fsrs_core import DECAY, FACTOR, interval_for_target, retrievability


class TestRetrievability:
    def test_zero_elapsed_is_one(self):
        assert retrievability(0.0, 10.0) == pytest.approx(1.0)

    def test_at_stability_equals_zero_point_nine(self):
        # R(S, S) = 0.90 by construction
        for s in (1.0, 5.0, 21.0, 100.0):
            r = retrievability(s, s)
            assert r == pytest.approx(0.90, abs=1e-6)

    def test_longer_elapsed_lower_retrieval(self):
        r1 = retrievability(5.0, 10.0)
        r2 = retrievability(15.0, 10.0)
        assert r1 > r2

    def test_zero_stability_returns_zero(self):
        assert retrievability(1.0, 0.0) == pytest.approx(0.0)

    def test_negative_stability_returns_zero(self):
        assert retrievability(1.0, -5.0) == pytest.approx(0.0)

    def test_large_elapsed_very_low(self):
        # Power-law curve: need t >> S; t=100_000, S=1 → R≈0.0065
        r = retrievability(100_000.0, 1.0)
        assert r < 0.01

    def test_constants(self):
        # Verify mathematical relationship: R(S,S)=0.90
        r = (1.0 + FACTOR * 1.0) ** DECAY   # t=S=1
        assert r == pytest.approx(0.90, abs=1e-6)


class TestIntervalForTarget:
    def test_zero_stability_returns_one(self):
        assert interval_for_target(0.0) == 1

    def test_negative_stability_returns_one(self):
        assert interval_for_target(-10.0) == 1

    def test_result_at_least_one(self):
        for s in (0.001, 0.5, 1.0, 30.0):
            assert interval_for_target(s) >= 1

    def test_higher_stability_longer_interval(self):
        i1 = interval_for_target(1.0)
        i2 = interval_for_target(30.0)
        assert i2 > i1

    def test_default_target_is_point_nine(self):
        # At target=0.90, R(interval, S)=0.90 → interval≈S (integer rounding)
        for s in (7.0, 14.0, 21.0):
            interval = interval_for_target(s, 0.90)
            r = retrievability(float(interval), s)
            # Should be close to 0.90
            assert abs(r - 0.90) < 0.05

    def test_custom_target_lower_gives_longer_interval(self):
        # Lower retention target → wait longer before review
        i_strict = interval_for_target(10.0, 0.95)
        i_relaxed = interval_for_target(10.0, 0.70)
        assert i_relaxed > i_strict
