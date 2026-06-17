"""Tests for engine/generation/difficulty.py."""
from __future__ import annotations

from engine.generation.difficulty import ASK_DIFFICULTY, filter_asks, max_tier_for_reps


class TestMaxTierForReps:
    def test_below_tier2_threshold_returns_one(self):
        assert max_tier_for_reps(0, tier2_threshold=3, tier3_threshold=6) == 1
        assert max_tier_for_reps(2, tier2_threshold=3, tier3_threshold=6) == 1

    def test_at_tier2_threshold_returns_two(self):
        assert max_tier_for_reps(3, tier2_threshold=3, tier3_threshold=6) == 2
        assert max_tier_for_reps(5, tier2_threshold=3, tier3_threshold=6) == 2

    def test_at_tier3_threshold_returns_three(self):
        assert max_tier_for_reps(6, tier2_threshold=3, tier3_threshold=6) == 3
        assert max_tier_for_reps(100, tier2_threshold=3, tier3_threshold=6) == 3


class TestFilterAsks:
    def test_all_tier1_asks_available_at_max_one(self):
        asks = ["future_value", "present_value"]
        result = filter_asks("interest_tvm", asks, max_tier=1)
        assert "future_value" in result
        assert "present_value" in result

    def test_tier2_ask_excluded_at_max_one(self):
        # "interest_rate_solve" is tier 1 for interest_tvm — use a tier2 kind/ask
        # "cdf_from_pdf" is tier 2 for density_basics
        asks = ["normalize_constant", "prob_interval", "cdf_from_pdf"]
        result = filter_asks("density_basics", asks, max_tier=1)
        assert "normalize_constant" in result
        assert "prob_interval" in result
        assert "cdf_from_pdf" not in result

    def test_fallback_to_full_list_when_none_qualify(self):
        # Ask list with only tier-3 items at max_tier=1 → falls back to full list
        asks = ["memoryless"]   # exponential:memoryless is tier 3
        result = filter_asks("exponential", asks, max_tier=1)
        assert result == asks  # full fallback

    def test_unknown_kind_defaults_to_tier_two(self):
        asks = ["some_ask"]
        result = filter_asks("totally_unknown_kind", asks, max_tier=2)
        assert result == asks  # default tier=2 <= max_tier=2

    def test_unknown_kind_excluded_at_max_one(self):
        asks = ["some_ask"]
        result = filter_asks("totally_unknown_kind", asks, max_tier=1)
        # Default tier=2 > max_tier=1 → none qualify → fallback to full list
        assert result == asks

    def test_all_tiers_available_at_max_three(self):
        asks = ["normalize_constant", "cdf_from_pdf"]  # tier 1 and 2
        result = filter_asks("density_basics", asks, max_tier=3)
        assert set(result) == {"normalize_constant", "cdf_from_pdf"}

    def test_ask_difficulty_populated(self):
        assert len(ASK_DIFFICULTY) > 0
        # Dict contains Exam P entries (FM generators default to tier 2 when absent)
        assert ("exponential", "memoryless") in ASK_DIFFICULTY
        assert ASK_DIFFICULTY[("exponential", "memoryless")] == 3
