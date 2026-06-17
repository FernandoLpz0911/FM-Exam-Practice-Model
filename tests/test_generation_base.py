"""Tests for engine/generation/base.py — generator framework."""
from __future__ import annotations

import numpy as np
import pytest

import engine.generation  # noqa: F401 — ensures all kinds are registered
from engine.generation.base import (
    Problem,
    generate,
    list_kinds,
    make_mc_choices,
    pick_ask,
    register,
)


class TestRegisterAndGenerate:
    def test_generate_raises_for_unknown_kind(self):
        with pytest.raises(ValueError, match="No generator registered"):
            generate("nonexistent_kind_xyz", "ask", {}, seed=0)

    def test_list_kinds_contains_all_expected(self):
        kinds = set(list_kinds())
        assert "interest_tvm" in kinds
        assert "annuity_immediate" in kinds
        assert "bond_price" in kinds
        assert "swap_rate" in kinds

    def test_register_decorator(self):
        @register("_test_kind_tmp")
        def _gen(ask, ranges, seed):
            return Problem(kind="_test_kind_tmp", ask=ask, statement="test",
                           correct_answer=1.0, choices=None)

        assert "_test_kind_tmp" in list_kinds()
        prob = generate("_test_kind_tmp", "x", {}, seed=0)
        assert prob.correct_answer == pytest.approx(1.0)


class TestPickAsk:
    def test_returns_valid_ask(self):
        ask_list = ["future_value", "present_value", "interest_rate_solve"]
        ask, seed = pick_ask(ask_list)
        assert ask in ask_list
        assert isinstance(seed, int)

    def test_single_item_list(self):
        ask, _ = pick_ask(["only_one"])
        assert ask == "only_one"

    def test_different_seeds_can_produce_different_asks(self):
        asks = set()
        for _ in range(50):
            a, _ = pick_ask(["a", "b", "c"])
            asks.add(a)
        # With 50 draws from 3 options, should see at least 2
        assert len(asks) >= 2


class TestMakeMcChoices:
    def _rng(self, seed=42):
        return np.random.default_rng(seed)

    def test_always_four_choices(self):
        choices = make_mc_choices(100.0, [90.0, 110.0, 120.0], self._rng())
        assert len(choices) == 4

    def test_correct_answer_in_choices(self):
        correct = 123.4567
        choices = make_mc_choices(correct, [100.0, 200.0, 300.0], self._rng())
        assert f"{correct:.4f}" in choices

    def test_no_duplicate_choices(self):
        choices = make_mc_choices(100.0, [90.0, 110.0, 120.0], self._rng())
        assert len(choices) == len(set(choices))

    def test_fallback_perturbation_when_fewer_than_three_wrongs(self):
        # Only 1 wrong provided → fallback to random perturbation
        choices = make_mc_choices(100.0, [90.0], self._rng())
        assert len(choices) == 4
        assert f"{100.0:.4f}" in choices

    def test_wrong_matching_target_is_excluded(self):
        correct = 100.0
        # wrong that formats the same as correct
        wrongs = [100.00001, 110.0, 120.0]
        choices = make_mc_choices(correct, wrongs, self._rng())
        assert choices.count(f"{correct:.4f}") == 1  # exactly once

    def test_custom_decimals(self):
        choices = make_mc_choices(100.0, [90.0, 110.0, 120.0], self._rng(), decimals=2)
        assert all("." in c for c in choices)
        assert f"{100.0:.2f}" in choices

    def test_zero_correct_answer_handled(self):
        choices = make_mc_choices(0.0, [0.1, -0.1, 0.2], self._rng())
        assert len(choices) == 4

    def test_small_correct_answer_perturbation(self):
        # Very small value — scale = max(abs(correct), 0.01) = 0.01
        choices = make_mc_choices(0.001, [], self._rng())
        assert len(choices) == 4

    def test_all_choices_are_strings(self):
        choices = make_mc_choices(42.0, [10.0, 20.0, 30.0], self._rng())
        assert all(isinstance(c, str) for c in choices)

    def test_perturbation_collision_triggers_false_branch(self):
        """While-loop if-condition False branch (line 85→87): duplicate perturbed value."""
        from unittest.mock import MagicMock

        rng = MagicMock()
        # wrongs=[0.005] → candidates=["0.0050"] (len=1); while loop runs.
        # 1st perturbation: u=0.5 → perturbed=0.005 → s="0.0050" already in candidates
        #   → condition False → branch 85→87 taken
        # 2nd perturbation: u=-0.3 → perturbed=-0.003 → new → candidates has 2
        # 3rd perturbation: u=0.2  → perturbed=0.002  → new → candidates has 3 → exit
        rng.uniform.side_effect = [0.5, -0.3, 0.2]
        rng.permutation.return_value = np.array([0, 1, 2, 3])
        choices = make_mc_choices(0.0, [0.005], rng)
        assert len(choices) == 4
        assert "0.0000" in choices


class TestProblemDataclass:
    def test_defaults(self):
        p = Problem(
            kind="k", ask="a", statement="s", correct_answer=1.0, choices=None
        )
        assert p.tolerance == pytest.approx(1e-4)
        assert p.params == {}
        assert p.seed == 0
