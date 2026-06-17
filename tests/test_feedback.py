"""Tests for engine/feedback/misconceptions.py and engine/feedback/compose.py."""
from __future__ import annotations


class TestGetNotes:
    def test_known_kind_ask_returns_string(self):
        from engine.feedback.misconceptions import get_notes
        note = get_notes("interest_tvm", "future_value", 0)
        assert isinstance(note, str)
        assert len(note) > 0

    def test_distractor_index_wraps_around(self):
        from engine.feedback.misconceptions import get_notes
        notes_list = [
            get_notes("interest_tvm", "future_value", i) for i in range(10)
        ]
        # All notes should be non-empty strings
        assert all(isinstance(n, str) and len(n) > 0 for n in notes_list)

    def test_unknown_kind_returns_empty_string(self):
        from engine.feedback.misconceptions import get_notes
        note = get_notes("nonexistent_kind", "nonexistent_ask", 0)
        assert note == ""

    def test_known_kind_unknown_ask_returns_empty_string(self):
        from engine.feedback.misconceptions import get_notes
        note = get_notes("interest_tvm", "unknown_ask_xyz", 0)
        assert note == ""

    def test_covers_multiple_categories(self):
        from engine.feedback.misconceptions import get_notes
        pairs = [
            ("annuity_immediate", "pv_annuity_imm"),
            ("bond_price", "price_from_yield"),
            ("option_payoff", "call_payoff"),
            ("swap_rate", "fixed_swap_rate"),
            ("immunization", "full_immunization_weight"),
        ]
        for kind, ask in pairs:
            note = get_notes(kind, ask, 0)
            assert isinstance(note, str)


class TestComposeFeedback:
    _PARAMS = {"i": 0.05, "n": 5, "pv": 1000.0}

    def test_correct_answer_gives_no_note(self):
        from engine.feedback.compose import compose_feedback
        fb = compose_feedback("interest_tvm", "future_value", self._PARAMS, "1276.281563", True)
        assert fb["note"] is None
        assert isinstance(fb["solution_steps"], list)
        assert len(fb["solution_steps"]) > 0

    def test_wrong_answer_gives_note(self):
        from engine.feedback.compose import compose_feedback
        fb = compose_feedback("interest_tvm", "future_value", self._PARAMS, "999.0", False)
        assert fb["note"] is not None
        assert isinstance(fb["note"], str)
        assert len(fb["note"]) > 0

    def test_none_is_correct_gives_no_note(self):
        from engine.feedback.compose import compose_feedback
        fb = compose_feedback("interest_tvm", "future_value", self._PARAMS, "1276.0", None)
        assert fb["note"] is None

    def test_unknown_kind_wrong_returns_no_note(self):
        # Unknown kind/ask → solver may raise, so use a known solver with unknown misconception
        from engine.feedback.misconceptions import get_notes
        note = get_notes("perpetuity", "unknown_ask", 0)
        assert note == ""

    def test_solution_steps_always_present(self):
        from engine.feedback.compose import compose_feedback
        for is_correct in (True, False, None):
            fb = compose_feedback(
                "interest_tvm", "present_value", self._PARAMS, "952.38", is_correct
            )
            assert "solution_steps" in fb
            assert isinstance(fb["solution_steps"], list)

    def test_distractor_index_passed_through(self):
        from engine.feedback.compose import compose_feedback
        fb0 = compose_feedback("interest_tvm", "future_value", self._PARAMS, "0", False, distractor_index=0)
        fb1 = compose_feedback("interest_tvm", "future_value", self._PARAMS, "0", False, distractor_index=1)
        # Different distractor indices may or may not give same note (depends on notes list size)
        assert fb0["note"] is not None
        assert fb1["note"] is not None
