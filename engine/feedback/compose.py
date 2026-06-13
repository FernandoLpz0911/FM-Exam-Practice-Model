"""
Compose full feedback from (problem_kind, ask, params, user_answer, is_correct).
Returns {note, solution_steps} — note is None when the answer is correct.
"""
from __future__ import annotations

from engine.feedback.misconceptions import get_notes
from engine.feedback.solve import solve


def compose_feedback(
    kind: str,
    ask: str,
    params: dict,
    user_answer: str,
    is_correct: bool | None,
    distractor_index: int = 0,
) -> dict:
    """Build feedback for one answered problem.

    Returns:
        note           — misconception note (str) if wrong, else None
        solution_steps — list of step strings (always present)
    """
    solved = solve(kind, ask, params)
    note: str | None = None
    if is_correct is False:
        note = get_notes(kind, ask, distractor_index) or None
    return {"note": note, "solution_steps": solved.steps}
