"""FSRS-5 core math — pure NumPy implementation.

Implements the forgetting curve and interval formula from the FSRS spec.
The stability update functions are left to the `fsrs` library (py-fsrs),
which this module cross-tests against in tests/test_phase2.py.
"""
from __future__ import annotations

DECAY: float = -0.5
FACTOR: float = 19.0 / 81.0  # guarantees R(S, S) = 0.90 exactly


def retrievability(elapsed_days: float, stability: float) -> float:
    """Probability of recall given elapsed days t and current stability S.

    R(0, S) = 1.0 always.
    R(S, S) = 0.90 by definition of stability (that's what FACTOR is tuned for).
    """
    if stability <= 0:
        # A concept with no/invalid stability hasn't been learned yet —
        # treat recall probability as zero rather than dividing by it below.
        return 0.0
    return float((1.0 + FACTOR * elapsed_days / stability) ** DECAY)


def interval_for_target(stability: float, target_retention: float = 0.9) -> int:
    """Days until retrievability drops to target_retention.

    Derived by inverting the forgetting curve for t.
    Result is at least 1 day.
    """
    if stability <= 0:
        return 1
    days = stability / FACTOR * (target_retention ** (1.0 / DECAY) - 1.0)
    return max(1, round(days))
