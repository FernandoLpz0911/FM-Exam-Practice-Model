"""P5.7 — Generate synthetic interaction data to validate the DKT training pipeline.

Simulates a student who starts weak and improves over time.  Writes directly
to the interaction and session tables so `train.py` can run immediately.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from engine.db.connection import get_connection
from engine.tracing.concept_index import load as load_index


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_synthetic(
    n_sessions: int = 20,
    steps_per_session: int = 30,
    base_accuracy: float = 0.4,
    improvement_rate: float = 0.015,
    rng_seed: int = 42,
) -> int:
    """Insert synthetic interactions into the DB.

    Models a student whose accuracy per concept starts at `base_accuracy` and
    increases by `improvement_rate` per step (capped at 0.95).

    Returns total number of interactions inserted.
    """
    rng = random.Random(rng_seed)
    concept_index = load_index()
    concept_ids = sorted(concept_index.keys())
    len(concept_ids)

    # Per-concept skill: starts low, improves independently
    skill: dict[str, float] = {cid: base_accuracy for cid in concept_ids}

    total = 0
    with get_connection() as conn:
        for s in range(n_sessions):
            started_at = _now_iso()
            cursor = conn.execute(
                "INSERT INTO session (started_at) VALUES (?)", (started_at,)
            )
            session_id = cursor.lastrowid

            for _ in range(steps_per_session):
                cid = rng.choice(concept_ids)
                p_correct = min(0.95, skill[cid])
                is_correct = int(rng.random() < p_correct)

                shown_at = _now_iso()
                cursor = conn.execute(
                    """
                    INSERT INTO interaction
                        (session_id, concept_id, seed, problem_kind, params_json,
                         correct_answer, user_answer, is_correct, grade, elapsed_ms,
                         shown_at, answered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        cid,
                        0,
                        "synthetic:placeholder",
                        "{}",
                        "0.0000",
                        "0.0000",
                        is_correct,
                        3 if is_correct else 1,
                        1000,
                        shown_at,
                        shown_at,
                    ),
                )
                total += 1
                skill[cid] = min(0.95, skill[cid] + improvement_rate)

            conn.execute(
                "UPDATE session SET ended_at = ? WHERE id = ?",
                (_now_iso(), session_id),
            )

    return total
