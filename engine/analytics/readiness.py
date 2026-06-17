"""FM readiness score: DKT P(correct) × category bands × tiers → 0–1.

Formula:
    readiness = Σ_category  band_weight(c) ·
                  (Σ_{c in cat} tier_weight(c) · P_correct(c) /
                   Σ_{c in cat} tier_weight(c))

When DKT is inactive, falls back to FSRS retrievability per concept.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from engine.db.connection import get_connection
from engine.db.dao import Concept

# SOA Exam FM approximate topic weights (sum ≈ 1.0)
BAND_WEIGHT: dict[str, float] = {
    "interest":    0.25,
    "annuity":     0.15,
    "loan":        0.10,
    "bond":        0.10,
    "duration":    0.15,
    "derivatives": 0.25,
}


def _fsrs_p_correct(concept_id: str) -> float:
    """FSRS-based P(correct) fallback when DKT is inactive."""
    from engine.scheduler import store
    from engine.scheduler.fsrs_core import retrievability

    concept_state = store.get_or_create(concept_id)
    if concept_state.reps == 0 or concept_state.stability is None or concept_state.stability <= 0:
        return 0.3  # cold-start prior: no review history to base an estimate on
    if concept_state.due is None:
        return 0.5
    now = datetime.now(timezone.utc)
    days_overdue = max(0.0, (now - concept_state.due).total_seconds() / 86400)
    # Re-anchor elapsed time at the review's scheduled stability so a concept
    # exactly on time reads as "elapsed == stability" in the forgetting curve.
    return float(retrievability(days_overdue + concept_state.stability, concept_state.stability))


def compute_readiness(
    concepts: list[Concept],
    p_correct: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """Compute the readiness score (0–1) and per-category detail.

    Args:
        concepts:  all Concept objects (from dao.get_all_concepts())
        p_correct: {concept_id: probability} from DKT inference; if None,
                   falls back to FSRS retrievability per concept.

    Returns:
        (score, detail) where detail = {category: {score, weight, concepts: [...]}}
    """
    concepts_by_category: dict[str, list[Concept]] = {}
    for concept in concepts:
        concepts_by_category.setdefault(concept.category, []).append(concept)

    # If none of the loaded concepts' categories appear in BAND_WEIGHT, there's
    # nothing to weight against — this is the failure mode that silently zeroed
    # out FM readiness when BAND_WEIGHT still held P-exam category names.
    total_weight = sum(BAND_WEIGHT.get(category, 0.0) for category in concepts_by_category)
    if total_weight <= 0:
        return 0.0, {}

    detail: dict = {}
    readiness = 0.0

    for category, category_concepts in concepts_by_category.items():
        band_weight = BAND_WEIGHT.get(category, 0.0)
        if band_weight == 0.0:
            continue

        tier_weight_sum = sum(concept.exam_weight_tier for concept in category_concepts)
        if tier_weight_sum == 0:
            continue

        category_score = 0.0
        concept_detail = []
        for concept in category_concepts:
            if p_correct is not None:
                p_correct_value = p_correct.get(concept.id, 0.5)
            else:
                p_correct_value = _fsrs_p_correct(concept.id)
            category_score += concept.exam_weight_tier * p_correct_value
            concept_detail.append({
                "id": concept.id, "name": concept.name,
                "p_correct": round(p_correct_value, 4),
            })

        # Normalize by total tier weight so each category's score is itself a
        # weighted average in [0,1], independent of how many concepts it has.
        category_score /= tier_weight_sum
        readiness += (band_weight / total_weight) * category_score
        detail[category] = {
            "score": round(category_score, 4),
            "band_weight": band_weight,
            "concepts": concept_detail,
        }

    return round(readiness, 4), detail


def save_snapshot(score: float, detail: dict) -> int:
    """Persist a readiness snapshot. Returns the new row id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO readiness_snapshot (taken_at, score, detail_json) VALUES (?, ?, ?)",
            (now, score, json.dumps(detail)),
        )
        return cur.lastrowid


def get_snapshots(limit: int = 200) -> list[dict]:
    """Return recent readiness snapshots (oldest first)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, taken_at, score FROM readiness_snapshot "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r["id"], "taken_at": r["taken_at"], "score": r["score"]}
            for r in reversed(rows)]
