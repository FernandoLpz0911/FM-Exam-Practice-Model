"""P7.1 — Readiness score: DKT P(correct) × category bands × tiers → 0–1.

Formula (from KNOWLEDGE_TRACING_DKT.md §5):
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

# SOA Exam P weight band midpoints per category
BAND_WEIGHT: dict[str, float] = {
    "general": 0.135,       # midpoint of [0.10, 0.17]
    "univariate": 0.435,    # midpoint of [0.40, 0.47]
    "multivariate": 0.435,  # midpoint of [0.40, 0.47]
}


def _fsrs_p_correct(concept_id: str) -> float:
    """FSRS-based P(correct) fallback when DKT is inactive."""
    from engine.scheduler import store
    from engine.scheduler.fsrs_core import retrievability

    cs = store.get_or_create(concept_id)
    if cs.reps == 0 or cs.stability is None or cs.stability <= 0:
        return 0.3  # cold-start prior
    if cs.due is None:
        return 0.5
    now = datetime.now(timezone.utc)
    elapsed = max(0.0, (now - cs.due).total_seconds() / 86400)
    return float(retrievability(elapsed + cs.stability, cs.stability))


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
    by_cat: dict[str, list[Concept]] = {}
    for c in concepts:
        by_cat.setdefault(c.category, []).append(c)

    total_weight = sum(BAND_WEIGHT.get(cat, 0.0) for cat in by_cat)
    if total_weight <= 0:
        return 0.0, {}

    detail: dict = {}
    readiness = 0.0

    for cat, cat_concepts in by_cat.items():
        band_w = BAND_WEIGHT.get(cat, 0.0)
        if band_w == 0.0:
            continue

        tier_sum = sum(c.exam_weight_tier for c in cat_concepts)
        if tier_sum == 0:
            continue

        cat_score = 0.0
        concept_detail = []
        for c in cat_concepts:
            if p_correct is not None:
                p = p_correct.get(c.id, 0.5)
            else:
                p = _fsrs_p_correct(c.id)
            cat_score += c.exam_weight_tier * p
            concept_detail.append({"id": c.id, "name": c.name, "p_correct": round(p, 4)})

        cat_score /= tier_sum
        readiness += (band_w / total_weight) * cat_score
        detail[cat] = {
            "score": round(cat_score, 4),
            "band_weight": band_w,
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
