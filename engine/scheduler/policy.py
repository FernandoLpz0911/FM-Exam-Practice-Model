"""
Concept-selection policy: picks the next concept to study.

Entry point: next_concept(weakness) → Concept | None

Priority order
--------------
0. Retry queue — exact (kind, ask) of a recently missed problem.
1. Overdue cards — ranked by weakness × exam_weight_tier.
   weakness = 1 − P(correct) from DKT when active, else FSRS retrievability urgency.
2. Frontier (new) concepts — ranked by exam_weight_tier.

Returns None when no concepts are due yet.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from engine.config import PREREQUISITE_WARMTH_DAYS
from engine.db import dao
from engine.db.dao import Concept
from engine.scheduler import mastery, retry, store
from engine.scheduler.fsrs_core import retrievability
from engine.scheduler.store import CardState


def next_concept(
    weakness: dict[str, float] | None = None,
) -> Concept | None:
    """Select the highest-priority concept for the next study item.

    Priority order:
      0. Retry queue — same (kind, ask) as a recently missed problem.
      1. Overdue review cards — ranked by weakness × exam_weight_tier.
         weakness = 1 − P(correct) from DKT when active, else FSRS urgency.
      2. New/frontier concepts — ranked by exam_weight_tier.

    Returns None when everything is due in the future.
    """
    # Error-correction: replay the missed (kind, ask) before FSRS scheduling.
    entry = retry.dequeue()
    if entry is not None:
        concept = _find_concept(entry["concept_id"])
        if concept is not None and concept.generator is not None:
            return _pin_ask(concept, entry["ask"])
        # If concept missing or has no generator, fall through to normal scheduling.

    concepts = dao.get_all_concepts()
    available = mastery.unlocked_concepts(concepts)
    if not available:
        return None

    now = datetime.now(timezone.utc)
    overdue: list[tuple[float, Concept]] = []
    frontier: list[tuple[int, Concept]] = []

    for concept in available:
        card_state = store.get_or_create(concept.id)
        if card_state.reps == 0:
            frontier.append((concept.exam_weight_tier, concept))
        elif card_state.due is not None and card_state.due <= now:
            if weakness is not None:
                urgency = weakness.get(concept.id, 0.5)
            else:
                urgency = _urgency(card_state, now)
            score = urgency * concept.exam_weight_tier
            overdue.append((score, concept))

    if overdue:
        # Boost concepts whose prerequisites are still warm in memory;
        # penalise those whose dependency chain has gone cold.
        scored = [
            (score * _warmth_multiplier(concept, now), concept)
            for score, concept in overdue
        ]
        return max(scored, key=lambda x: x[0])[1]
    if frontier:
        return max(frontier, key=lambda x: x[0])[1]
    return None


def _urgency(cs: CardState, now: datetime) -> float:
    """How urgently a card needs review: 1 - R(elapsed, S), capped at 1."""
    if cs.stability is None or cs.stability <= 0:
        return 1.0
    elapsed = (now - cs.due).total_seconds() / 86400 if cs.due else 0.0
    r = retrievability(elapsed + (cs.stability or 1.0), cs.stability)
    return max(0.0, 1.0 - r)


def _prereqs_warm(concept: Concept, now: datetime) -> bool:
    """True when all prerequisites were reviewed within PREREQUISITE_WARMTH_DAYS."""
    for prereq_id in concept.prerequisites:
        cs = store.get_or_create(prereq_id)
        if cs.last_review is None:
            return False
        lr = cs.last_review
        if lr.tzinfo is None:
            lr = lr.replace(tzinfo=timezone.utc)
        ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        days_since = (ref - lr).total_seconds() / 86400
        if days_since > PREREQUISITE_WARMTH_DAYS:
            return False
    return True


def _warmth_multiplier(concept: Concept, now: datetime) -> float:
    """1.2 if all prerequisites are warm, 0.8 if any have gone cold."""
    if not concept.prerequisites:
        return 1.0
    return 1.2 if _prereqs_warm(concept, now) else 0.8


def _find_concept(concept_id: str) -> Concept | None:
    for c in dao.get_all_concepts():
        if c.id == concept_id:
            return c
    return None


def _pin_ask(concept: Concept, ask: str) -> Concept:
    """Return a shallow copy of concept with generator ask list pinned to [ask]."""
    gen = copy.deepcopy(concept.generator)
    gen["params"]["ask"] = [ask]
    from dataclasses import replace
    return replace(concept, generator=gen)
