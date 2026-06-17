from __future__ import annotations

from engine.config import MASTERY_MIN_REPS, MASTERY_STABILITY_DAYS
from engine.db.dao import Concept
from engine.scheduler import store


def is_mastered(concept_id: str) -> bool:
    """A concept is mastered when it has been reviewed enough times and stability is high."""
    concept_state = store.get_or_create(concept_id)
    return (
        concept_state.state == "review"
        and concept_state.reps >= MASTERY_MIN_REPS
        and (concept_state.stability or 0.0) >= MASTERY_STABILITY_DAYS
    )


def is_introduced(concept_id: str) -> bool:
    """A concept is introduced once it has been seen at least once."""
    return store.get_or_create(concept_id).reps >= 1


def unlocked_concepts(concepts: list[Concept]) -> list[Concept]:
    """Return concepts whose prerequisites have been introduced at least once.

    Full mastery is not required to unlock dependents — one review is enough.
    The scheduler's warmth multiplier handles prioritisation of warm prereqs.
    """
    introduced_by_id: dict[str, bool] = {
        concept.id: is_introduced(concept.id) for concept in concepts
    }
    return [
        concept for concept in concepts
        if all(introduced_by_id.get(prereq_id, False) for prereq_id in concept.prerequisites)
    ]
