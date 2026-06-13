from __future__ import annotations

from engine.config import MASTERY_MIN_REPS, MASTERY_STABILITY_DAYS
from engine.db.dao import Concept
from engine.scheduler import store


def is_mastered(concept_id: str) -> bool:
    """A concept is mastered when it has been reviewed enough times and stability is high."""
    cs = store.get_or_create(concept_id)
    return (
        cs.state == "review"
        and cs.reps >= MASTERY_MIN_REPS
        and (cs.stability or 0.0) >= MASTERY_STABILITY_DAYS
    )


def is_introduced(concept_id: str) -> bool:
    """A concept is introduced once it has been seen at least once."""
    return store.get_or_create(concept_id).reps >= 1


def unlocked_concepts(concepts: list[Concept]) -> list[Concept]:
    """Return concepts whose prerequisites have been introduced at least once.

    Full mastery is not required to unlock dependents — one review is enough.
    The scheduler's warmth multiplier handles prioritisation of warm prereqs.
    """
    introduced: dict[str, bool] = {c.id: is_introduced(c.id) for c in concepts}
    return [c for c in concepts if all(introduced.get(p, False) for p in c.prerequisites)]
