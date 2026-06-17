"""
FSRS card state: persistence and rating application.

``CardState`` mirrors one row in the ``card_state`` table.  The three public
functions form the full lifecycle:

    cs = get_or_create(concept_id)   # load (or default) from DB
    cs = apply_rating(cs, rating)    # compute updated state via py-fsrs
    save(cs)                         # write back to DB
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State

from engine.config import EARLY_REINFORCEMENT_REPS, TARGET_RETENTION
from engine.db.connection import get_connection

_scheduler = Scheduler(desired_retention=TARGET_RETENTION)


@dataclass
class CardState:
    concept_id: str
    stability: float | None = None
    difficulty: float | None = None
    last_review: datetime | None = None
    due: datetime | None = None
    reps: int = 0
    lapses: int = 0
    step: int | None = None
    state: str = "learning"


def get_or_create(concept_id: str) -> CardState:
    """Load card state from DB, or return a fresh default if first encounter."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM card_state WHERE concept_id = ?", (concept_id,)
        ).fetchone()
    if row is None:
        return CardState(concept_id=concept_id)
    return CardState(
        concept_id=concept_id,
        stability=row["stability"],
        difficulty=row["difficulty"],
        last_review=_parse_dt(row["last_review"]),
        due=_parse_dt(row["due"]),
        reps=row["reps"],
        lapses=row["lapses"],
        step=row["step"],
        state=row["state"],
    )


def apply_rating(card_state: CardState, rating: int) -> CardState:
    """Run a py-fsrs review and return the updated card state (not yet persisted)."""
    fsrs_card = _to_fsrs_card(card_state)
    updated, _ = _scheduler.review_card(fsrs_card, Rating(rating))
    was_lapse = card_state.state == "review" and Rating(rating) == Rating.Again
    new_reps = card_state.reps + 1

    due = updated.due
    # Early reinforcement: keep new concepts on a daily schedule for the first
    # EARLY_REINFORCEMENT_REPS reviews. FSRS schedules "Easy" immediately to
    # 8+ days, which is too sparse for procedural math skill formation.
    if new_reps < EARLY_REINFORCEMENT_REPS and due is not None:
        cap = datetime.now(timezone.utc) + timedelta(days=1)
        # `due` (from py-fsrs) and `cap` must both be naive or both be aware —
        # comparing them mixed raises TypeError, so match cap's awareness to
        # whatever py-fsrs gave us for `due` rather than forcing one or the other.
        if due.tzinfo is None:
            cap = datetime.now() + timedelta(days=1)
        if due > cap:
            due = cap

    return CardState(
        concept_id=card_state.concept_id,
        stability=updated.stability,
        difficulty=updated.difficulty,
        last_review=updated.last_review,
        due=due,
        reps=new_reps,
        lapses=card_state.lapses + (1 if was_lapse else 0),
        step=updated.step,
        state=updated.state.name.lower(),
    )


def save(card_state: CardState) -> None:
    """Upsert card state to DB (INSERT OR REPLACE on concept_id primary key)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO card_state
                (concept_id, stability, difficulty, last_review, due,
                 reps, lapses, step, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card_state.concept_id,
                card_state.stability,
                card_state.difficulty,
                _fmt_dt(card_state.last_review),
                _fmt_dt(card_state.due),
                card_state.reps,
                card_state.lapses,
                card_state.step,
                card_state.state,
            ),
        )


def _to_fsrs_card(card_state: CardState) -> FsrsCard:
    """Translate our persisted CardState into the py-fsrs library's own Card type."""
    card = FsrsCard()
    card.state = State[card_state.state.capitalize()]
    card.step = card_state.step if card_state.step is not None else 0
    card.stability = card_state.stability
    card.difficulty = card_state.difficulty
    if card_state.last_review is not None:
        card.last_review = card_state.last_review
    if card_state.due is not None:
        card.due = card_state.due
    return card


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _fmt_dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None
