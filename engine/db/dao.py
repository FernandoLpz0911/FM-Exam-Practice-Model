"""Data-access layer — all SQLite reads/writes go through here."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from engine.db.connection import get_connection


@dataclass
class Concept:
    """One node in the concept graph."""

    id: str
    name: str
    category: str
    exam_weight_tier: int
    summary: str | None
    generator: dict | None
    prerequisites: list[str]
    theory_md: str | None = None


def _row_to_concept(row: object, prereqs: list[str]) -> Concept:
    return Concept(
        id=row["id"],
        name=row["name"],
        category=row["category"],
        exam_weight_tier=row["exam_weight_tier"],
        summary=row["summary"],
        generator=(
            json.loads(row["generator_json"])
            if row["generator_json"]
            else None
        ),
        prerequisites=prereqs,
        theory_md=row["theory_md"] if "theory_md" in row.keys() else None,
    )


def get_all_concepts() -> list[Concept]:
    """Return all concepts with prerequisite lists. Two queries total."""
    with get_connection() as conn:
        concept_rows = conn.execute(
            "SELECT * FROM concept ORDER BY id"
        ).fetchall()
        prereq_rows = conn.execute(
            "SELECT * FROM concept_prereq"
        ).fetchall()

    prereqs_by_concept_id: dict[str, list[str]] = {}
    for row in prereq_rows:
        prereqs_by_concept_id.setdefault(row["concept_id"], []).append(row["prereq_id"])

    return [
        _row_to_concept(row, prereqs_by_concept_id.get(row["id"], []))
        for row in concept_rows
    ]


def get_concept(concept_id: str) -> Concept | None:
    """Return a single concept with prerequisites, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM concept WHERE id = ?", (concept_id,)
        ).fetchone()
        if row is None:
            return None
        prereq_rows = conn.execute(
            "SELECT prereq_id FROM concept_prereq WHERE concept_id = ?",
            (concept_id,),
        ).fetchall()

    return _row_to_concept(row, [r["prereq_id"] for r in prereq_rows])


def create_session() -> tuple[int, str]:
    """Create a new study session. Returns (session_id, started_at ISO)."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO session (started_at) VALUES (?)", (now,)
        )
        return cursor.lastrowid, now


def close_session(session_id: int) -> str:
    """Mark session ended. Returns ended_at ISO string."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE session SET ended_at = ? WHERE id = ?",
            (now, session_id),
        )
    return now


def log_shown(
    session_id: int,
    concept_id: str,
    seed: int = 0,
    problem_kind: str = "placeholder",
    params_json: str = "{}",
    correct_answer: str = "N/A",
) -> int:
    """Insert interaction row when problem is served. Returns item_id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO interaction
                (session_id, concept_id, seed, problem_kind, params_json,
                 correct_answer, shown_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                concept_id,
                seed,
                problem_kind,
                params_json,
                correct_answer,
                now,
            ),
        )
        return cursor.lastrowid


def log_answered(
    item_id: int,
    user_answer: str | None,
    is_correct: bool | None,
    grade: int,
    elapsed_ms: int,
) -> None:
    """Update interaction row when student answers."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE interaction
            SET user_answer = ?, is_correct = ?, grade = ?,
                elapsed_ms = ?, answered_at = ?
            WHERE id = ?
            """,
            (
                user_answer,
                int(is_correct) if is_correct is not None else None,
                grade,
                elapsed_ms,
                now,
                item_id,
            ),
        )


def get_interaction_concept(item_id: int) -> str | None:
    """Return concept_id for a given interaction id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT concept_id FROM interaction WHERE id = ?", (item_id,)
        ).fetchone()
    return row["concept_id"] if row else None


def get_interaction_correct_answer(item_id: int) -> str | None:
    """Return stored correct_answer for a given interaction id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT correct_answer FROM interaction WHERE id = ?",
            (item_id,),
        ).fetchone()
    return row["correct_answer"] if row else None


def get_interaction_details(item_id: int) -> dict | None:
    """Return problem_kind and params dict for a given interaction id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT problem_kind, params_json FROM interaction WHERE id = ?",
            (item_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "problem_kind": row["problem_kind"],
        "params": json.loads(row["params_json"]),
    }


def get_interaction_history(
    limit: int = 1000,
) -> list[tuple[str, bool]]:
    """Return recent (concept_id, is_correct) pairs for DKT, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT concept_id, is_correct FROM interaction
            WHERE is_correct IS NOT NULL
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        (row["concept_id"], bool(row["is_correct"]))
        for row in reversed(rows)
    ]


def log_sample_exam(
    score: float,
    n_questions: int | None = None,
    passing_score: float = 0.70,
    predicted: float | None = None,
    notes: str | None = None,
) -> int:
    """Record a sample exam result. Returns new row id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO sample_exam
               (taken_at, score, n_questions, passing_score, predicted, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, score, n_questions, passing_score, predicted, notes),
        )
        return cur.lastrowid


def get_sample_exams() -> list[dict]:
    """Return all sample exam records, oldest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sample_exam ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def get_journal_stats() -> dict:
    """Aggregate study stats from the interaction log."""
    with get_connection() as conn:
        total_answered = conn.execute(
            "SELECT COUNT(*) AS n FROM interaction "
            "WHERE is_correct IS NOT NULL"
        ).fetchone()["n"]
        total_correct = conn.execute(
            "SELECT COUNT(*) AS n FROM interaction WHERE is_correct = 1"
        ).fetchone()["n"]
        total_ms = conn.execute(
            "SELECT COALESCE(SUM(elapsed_ms), 0) AS ms FROM interaction"
        ).fetchone()["ms"]
        sessions_done = conn.execute(
            "SELECT COUNT(*) AS n FROM session WHERE ended_at IS NOT NULL"
        ).fetchone()["n"]
        concepts_touched = conn.execute(
            "SELECT COUNT(DISTINCT concept_id) AS n FROM interaction "
            "WHERE is_correct IS NOT NULL"
        ).fetchone()["n"]
    return {
        "total_answered": total_answered,
        "total_correct": total_correct,
        "accuracy": (
            round(total_correct / total_answered, 4)
            if total_answered
            else 0.0
        ),
        "total_hours": round(total_ms / 3_600_000, 2),
        "sessions_completed": sessions_done,
        "concepts_touched": concepts_touched,
    }


def count_answered_interactions() -> int:
    """Return total answered interactions (for DKT gate check)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM interaction "
            "WHERE is_correct IS NOT NULL"
        ).fetchone()
    return row["n"] if row else 0
