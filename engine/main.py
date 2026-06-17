"""FastAPI application — HTTP entry point for the FM Exam engine.

Mirrors the P engine structure. Runs on port 8001 to avoid conflict with
the P engine (port 8000). Categories: interest, annuity, loan, bond, duration, derivatives.
"""
import json
import secrets
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.analytics import charts
from engine.analytics import readiness as readiness_mod
from engine.config import (
    DIFFICULTY_TIER2_REPS,
    DIFFICULTY_TIER3_REPS,
    DKT_MIN_AUC,
    DKT_MIN_INTERACTIONS,
)
from engine.db import dao
from engine.db.connection import get_connection, init_db
from engine.feedback.compose import compose_feedback
from engine.feedback.solve import solve as _solve
from engine.generation.base import generate, pick_ask
from engine.generation.difficulty import filter_asks, max_tier_for_reps
from engine.scheduler import policy, retry, store
from engine.tracing import infer as dkt_infer
from engine.tracing.train import train as dkt_train


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Exam FM Engine", version="0.1.0", lifespan=_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],   # FM frontend on 5174
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionOut(BaseModel):
    session_id: int
    started_at: str


class NextOut(BaseModel):
    item_id: int | None
    concept_id: str | None
    concept_name: str | None
    statement: str | None
    choices: list[str] | None
    message: str | None
    is_new_concept: bool = False


class AnswerIn(BaseModel):
    item_id: int
    user_answer: str = ""
    grade: int | None = None
    elapsed_ms: int = 0


class AnswerOut(BaseModel):
    is_correct: bool | None
    correct_answer: str | None
    note: str | None = None
    solution_steps: list[str] = []
    next_review_at: str


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"cuda": torch.cuda.is_available(), "db": _db_reachable()}


@app.get("/concepts")
def get_concepts() -> dict:
    """Return all concept nodes and prerequisite edges."""
    concepts = dao.get_all_concepts()
    edges = [
        {"concept_id": c.id, "prereq_id": prereq}
        for c in concepts for prereq in c.prerequisites
    ]
    return {
        "nodes": [
            {
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "exam_weight_tier": c.exam_weight_tier,
                "summary": c.summary,
                "prerequisites": c.prerequisites,
            }
            for c in concepts
        ],
        "edges": edges,
    }


@app.get("/concept/{concept_id}")
def get_concept(concept_id: str) -> dict:
    concept = dao.get_concept(concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="concept not found")
    return {
        "id": concept.id,
        "name": concept.name,
        "category": concept.category,
        "exam_weight_tier": concept.exam_weight_tier,
        "summary": concept.summary,
        "prerequisites": concept.prerequisites,
        "theory_md": concept.theory_md,
    }


@app.post("/session", response_model=SessionOut)
def create_session() -> SessionOut:
    session_id, started_at = dao.create_session()
    return SessionOut(session_id=session_id, started_at=started_at)


@app.post("/session/{session_id}/close")
def close_session(session_id: int) -> dict:
    ended_at = dao.close_session(session_id)
    return {"ended_at": ended_at}


@app.get("/next", response_model=NextOut)
def get_next(session_id: int) -> NextOut:
    """Select next concept via FSRS policy (DKT-weighted when active) and serve a problem."""
    weakness: dict[str, float] | None = None
    if dkt_infer.dkt_is_active():
        history = dao.get_interaction_history()
        dkt_predictions = dkt_infer.predict(history)
        if dkt_predictions:
            # Invert P(correct) so higher weakness → concept needs more practice.
            weakness = {cid: 1.0 - p for cid, p in dkt_predictions.items()}

    concept = policy.next_concept(weakness=weakness)
    if concept is None:
        return NextOut(
            item_id=None, concept_id=None, concept_name=None,
            statement=None, choices=None,
            message="No concepts due. Check back later.",
        )

    card_state = store.get_or_create(concept.id)
    is_new_concept = card_state.reps == 0

    if concept.generator is not None:
        generator_spec = concept.generator
        max_difficulty_tier = max_tier_for_reps(
            card_state.reps, DIFFICULTY_TIER2_REPS, DIFFICULTY_TIER3_REPS
        )
        eligible_asks = filter_asks(
            generator_spec["kind"], generator_spec["params"]["ask"], max_difficulty_tier
        )
        ask, _ = pick_ask(eligible_asks)
        problem_seed = secrets.randbelow(2 ** 31)
        problem = generate(generator_spec["kind"], ask, generator_spec["params"], problem_seed)
        correct_answer_value = problem.correct_answer
        correct_answer_str = (
            f"{correct_answer_value:.6f}"
            if isinstance(correct_answer_value, (int, float))
            else str(correct_answer_value)
        )
        item_id = dao.log_shown(
            session_id=session_id,
            concept_id=concept.id,
            seed=problem_seed,
            problem_kind=f"{generator_spec['kind']}:{ask}",
            params_json=json.dumps(problem.params),
            correct_answer=correct_answer_str,
        )
        return NextOut(
            item_id=item_id,
            concept_id=concept.id,
            concept_name=concept.name,
            statement=problem.statement,
            choices=problem.choices,
            message=None,
            is_new_concept=is_new_concept,
        )

    # Concepts without a generator (theory-only nodes) get a plain review prompt.
    item_id = dao.log_shown(session_id, concept.id)
    return NextOut(
        item_id=item_id,
        concept_id=concept.id,
        concept_name=concept.name,
        statement=f"{concept.name} — {concept.summary or '(review this concept)'}",
        choices=None,
        message=None,
        is_new_concept=is_new_concept,
    )


@app.post("/answer", response_model=AnswerOut)
def post_answer(body: AnswerIn) -> AnswerOut:
    """Grade an answer, update the FSRS card, return deterministic feedback."""
    concept_id = dao.get_interaction_concept(body.item_id)
    if concept_id is None:
        raise HTTPException(status_code=404, detail="item_id not found")

    correct_answer_str = dao.get_interaction_correct_answer(body.item_id)

    if body.grade is not None:
        if body.grade not in (1, 2, 3, 4):
            raise HTTPException(status_code=422, detail="grade must be 1–4")
        grade = body.grade
        is_correct = None
    elif correct_answer_str and correct_answer_str != "N/A":
        is_correct = _grade_answer(body.user_answer, correct_answer_str)
        # FSRS rating: 3=Good (answered correctly), 1=Again (answered wrong).
        grade = 3 if is_correct else 1
    else:
        is_correct = None
        grade = 3

    card_state = store.get_or_create(concept_id)
    updated = store.apply_rating(card_state, grade)
    store.save(updated)

    dao.log_answered(
        item_id=body.item_id,
        user_answer=body.user_answer or None,
        is_correct=is_correct,
        grade=grade,
        elapsed_ms=body.elapsed_ms,
    )

    note: str | None = None
    solution_steps: list[str] = []
    interaction_details = dao.get_interaction_details(body.item_id)
    if interaction_details and ":" in interaction_details["problem_kind"]:
        kind, ask = interaction_details["problem_kind"].split(":", 1)
        feedback = compose_feedback(
            kind, ask, interaction_details["params"], body.user_answer, is_correct
        )
        note = feedback["note"]
        solution_steps = feedback["solution_steps"]
        if is_correct is False:
            # Enqueue missed (kind, ask) so the scheduler replays it before
            # moving on — immediate error-correction reinforces the skill.
            retry.enqueue(concept_id, kind, ask)

    return AnswerOut(
        is_correct=is_correct,
        correct_answer=correct_answer_str,
        note=note,
        solution_steps=solution_steps,
        next_review_at=updated.due.isoformat() if updated.due else "",
    )


class HintOut(BaseModel):
    hint_steps: list[str]
    total: int


@app.get("/hint/{item_id}", response_model=HintOut)
def get_hint(item_id: int) -> HintOut:
    details = dao.get_interaction_details(item_id)
    if not details or ":" not in details["problem_kind"]:
        raise HTTPException(status_code=404, detail="No hint available")
    kind, ask = details["problem_kind"].split(":", 1)
    solved = _solve(kind, ask, details["params"])
    hint_steps = solved.steps[:-1] if len(solved.steps) > 1 else []
    return HintOut(hint_steps=hint_steps, total=len(hint_steps))


class SolutionOut(BaseModel):
    steps: list[str]
    correct_answer: str | None


@app.get("/solution/{item_id}", response_model=SolutionOut)
def get_solution(item_id: int) -> SolutionOut:
    details = dao.get_interaction_details(item_id)
    if not details or ":" not in details["problem_kind"]:
        raise HTTPException(status_code=404, detail="No solution available")
    kind, ask = details["problem_kind"].split(":", 1)
    solved = _solve(kind, ask, details["params"])
    correct_answer = dao.get_interaction_correct_answer(item_id)
    return SolutionOut(steps=solved.steps, correct_answer=correct_answer)


class TrainOut(BaseModel):
    val_auc: float | None
    best_epoch: int | None
    n_interactions: int
    epochs_run: int
    checkpoint_path: str | None
    dkt_active: bool
    error: str | None = None


@app.post("/train", response_model=TrainOut)
def post_train() -> TrainOut:
    training_result = dkt_train()
    val_auc = training_result.get("val_auc")
    return TrainOut(
        # `val_auc != val_auc` is the float NaN check (NaN is the only value not equal to itself).
        val_auc=val_auc if val_auc == val_auc else None,
        best_epoch=training_result.get("best_epoch"),
        n_interactions=training_result["n_interactions"],
        epochs_run=training_result["epochs_run"],
        checkpoint_path=training_result.get("checkpoint_path"),
        dkt_active=dkt_infer.dkt_is_active(),
        error=training_result.get("error"),
    )


@app.get("/state")
def get_state() -> dict:
    concepts = dao.get_all_concepts()
    states = []
    for concept in concepts:
        card_state = store.get_or_create(concept.id)
        states.append({
            "concept_id": concept.id,
            "concept_name": concept.name,
            "reps": card_state.reps,
            "lapses": card_state.lapses,
            "stability": card_state.stability,
            "due": card_state.due.isoformat() if card_state.due else None,
            "state": card_state.state,
        })
    return {"states": states}


@app.get("/dkt/status")
def get_dkt_status() -> dict:
    n = dao.count_answered_interactions()
    meta = dkt_infer.checkpoint_meta()
    return {
        "n_interactions": n,
        "dkt_min_interactions": DKT_MIN_INTERACTIONS,
        "dkt_min_auc": DKT_MIN_AUC,
        "checkpoint": meta,
        "dkt_active": dkt_infer.dkt_is_active(),
    }


@app.get("/readiness")
def get_readiness(snapshot: bool = True) -> dict:
    concepts = dao.get_all_concepts()
    p_correct: dict[str, float] | None = None
    if dkt_infer.dkt_is_active():
        history = dao.get_interaction_history()
        dkt_predictions = dkt_infer.predict(history)
        p_correct = dkt_predictions
    score, detail = readiness_mod.compute_readiness(concepts, p_correct)
    if snapshot:
        readiness_mod.save_snapshot(score, detail)
    return {"score": score, "dkt_active": p_correct is not None, "detail": detail}


@app.get("/readiness/history")
def get_readiness_history() -> dict:
    return {"snapshots": readiness_mod.get_snapshots()}


@app.get("/readiness/chart/timeline")
def get_chart_timeline() -> object:
    from fastapi.responses import Response
    png = charts.readiness_over_time(readiness_mod.get_snapshots())
    return Response(content=png, media_type="image/png")


@app.get("/readiness/chart/category")
def get_chart_category() -> object:
    from fastapi.responses import Response
    concepts = dao.get_all_concepts()
    p_correct = None
    if dkt_infer.dkt_is_active():
        p_correct = dkt_infer.predict(dao.get_interaction_history())
    _, detail = readiness_mod.compute_readiness(concepts, p_correct)
    return Response(content=charts.category_mastery(detail), media_type="image/png")


class SampleExamIn(BaseModel):
    score: float
    n_questions: int | None = None
    passing_score: float = 0.70
    notes: str | None = None


@app.post("/sample-exam")
def post_sample_exam(body: SampleExamIn) -> dict:
    predicted: float | None = None
    if dkt_infer.dkt_is_active():
        p_correct = dkt_infer.predict(dao.get_interaction_history())
        if p_correct:
            concepts = dao.get_all_concepts()
            score, _ = readiness_mod.compute_readiness(concepts, p_correct)
            predicted = score
    row_id = dao.log_sample_exam(
        score=body.score, n_questions=body.n_questions,
        passing_score=body.passing_score, predicted=predicted, notes=body.notes,
    )
    return {"id": row_id, "predicted": predicted}


@app.get("/journal")
def get_journal() -> dict:
    stats = dao.get_journal_stats()
    exams = dao.get_sample_exams()
    snapshots = readiness_mod.get_snapshots(limit=10)
    return {"stats": stats, "sample_exams": exams, "recent_readiness": snapshots}


@app.get("/analytics/timing")
def get_timing() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT concept_id, COUNT(*) AS n, AVG(elapsed_ms) AS avg_ms,
                   SUM(CASE WHEN elapsed_ms > 360000 THEN 1 ELSE 0 END) AS n_over
            FROM interaction
            WHERE elapsed_ms IS NOT NULL AND elapsed_ms > 0
              AND user_answer IS NOT NULL
            GROUP BY concept_id ORDER BY avg_ms DESC
            """
        ).fetchall()
    concepts = {c.id: c.name for c in dao.get_all_concepts()}
    return [
        {
            "concept_id": row["concept_id"],
            "concept_name": concepts.get(row["concept_id"], row["concept_id"]),
            "avg_s": round(row["avg_ms"] / 1000, 1),
            "n_answered": row["n"],
            "pct_over_target": round(row["n_over"] / row["n"] * 100, 1) if row["n"] > 0 else 0.0,
        }
        for row in rows
    ]


class MockExamAnswerIn(BaseModel):
    item_id: int
    user_answer: str = ""


class MockExamGradeIn(BaseModel):
    session_id: int
    answers: list[MockExamAnswerIn]
    elapsed_s: int = 0


@app.post("/mock-exam/start")
def start_mock_exam(n: int = 30) -> dict:
    """Generate n SOA FM-weighted problems."""
    session_id, _ = dao.create_session()
    concepts = [c for c in dao.get_all_concepts() if c.generator is not None]

    # FM SOA category weights (approximate): interest/annuity/loan ~45%, bonds ~15%,
    # duration/immunization ~15%, derivatives ~10%, theory/other ~15%
    concepts_by_category: dict[str, list] = {
        "interest": [], "annuity": [], "loan": [],
        "bond": [], "duration": [], "derivatives": [],
    }
    for c in concepts:
        if c.category in concepts_by_category:
            concepts_by_category[c.category].append(c)

    n_interest = round(n * 0.20)
    n_annuity = round(n * 0.20)
    n_loan = round(n * 0.10)
    n_bond = round(n * 0.15)
    n_duration = round(n * 0.15)
    # Remainder goes to derivatives so total always equals n exactly.
    n_derivatives = n - n_interest - n_annuity - n_loan - n_bond - n_duration

    rng = np.random.default_rng(secrets.randbelow(2 ** 31))

    def _sample(pool: list, count: int) -> list:
        if not pool or count <= 0:
            return []
        # Weight by exam_weight_tier so higher-tier concepts appear more often.
        weights = np.array([float(c.exam_weight_tier) for c in pool], dtype=float)
        weights /= weights.sum()
        sampled_indices = rng.choice(len(pool), size=count, replace=True, p=weights)
        return [pool[i] for i in sampled_indices]

    selected = (
        _sample(concepts_by_category["interest"], n_interest)
        + _sample(concepts_by_category["annuity"], n_annuity)
        + _sample(concepts_by_category["loan"], n_loan)
        + _sample(concepts_by_category["bond"], n_bond)
        + _sample(concepts_by_category["duration"], n_duration)
        + _sample(concepts_by_category["derivatives"], n_derivatives)
    )
    # Shuffle so category clusters don't bias how students experience time pressure.
    rng.shuffle(selected)

    problems = []
    for concept in selected:
        generator_spec = concept.generator
        ask, _ = pick_ask(generator_spec["params"]["ask"])
        problem_seed = secrets.randbelow(2 ** 31)
        problem = generate(generator_spec["kind"], ask, generator_spec["params"], problem_seed)
        correct_answer_value = problem.correct_answer
        correct_answer_str = (
            f"{correct_answer_value:.6f}"
            if isinstance(correct_answer_value, (int, float))
            else str(correct_answer_value)
        )
        item_id = dao.log_shown(
            session_id=session_id,
            concept_id=concept.id,
            seed=problem_seed,
            problem_kind=f"{generator_spec['kind']}:{ask}",
            params_json=json.dumps(problem.params),
            correct_answer=correct_answer_str,
        )
        problems.append({
            "item_id": item_id,
            "concept_id": concept.id,
            "concept_name": concept.name,
            "category": concept.category,
            "statement": problem.statement,
            "choices": problem.choices,
        })

    return {"session_id": session_id, "problems": problems}


@app.post("/mock-exam/grade")
def grade_mock_exam(body: MockExamGradeIn) -> dict:
    by_category: dict[str, dict] = {}
    results = []

    for submission in body.answers:
        stored_correct_answer = dao.get_interaction_correct_answer(submission.item_id)
        concept_id = dao.get_interaction_concept(submission.item_id)
        is_correct = _grade_answer(submission.user_answer, stored_correct_answer or "")
        dao.log_answered(
            item_id=submission.item_id, user_answer=submission.user_answer or None,
            is_correct=is_correct, grade=3 if is_correct else 1, elapsed_ms=0,
        )
        concept = dao.get_concept(concept_id or "")
        category = concept.category if concept else "unknown"
        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}
        by_category[category]["total"] += 1
        if is_correct:
            by_category[category]["correct"] += 1
        results.append({
            "item_id": submission.item_id, "correct_answer": stored_correct_answer,
            "user_answer": submission.user_answer, "is_correct": is_correct,
            "concept_id": concept_id,
        })

    n_total = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    score = round(n_correct / n_total, 4) if n_total else 0.0
    dao.log_sample_exam(
        score=score, n_questions=n_total, passing_score=0.70,
        notes=f"Timed mock exam — {body.elapsed_s}s",
    )
    dao.close_session(body.session_id)
    return {
        "score": score, "n_correct": n_correct, "n_total": n_total,
        "elapsed_s": body.elapsed_s, "by_category": by_category, "results": results,
    }


def _grade_answer(user_answer: str, correct_answer: str, tolerance: float = 1e-3) -> bool:
    """Exact string match (MC) or float within tolerance (numeric).

    Tolerance 1e-3 matches SOA FM grading convention: answers rounded to 4 decimal places
    are accepted if within 0.001 of the stored value.
    """
    if user_answer.strip() == correct_answer.strip():
        return True
    try:
        return abs(float(user_answer) - float(correct_answer)) <= tolerance
    except ValueError:
        # Non-numeric MC answer that didn't match exactly → wrong.
        return False


def _db_reachable() -> bool:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
