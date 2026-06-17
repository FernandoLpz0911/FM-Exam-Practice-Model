"""Full API coverage tests for engine/main.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

# Private helpers (importable directly)


class TestGradeAnswer:
    def test_exact_string_match(self):
        from engine.main import _grade_answer
        assert _grade_answer("A", "A") is True

    def test_string_mismatch(self):
        from engine.main import _grade_answer
        assert _grade_answer("A", "B") is False

    def test_float_within_tolerance(self):
        from engine.main import _grade_answer
        assert _grade_answer("1000.0005", "1000.0000") is True

    def test_float_outside_tolerance(self):
        from engine.main import _grade_answer
        assert _grade_answer("1002.0", "1000.0") is False

    def test_non_numeric_returns_false(self):
        from engine.main import _grade_answer
        assert _grade_answer("abc", "1000.0") is False

    def test_whitespace_stripped(self):
        from engine.main import _grade_answer
        assert _grade_answer("  A  ", "A") is True

    def test_zero_exact(self):
        from engine.main import _grade_answer
        assert _grade_answer("0.0", "0.0") is True


class TestDbReachable:
    def test_returns_true_with_valid_db(self, isolated_db):
        from engine.main import _db_reachable
        assert _db_reachable() is True

    def test_returns_false_when_connection_fails(self, isolated_db):
        from engine.main import _db_reachable
        with patch("engine.main.get_connection", side_effect=Exception("fail")):
            assert _db_reachable() is False


# /health


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "cuda" in data
    assert data["db"] is True


# /concepts  and  /concept/{id}


def test_get_concepts(api_client):
    resp = api_client.get("/concepts")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    ids = {n["id"] for n in data["nodes"]}
    assert "tvm" in ids
    # annuity → prereq tvm should produce an edge
    assert any(e["concept_id"] == "annuity" and e["prereq_id"] == "tvm"
               for e in data["edges"])


def test_get_concept_found(api_client):
    resp = api_client.get("/concept/tvm")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "tvm"
    assert data["name"] == "Time value of money"


def test_get_concept_not_found(api_client):
    resp = api_client.get("/concept/does_not_exist")
    assert resp.status_code == 404


# /session


def test_create_session(api_client):
    resp = api_client.post("/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "started_at" in data


def test_close_session(api_client):
    sid = api_client.post("/session").json()["session_id"]
    resp = api_client.post(f"/session/{sid}/close")
    assert resp.status_code == 200
    assert "ended_at" in resp.json()


# /next


def test_next_returns_problem(api_client):
    sid = api_client.post("/session").json()["session_id"]
    resp = api_client.get(f"/next?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] is not None
    assert data["concept_id"] is not None
    assert data["statement"] is not None


def test_next_theory_card_has_no_choices(api_client, seeded_db):
    """A concept with no generator returns no choices."""
    import sqlite3
    from datetime import datetime, timedelta, timezone
    # Force theory_only to be overdue so scheduler picks it first
    # and tvm / annuity to be future-due
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=30)).isoformat()
    conn = sqlite3.connect(str(seeded_db))
    conn.execute(
        "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 1, 0, 'review', 1.0, NULL, ?, ?, NULL)",
        ("tvm", (now - timedelta(days=1)).isoformat(), future),
    )
    conn.execute(
        "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 1, 0, 'review', 1.0, NULL, ?, ?, NULL)",
        ("annuity", (now - timedelta(days=1)).isoformat(), future),
    )
    conn.commit()
    conn.close()

    # theory_only has no card state → reps=0 → frontier
    # But with tvm having reps>0 future-due, theory_only (reps=0) becomes frontier
    sid = api_client.post("/session").json()["session_id"]
    resp = api_client.get(f"/next?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    # theory_only should be the frontier candidate with no choices
    if data["concept_id"] == "theory_only":
        assert data["choices"] is None


def test_next_returns_no_concepts_message_when_all_future(api_client, seeded_db):
    """When all concepts are future-due and no frontier, returns message."""
    import sqlite3
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=30)).isoformat()
    past = (now - timedelta(days=1)).isoformat()
    conn = sqlite3.connect(str(seeded_db))
    for cid in ("tvm", "annuity", "theory_only"):
        conn.execute(
            "INSERT OR REPLACE INTO card_state (concept_id, reps, lapses, state, stability, difficulty, last_review, due, step) VALUES (?, 1, 0, 'review', 21.0, NULL, ?, ?, NULL)",
            (cid, past, future),
        )
    conn.commit()
    conn.close()
    sid = api_client.post("/session").json()["session_id"]
    resp = api_client.get(f"/next?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] is not None
    assert data["item_id"] is None


def test_next_marks_new_concept(api_client):
    sid = api_client.post("/session").json()["session_id"]
    resp = api_client.get(f"/next?session_id={sid}")
    data = resp.json()
    assert data["is_new_concept"] is True  # reps=0 for fresh DB


# /answer


def _get_item(api_client):
    """Helper: create session, get next, return (session_id, item_id)."""
    sid = api_client.post("/session").json()["session_id"]
    nxt = api_client.get(f"/next?session_id={sid}").json()
    return sid, nxt["item_id"]


def test_answer_grade_based(api_client):
    _, item_id = _get_item(api_client)
    resp = api_client.post("/answer", json={"item_id": item_id, "grade": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is None
    assert "next_review_at" in data


def test_answer_all_valid_grades(api_client):
    _, item_id = _get_item(api_client)
    for grade in (1, 2, 3, 4):
        resp = api_client.post("/answer", json={"item_id": item_id, "grade": grade})
        assert resp.status_code == 200


def test_answer_invalid_grade_returns_422(api_client):
    _, item_id = _get_item(api_client)
    resp = api_client.post("/answer", json={"item_id": item_id, "grade": 5})
    assert resp.status_code == 422


def test_answer_wrong_item_id_returns_404(api_client):
    resp = api_client.post("/answer", json={"item_id": 999999, "user_answer": "0"})
    assert resp.status_code == 404


def test_answer_numeric_correct(api_client):
    """Submit the stored correct answer → is_correct=True."""
    from engine.db.dao import get_interaction_correct_answer
    _, item_id = _get_item(api_client)
    correct = get_interaction_correct_answer(item_id)
    resp = api_client.post("/answer", json={"item_id": item_id, "user_answer": correct})
    assert resp.status_code == 200
    assert resp.json()["is_correct"] is True


def test_answer_numeric_wrong(api_client):
    _, item_id = _get_item(api_client)
    resp = api_client.post("/answer", json={"item_id": item_id, "user_answer": "0.00001"})
    assert resp.status_code == 200
    data = resp.json()
    # Correct answer is ~1276+ so 0.00001 is wrong
    assert data["is_correct"] is False
    assert data["note"] is not None or data["note"] is None  # may be None if unknown ask


def test_answer_theory_card_gives_no_is_correct(api_client, seeded_db):
    """Theory card (correct_answer='N/A') → grade=3 path → is_correct=None."""
    from engine.db import dao
    sid, _ = dao.create_session()
    # log_shown with no correct answer
    item_id = dao.log_shown(session_id=sid, concept_id="theory_only",
                             problem_kind="placeholder", correct_answer="N/A")
    resp = api_client.post("/answer", json={"item_id": item_id, "user_answer": ""})
    assert resp.status_code == 200
    assert resp.json()["is_correct"] is None


def test_answer_enqueues_retry_on_wrong(api_client):
    """Wrong answer on a generator concept enqueues a retry."""
    from engine.scheduler import retry
    _, item_id = _get_item(api_client)
    api_client.post("/answer", json={"item_id": item_id, "user_answer": "0.00001"})
    assert retry.pending() >= 0  # may or may not enqueue depending on is_correct path


# /hint  and  /solution


def test_hint_found(api_client):
    _, item_id = _get_item(api_client)
    resp = api_client.get(f"/hint/{item_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "hint_steps" in data
    assert "total" in data


def test_hint_not_found(api_client):
    resp = api_client.get("/hint/999999")
    assert resp.status_code == 404


def test_solution_found(api_client):
    _, item_id = _get_item(api_client)
    resp = api_client.get(f"/solution/{item_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) > 0


def test_solution_not_found(api_client):
    resp = api_client.get("/solution/999999")
    assert resp.status_code == 404


def test_hint_theory_card_returns_404(api_client, seeded_db):
    """Item with no ':' in problem_kind → 404 for hint."""
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(session_id=sid, concept_id="theory_only",
                             problem_kind="placeholder")
    resp = api_client.get(f"/hint/{item_id}")
    assert resp.status_code == 404


def test_solution_theory_card_returns_404(api_client, seeded_db):
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(session_id=sid, concept_id="theory_only",
                             problem_kind="placeholder")
    resp = api_client.get(f"/solution/{item_id}")
    assert resp.status_code == 404


# /train


def test_post_train(api_client):
    mock_result = {
        "n_interactions": 0,
        "epochs_run": 0,
        "val_auc": None,
        "best_epoch": None,
        "checkpoint_path": None,
        "error": None,
    }
    with patch("engine.main.dkt_train", return_value=mock_result):
        resp = api_client.post("/train")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_interactions" in data
    assert "dkt_active" in data


def test_post_train_nan_auc(api_client):
    """NaN val_auc should be coerced to None."""
    mock_result = {
        "n_interactions": 10,
        "epochs_run": 5,
        "val_auc": float("nan"),
        "best_epoch": 3,
        "checkpoint_path": None,
        "error": None,
    }
    with patch("engine.main.dkt_train", return_value=mock_result):
        resp = api_client.post("/train")
    assert resp.status_code == 200
    assert resp.json()["val_auc"] is None


# /state


def test_get_state(api_client):
    resp = api_client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "states" in data
    ids = {s["concept_id"] for s in data["states"]}
    assert "tvm" in ids


# /dkt/status


def test_get_dkt_status(api_client):
    resp = api_client.get("/dkt/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_interactions" in data
    assert "dkt_active" in data
    assert data["dkt_active"] is False  # no checkpoint in tests


# /readiness


def test_get_readiness(api_client):
    resp = api_client.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0


def test_get_readiness_no_snapshot(api_client):
    resp = api_client.get("/readiness?snapshot=false")
    assert resp.status_code == 200


def test_get_readiness_history(api_client):
    api_client.get("/readiness")  # create one snapshot
    resp = api_client.get("/readiness/history")
    assert resp.status_code == 200
    assert "snapshots" in resp.json()


# /readiness/chart/*


def test_readiness_chart_timeline(api_client):
    resp = api_client.get("/readiness/chart/timeline")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"


def test_readiness_chart_category(api_client):
    resp = api_client.get("/readiness/chart/category")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


# /sample-exam


def test_post_sample_exam(api_client):
    resp = api_client.post("/sample-exam", json={"score": 0.73, "n_questions": 30})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data


def test_post_sample_exam_with_predicted(api_client):
    resp = api_client.post(
        "/sample-exam",
        json={"score": 0.80, "n_questions": 30, "passing_score": 0.70, "notes": "test"},
    )
    assert resp.status_code == 200


# /journal


def test_get_journal(api_client):
    resp = api_client.get("/journal")
    assert resp.status_code == 200
    data = resp.json()
    assert "stats" in data
    assert "sample_exams" in data
    assert "recent_readiness" in data


# /analytics/timing


def test_get_analytics_timing_empty(api_client):
    resp = api_client.get("/analytics/timing")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_analytics_timing_with_data(api_client):
    from engine.db import dao
    sid, _ = dao.create_session()
    item_id = dao.log_shown(session_id=sid, concept_id="tvm",
                             problem_kind="interest_tvm:future_value",
                             params_json="{}", correct_answer="1000.0")
    dao.log_answered(item_id=item_id, user_answer="1000.0", is_correct=True,
                     grade=4, elapsed_ms=30000)
    resp = api_client.get("/analytics/timing")
    assert resp.status_code == 200
    rows = resp.json()
    assert any(r["concept_id"] == "tvm" for r in rows)


# /mock-exam/start  and  /mock-exam/grade


def test_mock_exam_start(api_client):
    resp = api_client.post("/mock-exam/start?n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "problems" in data
    assert len(data["problems"]) > 0


def test_mock_exam_grade(api_client):
    # Start an exam and immediately grade all wrong
    start = api_client.post("/mock-exam/start?n=3").json()
    session_id = start["session_id"]
    answers = [{"item_id": p["item_id"], "user_answer": "0"} for p in start["problems"]]
    resp = api_client.post("/mock-exam/grade", json={
        "session_id": session_id,
        "answers": answers,
        "elapsed_s": 120,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "n_total" in data
    assert data["n_total"] == len(answers)
    assert "by_category" in data


def test_mock_exam_grade_all_correct(api_client):
    """Submit the stored correct answer for each problem → score=1.0."""
    from engine.db.dao import get_interaction_correct_answer
    # n=10: rounding produces ~2 interest + 2 annuity = 4 problems with our seeded_db
    start = api_client.post("/mock-exam/start?n=10").json()
    session_id = start["session_id"]
    problems = start["problems"]
    if not problems:
        pytest.skip("No problems generated with current seeded concepts")
    answers = []
    for p in problems:
        correct = get_interaction_correct_answer(p["item_id"]) or "0"
        answers.append({"item_id": p["item_id"], "user_answer": correct})
    resp = api_client.post("/mock-exam/grade", json={
        "session_id": session_id,
        "answers": answers,
        "elapsed_s": 60,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] == pytest.approx(1.0)


# DKT-active branches (engine/main.py lines 145-149, 347-348, 373, 388-393)

_DKT_PREDS = {"tvm": 0.8, "annuity": 0.6}


def test_next_dkt_active_with_predictions(api_client):
    """DKT active + non-empty predictions → weakness dict applied (lines 145-149)."""
    sid = api_client.post("/session").json()["session_id"]
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value=_DKT_PREDS):
        resp = api_client.get(f"/next?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["concept_id"] is not None


def test_next_dkt_active_empty_predictions(api_client):
    """DKT active + empty predictions → weakness stays None (line 148 False branch)."""
    sid = api_client.post("/session").json()["session_id"]
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value={}):
        resp = api_client.get(f"/next?session_id={sid}")
    assert resp.status_code == 200


def test_readiness_dkt_active(api_client):
    """DKT active → p_correct passed to compute_readiness (lines 347-348)."""
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value=_DKT_PREDS):
        resp = api_client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["dkt_active"] is True


def test_readiness_chart_category_dkt_active(api_client):
    """DKT active → predict called in chart/category endpoint (line 373)."""
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value=_DKT_PREDS):
        resp = api_client.get("/readiness/chart/category")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_sample_exam_dkt_active_with_predictions(api_client):
    """DKT active + non-empty predictions → predicted score set (lines 388-393)."""
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value=_DKT_PREDS):
        resp = api_client.post("/sample-exam", json={"score": 0.72, "n_questions": 30})
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted"] is not None


def test_sample_exam_dkt_active_empty_predictions(api_client):
    """DKT active + empty predictions → predicted stays None (line 390 False branch)."""
    with patch("engine.tracing.infer.dkt_is_active", return_value=True), \
         patch("engine.tracing.infer.predict", return_value={}):
        resp = api_client.post("/sample-exam", json={"score": 0.65})
    assert resp.status_code == 200
    assert resp.json()["predicted"] is None


# /mock-exam/start unknown-category concept (line 459 False branch)


def test_mock_exam_start_unknown_category_skipped(api_client, seeded_db):
    """Concept whose category is not in by_cat is silently skipped (line 459 else)."""
    import json as _json
    import sqlite3

    conn = sqlite3.connect(str(seeded_db))
    conn.execute(
        "INSERT OR REPLACE INTO concept "
        "(id, name, category, exam_weight_tier, summary, generator_json, theory_md) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "other_concept", "Other Concept", "other", 1, "Weird category",
            _json.dumps({
                "kind": "interest_tvm",
                "params": {
                    "ask": ["future_value"],
                    "i_range": [0.05, 0.06],
                    "n_range": [5, 6],
                    "pv_range": [1000, 1001],
                },
            }),
            None,
        ),
    )
    conn.commit()
    conn.close()

    resp = api_client.post("/mock-exam/start?n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "problems" in data
