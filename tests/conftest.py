"""Shared pytest fixtures for ExamFMEngine tests.

Every test that touches SQLite gets an isolated temp DB via the
``isolated_db`` / ``seeded_db`` / ``api_client`` fixture chain.
The in-memory retry queue is cleared automatically before/after every test.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import engine.generation  # noqa: F401 — registers all generator kinds

MINIMAL_CONCEPTS = [
    {
        "id": "tvm",
        "name": "Time value of money",
        "category": "interest",
        "exam_weight_tier": 3,
        "summary": "PV / FV basics",
        "prerequisites": [],
        "generator": {
            "kind": "interest_tvm",
            "params": {
                "ask": ["future_value", "present_value"],
                "i_range": [0.05, 0.06],
                "n_range": [5, 6],
                "pv_range": [1000, 1001],
            },
        },
    },
    {
        "id": "annuity",
        "name": "Annuity immediate",
        "category": "annuity",
        "exam_weight_tier": 2,
        "summary": "PV / FV of level annuity",
        "prerequisites": ["tvm"],
        "generator": {
            "kind": "annuity_immediate",
            "params": {
                "ask": ["pv_annuity_imm"],
                "i_range": [0.05, 0.06],
                "n_range": [10, 11],
                "payment_range": [100, 101],
            },
        },
    },
    {
        "id": "theory_only",
        "name": "Theory card",
        "category": "interest",
        "exam_weight_tier": 1,
        "summary": "No generator — pure theory",
        "prerequisites": [],
        "generator": None,
    },
]


def _insert_concepts(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    for concept in MINIMAL_CONCEPTS:
        conn.execute(
            """INSERT OR REPLACE INTO concept
               (id, name, category, exam_weight_tier, summary, generator_json, theory_md)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                concept["id"],
                concept["name"],
                concept["category"],
                concept["exam_weight_tier"],
                concept.get("summary"),
                json.dumps(concept["generator"]) if concept.get("generator") else None,
                None,
            ),
        )
    for concept in MINIMAL_CONCEPTS:
        for prereq in concept.get("prerequisites", []):
            conn.execute(
                "INSERT OR IGNORE INTO concept_prereq (concept_id, prereq_id) VALUES (?, ?)",
                (concept["id"], prereq),
            )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def clear_retry_queue():
    """Clear the in-memory retry queue before and after every test."""
    from engine.scheduler import retry
    retry.clear()
    yield
    retry.clear()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Fresh SQLite DB in a temp directory; DB_PATH env var is patched."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from engine.db.connection import init_db
    init_db()
    yield db_file


@pytest.fixture
def seeded_db(isolated_db):
    """isolated_db + MINIMAL_CONCEPTS inserted."""
    _insert_concepts(isolated_db)
    yield isolated_db


@pytest.fixture
def api_client(seeded_db):
    """FastAPI TestClient backed by an isolated, seeded DB."""
    from fastapi.testclient import TestClient

    from engine.main import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
