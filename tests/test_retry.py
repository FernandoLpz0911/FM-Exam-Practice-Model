"""Tests for engine/scheduler/retry.py — in-memory retry queue."""
from __future__ import annotations

from engine.scheduler import retry


def test_dequeue_empty_returns_none():
    assert retry.dequeue() is None


def test_enqueue_dequeue_round_trip():
    retry.enqueue("tvm", "interest_tvm", "future_value")
    entry = retry.dequeue()
    assert entry == {"concept_id": "tvm", "kind": "interest_tvm", "ask": "future_value"}


def test_dequeue_fifo():
    retry.enqueue("a", "k1", "ask1")
    retry.enqueue("b", "k2", "ask2")
    e1 = retry.dequeue()
    e2 = retry.dequeue()
    assert e1["concept_id"] == "a"
    assert e2["concept_id"] == "b"


def test_pending_count():
    assert retry.pending() == 0
    retry.enqueue("x", "k", "a")
    assert retry.pending() == 1
    retry.enqueue("y", "k", "a")
    assert retry.pending() == 2
    retry.dequeue()
    assert retry.pending() == 1


def test_clear_empties_queue():
    retry.enqueue("a", "k", "a")
    retry.enqueue("b", "k", "a")
    retry.clear()
    assert retry.pending() == 0
    assert retry.dequeue() is None


def test_dequeue_returns_none_after_all_consumed():
    retry.enqueue("a", "k", "a")
    retry.dequeue()
    assert retry.dequeue() is None
