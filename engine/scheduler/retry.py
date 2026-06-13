"""In-memory retry queue for error-correction loop (P4.7).

When a student answers wrong, the (concept_id, kind, ask) is enqueued here.
policy.next_concept() drains this queue before normal FSRS scheduling, so the
same problem type reappears within the next 1–3 items.
"""
from __future__ import annotations

from collections import deque

_queue: deque[dict] = deque()


def enqueue(concept_id: str, kind: str, ask: str) -> None:
    _queue.append({"concept_id": concept_id, "kind": kind, "ask": ask})


def dequeue() -> dict | None:
    return _queue.popleft() if _queue else None


def clear() -> None:
    _queue.clear()


def pending() -> int:
    return len(_queue)
