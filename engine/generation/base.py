"""Problem generation framework: dataclass, registry, and shared MC helpers."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Problem:
    """A fully generated exam problem ready to serve to the frontend."""

    kind: str
    ask: str
    statement: str
    correct_answer: float
    choices: list[str] | None  # None for free-response numeric
    tolerance: float = 1e-4
    params: dict = field(default_factory=dict)
    seed: int = 0


_generators: dict[str, Callable] = {}


def register(kind: str) -> Callable:
    """Decorator that registers a generator function under the given kind key."""
    def decorator(fn: Callable) -> Callable:
        _generators[kind] = fn
        return fn
    return decorator


def generate(kind: str, ask: str, ranges: dict, seed: int) -> Problem:
    """Dispatch to the registered generator for kind. Raises if none found."""
    fn = _generators.get(kind)
    if fn is None:
        raise ValueError(f"No generator registered for kind '{kind}'")
    return fn(ask, ranges, seed)


def list_kinds() -> list[str]:
    """Return all registered generator kinds."""
    return list(_generators.keys())


def pick_ask(ask_list: list[str]) -> tuple[str, int]:
    """Pick an ask type at random. Returns (ask, ask_seed) — both must be logged."""
    ask_seed = secrets.randbelow(2**31)
    rng = np.random.default_rng(ask_seed)
    return str(rng.choice(ask_list)), ask_seed


def make_mc_choices(
    correct: float,
    wrongs: list[float],
    rng: np.random.Generator,
    decimals: int = 4,
) -> list[str]:
    """Combine correct answer with up to 3 distractor values into 4 shuffled choices.

    Deduplicates by formatted string. Falls back to random perturbations if
    fewer than 3 distinct wrong answers are provided.
    """
    def fmt(x: float) -> str:
        return f"{x:.{decimals}f}"

    target = fmt(correct)
    candidates: list[str] = []

    for w in wrongs:
        s = fmt(w)
        if s != target and s not in candidates:
            candidates.append(s)
        if len(candidates) == 3:
            break

    scale = max(abs(correct), 0.01)
    attempts = 0
    while len(candidates) < 3 and attempts < 200:
        perturbed = correct + rng.uniform(-0.6, 0.6) * scale
        s = fmt(perturbed)
        if s != target and s not in candidates:
            candidates.append(s)
        attempts += 1

    all_choices = [target] + candidates[:3]
    order = rng.permutation(4)
    return [all_choices[i] for i in order]


# FM submodule imports live in __init__.py to avoid circular imports.
