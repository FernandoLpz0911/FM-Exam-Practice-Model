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
    correct_answer: float,
    distractor_values: list[float],
    rng: np.random.Generator,
    decimals: int = 4,
) -> list[str]:
    """Combine the correct answer with up to 3 distractor values into 4 shuffled choices.

    Deduplicates by formatted string (two distractors that round to the same
    displayed value would otherwise produce a multiple-choice question with a
    duplicate option). Falls back to random perturbations of the correct answer
    if the caller supplied fewer than 3 distinct wrong answers.

    NOTE: if fewer than 3 unique distractors can be produced even after 200
    perturbation attempts (e.g. correct_answer == 0 and decimals is too coarse
    to separate small perturbations), `candidates` stays short and the final
    `all_choices[i]` lookup below will raise IndexError, since `rng.permutation(4)`
    always yields indices 0-3 regardless of how many choices actually exist.
    """
    def format_choice(value: float) -> str:
        return f"{value:.{decimals}f}"

    correct_choice = format_choice(correct_answer)
    distractor_choices: list[str] = []

    for distractor in distractor_values:
        formatted = format_choice(distractor)
        if formatted != correct_choice and formatted not in distractor_choices:
            distractor_choices.append(formatted)
        if len(distractor_choices) == 3:
            break

    # Caller-supplied distractors weren't enough (or weren't distinct after
    # rounding) — perturb the correct answer by a magnitude proportional to its
    # own size so the synthetic wrong answers stay plausible at any scale.
    perturbation_scale = max(abs(correct_answer), 0.01)
    attempts = 0
    while len(distractor_choices) < 3 and attempts < 200:
        perturbed_value = correct_answer + rng.uniform(-0.6, 0.6) * perturbation_scale
        formatted = format_choice(perturbed_value)
        if formatted != correct_choice and formatted not in distractor_choices:
            distractor_choices.append(formatted)
        attempts += 1

    all_choices = [correct_choice] + distractor_choices[:3]
    shuffled_order = rng.permutation(4)
    return [all_choices[i] for i in shuffled_order]


# FM submodule imports live in __init__.py to avoid circular imports.
