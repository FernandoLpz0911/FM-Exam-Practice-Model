"""P5.1 — SQLite interactions → padded DKT sequences.

Each session becomes one sequence of (concept_index, is_correct) steps.
Encoding: step t → 2M one-hot, where index = concept_idx if correct else
concept_idx + M.  Target at step t = is_correct of step t+1, gathered at
concept_index of step t+1.  Padded steps are masked out of the loss.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence

from engine.db.connection import get_connection
from engine.tracing.concept_index import load as load_index


class DKTBatch(NamedTuple):
    inputs: torch.Tensor    # (B, T, 2M)  — one-hot encoded history
    targets: torch.Tensor   # (B, T)      — 0/1 next-step outcome
    t_indices: torch.Tensor # (B, T)      — concept index of next step (for gather)
    mask: torch.Tensor      # (B, T) bool — True where step is valid


def load_sequences(
    concept_index: dict[str, int] | None = None,
) -> list[list[tuple[int, int]]]:
    """Return per-session lists of (concept_idx, is_correct) for answered interactions."""
    if concept_index is None:
        concept_index = load_index()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, concept_id, is_correct
            FROM interaction
            WHERE is_correct IS NOT NULL
            ORDER BY session_id, id
            """
        ).fetchall()

    sessions: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        concept_idx = concept_index.get(row["concept_id"])
        if concept_idx is None:
            # Interaction references a concept that isn't in the trained
            # index (e.g. added after the index was built) — skip rather
            # than crash, since DKT can't represent unknown concepts anyway.
            continue
        sessions.setdefault(row["session_id"], []).append(
            (concept_idx, int(row["is_correct"]))
        )

    # Only sessions with at least 2 steps (need one input step + one target)
    return [session for session in sessions.values() if len(session) >= 2]


def encode_sequence(
    session: list[tuple[int, int]],
    n_concepts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one session sequence into tensors (before padding).

    Returns:
        inputs       — (T-1, 2M) float one-hot input
        targets      — (T-1,)    float 0/1 next-step outcome
        next_concept — (T-1,)    long  next-step concept index (for loss gather)
    """
    n_steps = len(session)
    inputs = torch.zeros(n_steps - 1, 2 * n_concepts)
    targets = torch.zeros(n_steps - 1)
    next_concept = torch.zeros(n_steps - 1, dtype=torch.long)

    for t in range(n_steps - 1):
        concept_idx, is_correct = session[t]
        # Two-hot encoding: same concept index space, but offset by
        # n_concepts when the answer was wrong, so the model can distinguish
        # "saw concept X and got it right" from "...and got it wrong".
        hot_index = concept_idx if is_correct else concept_idx + n_concepts
        inputs[t, hot_index] = 1.0
        next_concept_idx, next_is_correct = session[t + 1]
        targets[t] = float(next_is_correct)
        next_concept[t] = next_concept_idx

    return inputs, targets, next_concept


def build_batches(
    sequences: list[list[tuple[int, int]]],
    n_concepts: int,
    batch_size: int = 32,
    shuffle: bool = True,
) -> list[DKTBatch]:
    """Encode sequences and group into padded batches."""
    encoded_sequences = [encode_sequence(session, n_concepts) for session in sequences]

    if shuffle:
        import random
        random.shuffle(encoded_sequences)

    batches: list[DKTBatch] = []
    for batch_start in range(0, len(encoded_sequences), batch_size):
        chunk = encoded_sequences[batch_start:batch_start + batch_size]
        chunk_inputs, chunk_targets, chunk_next_concepts = zip(*chunk)

        # Sessions in a chunk have different lengths; record the true length
        # of each so padded steps can be masked out of the loss later.
        session_lengths = torch.tensor([inputs.size(0) for inputs in chunk_inputs])

        padded_inputs = pad_sequence(chunk_inputs, batch_first=True)
        padded_targets = pad_sequence(chunk_targets, batch_first=True, padding_value=0.0)
        padded_next_concepts = pad_sequence(
            chunk_next_concepts, batch_first=True, padding_value=0
        )

        max_steps = padded_inputs.size(1)
        valid_step_mask = torch.arange(max_steps).unsqueeze(0) < session_lengths.unsqueeze(1)

        batches.append(
            DKTBatch(padded_inputs, padded_targets, padded_next_concepts, valid_step_mask)
        )

    return batches


def train_val_split(
    sequences: list[list[tuple[int, int]]],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Split by session (not by interaction) to prevent leakage.

    NOTE: n_val is floored at 1, so with very few sequences (e.g. exactly 1
    session total) the training split can end up empty — build_batches([])
    then yields no batches, and the training loop simply trains on zero
    batches per epoch without raising. Callers needing a real fit should
    ensure there are at least a handful of sessions.
    """
    import random
    rng = random.Random(seed)
    shuffled_sequences = list(sequences)
    rng.shuffle(shuffled_sequences)
    n_val = max(1, int(len(shuffled_sequences) * val_frac))
    return shuffled_sequences[n_val:], shuffled_sequences[:n_val]
