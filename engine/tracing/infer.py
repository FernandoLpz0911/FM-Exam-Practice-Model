"""
DKT inference: load the best checkpoint and predict P(correct) per concept.

Main entry points
-----------------
predict(history)     Given a (concept_id, is_correct) sequence, returns a dict
                     mapping each concept_id to its estimated P(correct).
dkt_is_active()      Returns True only when both the interaction-count gate and
                     the AUC gate are cleared (see engine/config.py).
checkpoint_meta()    Reads epoch/val_auc/n_concepts from the saved checkpoint.
"""
from __future__ import annotations

import torch

from engine.tracing.concept_index import load as load_index
from engine.tracing.model import DKT
from engine.tracing.train import _BEST_CHECKPOINT


def _load_model(device: torch.device) -> tuple[DKT, dict[str, int]] | None:
    if not _BEST_CHECKPOINT.exists():
        return None
    checkpoint = torch.load(_BEST_CHECKPOINT, map_location=device, weights_only=True)
    concept_index = load_index()
    model = DKT(
        n_concepts=checkpoint["n_concepts"],
        hidden=checkpoint["hidden"],
        layers=checkpoint["layers"],
        dropout=checkpoint.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, concept_index


def predict(
    history: list[tuple[str, bool]],
    device_str: str = "auto",
) -> dict[str, float] | None:
    """Given a sequence of (concept_id, is_correct) history steps,
    return {concept_id: P(correct)} for all concepts.

    Returns None if no checkpoint exists or history is empty.

    NOTE: history entries with a concept_id not in the trained index are
    silently skipped (left as zero rows) rather than raising — if *every*
    entry is unrecognized, this still returns a full set of predictions from
    an all-zero input rather than None, so callers shouldn't assume a
    non-None result means the history was actually used.
    """
    if not history:
        return None

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    loaded = _load_model(device)
    if loaded is None:
        return None

    model, concept_index = loaded
    n_concepts = len(concept_index)

    one_hot_input = torch.zeros(1, len(history), 2 * n_concepts)
    for t, (concept_id, is_correct) in enumerate(history):
        concept_idx = concept_index.get(concept_id)
        if concept_idx is None:
            continue
        hot_index = concept_idx if is_correct else concept_idx + n_concepts
        one_hot_input[0, t, hot_index] = 1.0

    with torch.no_grad():
        predictions = model(one_hot_input.to(device))   # (1, T, M)

    # Only the final timestep's prediction reflects the full history — this
    # is "P(correct) on each concept if asked right now."
    last_step_predictions = predictions[0, -1, :].cpu()     # (M,)

    index_to_concept_id = {idx: cid for cid, idx in concept_index.items()}
    return {
        index_to_concept_id[i]: float(last_step_predictions[i]) for i in range(n_concepts)
    }


def dkt_is_active() -> bool:
    """True when a trained checkpoint clears both the interaction-count and AUC gates.

    Both gates exist to avoid trusting an undertrained model: too few real
    interactions means the synthetic-only or sparse model hasn't seen enough
    real behavior, and a low AUC means it isn't actually predictive yet.
    """
    from engine.config import DKT_MIN_AUC, DKT_MIN_INTERACTIONS
    from engine.db.dao import count_answered_interactions

    if count_answered_interactions() < DKT_MIN_INTERACTIONS:
        return False
    meta = checkpoint_meta()
    if meta is None:
        return False
    val_auc = meta.get("val_auc")
    return val_auc is not None and val_auc >= DKT_MIN_AUC


def checkpoint_meta() -> dict | None:
    """Return metadata from the best checkpoint, or None if none exists."""
    if not _BEST_CHECKPOINT.exists():
        return None
    checkpoint = torch.load(_BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    return {
        "epoch": checkpoint.get("epoch"),
        "val_auc": checkpoint.get("val_auc"),
        "n_concepts": checkpoint.get("n_concepts"),
    }
