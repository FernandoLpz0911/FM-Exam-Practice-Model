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
    ckpt = torch.load(_BEST_CHECKPOINT, map_location=device, weights_only=True)
    concept_index = load_index()
    model = DKT(
        n_concepts=ckpt["n_concepts"],
        hidden=ckpt["hidden"],
        layers=ckpt["layers"],
        dropout=ckpt.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, concept_index


def predict(
    history: list[tuple[str, bool]],
    device_str: str = "auto",
) -> dict[str, float] | None:
    """Given a sequence of (concept_id, is_correct) history steps,
    return {concept_id: P(correct)} for all concepts.

    Returns None if no checkpoint exists or history is empty.
    """
    if not history:
        return None

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    result = _load_model(device)
    if result is None:
        return None

    model, concept_index = result
    M = len(concept_index)

    x = torch.zeros(1, len(history), 2 * M)
    for t, (cid, correct) in enumerate(history):
        idx = concept_index.get(cid)
        if idx is None:
            continue
        hot = idx if correct else idx + M
        x[0, t, hot] = 1.0

    with torch.no_grad():
        preds = model(x.to(device))   # (1, T, M)

    # Use the last timestep's output
    last = preds[0, -1, :].cpu()     # (M,)

    idx_to_cid = {v: k for k, v in concept_index.items()}
    return {idx_to_cid[i]: float(last[i]) for i in range(M)}


def dkt_is_active() -> bool:
    """True when a trained checkpoint clears both the interaction-count and AUC gates."""
    from engine.config import DKT_MIN_AUC, DKT_MIN_INTERACTIONS
    from engine.db.dao import count_answered_interactions

    if count_answered_interactions() < DKT_MIN_INTERACTIONS:
        return False
    meta = checkpoint_meta()
    if meta is None:
        return False
    auc = meta.get("val_auc")
    return auc is not None and auc >= DKT_MIN_AUC


def checkpoint_meta() -> dict | None:
    """Return metadata from the best checkpoint, or None if none exists."""
    if not _BEST_CHECKPOINT.exists():
        return None
    ckpt = torch.load(_BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    return {
        "epoch": ckpt.get("epoch"),
        "val_auc": ckpt.get("val_auc"),
        "n_concepts": ckpt.get("n_concepts"),
    }
