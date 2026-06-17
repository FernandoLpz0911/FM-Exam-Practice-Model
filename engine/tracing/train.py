"""DKT training loop: masked BCE loss, gradient clipping, AUC evaluation, checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from engine.tracing.concept_index import load as load_index
from engine.tracing.dataset import DKTBatch, build_batches, load_sequences, train_val_split
from engine.tracing.model import DKT

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

_BEST_CHECKPOINT = CHECKPOINT_DIR / "dkt_best.pt"
_TRAIN_LOG = CHECKPOINT_DIR / "train_log.jsonl"


def _bce_loss(
    predictions: torch.Tensor,   # (B, T, M)
    batch: DKTBatch,
) -> torch.Tensor:
    """Masked BCE: only valid (non-padding) steps contribute, gathered at next concept.

    The model outputs P(correct) for every concept at every step, but the
    loss only cares about the concept that's actually asked next — gather
    picks that one prediction per step before comparing to the true outcome.
    """
    next_concept_idx = batch.t_indices.unsqueeze(-1)     # (B, T, 1) — for gather
    p_correct_next = predictions.gather(2, next_concept_idx).squeeze(-1)  # (B, T)

    return nn.functional.binary_cross_entropy(
        p_correct_next[batch.mask], batch.targets[batch.mask]
    )


def _eval_auc(
    model: DKT,
    batches: list[DKTBatch],
    device: torch.device,
) -> float:
    """Compute AUC over a set of batches."""
    model.eval()
    all_predictions: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in batches:
            inputs = batch.inputs.to(device)
            predictions = model(inputs)
            next_concept_idx = batch.t_indices.unsqueeze(-1).to(device)
            p_correct_next = predictions.gather(2, next_concept_idx).squeeze(-1).cpu()
            valid_step_mask = batch.mask

            all_predictions.extend(p_correct_next[valid_step_mask].tolist())
            all_targets.extend(batch.targets[valid_step_mask].tolist())

    # roc_auc_score requires both classes present; a validation split that's
    # all-correct or all-incorrect (common with small/early synthetic data)
    # can't produce a meaningful AUC, so this returns NaN rather than raising.
    if len(set(all_targets)) < 2:
        return float("nan")
    return float(roc_auc_score(all_targets, all_predictions))


def train(
    n_epochs: int = 50,
    hidden: int = 128,
    layers: int = 1,
    dropout: float = 0.2,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 42,
    device_str: str = "auto",
) -> dict:
    """Train DKT. Returns {val_auc, n_interactions, epochs_run, checkpoint_path}."""
    torch.manual_seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    concept_index = load_index()
    n_concepts = len(concept_index)

    sequences = load_sequences(concept_index)
    n_interactions = sum(len(session) for session in sequences)

    if len(sequences) < 2:
        # DKT needs at least one training session and one validation session
        # to produce any signal at all; bail out early with a NaN AUC rather
        # than letting build_batches/train_val_split run on near-empty data.
        return {
            "val_auc": float("nan"),
            "n_interactions": n_interactions,
            "epochs_run": 0,
            "checkpoint_path": None,
            "error": "Not enough sequences to train (need ≥ 2 sessions with ≥ 2 steps each).",
        }

    train_sequences, val_sequences = train_val_split(sequences, val_frac=val_frac, seed=seed)

    train_batches = build_batches(train_sequences, n_concepts, batch_size, shuffle=True)
    val_batches = build_batches(val_sequences, n_concepts, batch_size, shuffle=False)

    model = DKT(n_concepts, hidden=hidden, layers=layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    best_epoch = 0
    epoch_log_rows: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        valid_step_count = 0

        for batch in train_batches:
            inputs = batch.inputs.to(device)
            predictions = model(inputs)
            # DKTBatch's `inputs` field is unused by _bce_loss (only
            # t_indices/mask/targets matter there) — `predictions` is passed
            # as a placeholder just to satisfy the NamedTuple's positional shape.
            loss = _bce_loss(predictions, DKTBatch(
                predictions,
                batch.targets.to(device),
                batch.t_indices.to(device),
                batch.mask.to(device),
            ))
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping guards against the LSTM's exploding-gradient
            # tendency on longer sequences.
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss_sum += loss.item() * batch.mask.sum().item()
            valid_step_count += batch.mask.sum().item()

        avg_train_loss = epoch_loss_sum / max(valid_step_count, 1)
        val_auc = _eval_auc(model, val_batches, device)

        epoch_log_rows.append({"epoch": epoch, "train_loss": avg_train_loss, "val_auc": val_auc})

        # `val_auc != val_auc` is True only for NaN — guards the comparison
        # below since NaN > best_val_auc is always False but we want to
        # explicitly skip (not just fail) saving a checkpoint on a NaN epoch.
        if not (val_auc != val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "n_concepts": n_concepts,
                    "hidden": hidden,
                    "layers": layers,
                    "dropout": dropout,
                    "model_state": model.state_dict(),
                },
                _BEST_CHECKPOINT,
            )

    with _TRAIN_LOG.open("a") as log_file:
        for row in epoch_log_rows:
            log_file.write(json.dumps(row) + "\n")

    return {
        "val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "n_interactions": n_interactions,
        "epochs_run": n_epochs,
        "checkpoint_path": str(_BEST_CHECKPOINT) if best_val_auc > -1 else None,
    }
