"""P7.6 — Ablation: FSRS-only vs FSRS+DKT next-concept selection.

Generates a self-contained simulation:
1. Synthetic warmup interactions → temp DB → train DKT → load weights.
2. Two independent simulated students, same skill model, different policies:
   - FSRS-only: select by exam_weight_tier / (reps + 1)  (urgency proxy)
   - DKT-hybrid: select by (1 − P_dkt) × exam_weight_tier
3. Track readiness over steps for both; return data for plotting.

Everything runs in isolated temp paths — no DB or checkpoint pollution.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import torch

from engine.analytics.readiness import compute_readiness
from engine.db.dao import Concept


def _fresh_skill(concept_ids: list[str], base_accuracy: float) -> dict[str, float]:
    return {concept_id: base_accuracy for concept_id in concept_ids}


def _fsrs_score(concept: Concept, reps: dict[str, int]) -> float:
    """Urgency proxy: high-tier and low-rep concepts score highest."""
    return concept.exam_weight_tier / (reps[concept.id] + 1)


def _dkt_score(
    concept: Concept,
    p_correct: dict[str, float],
    reps: dict[str, int],
) -> float:
    """DKT weakness weighted by tier; fall back to a neutral 0.5 prior if no pred."""
    p_correct_value = p_correct.get(concept.id, 0.5)
    return (1.0 - p_correct_value) * concept.exam_weight_tier


def _dkt_predict_batch(
    model: torch.nn.Module,
    history: list[tuple[int, int]],  # (concept_idx, is_correct)
    n_concepts: int,
    device: torch.device,
) -> dict[int, float]:
    """Run one forward pass on history; return {concept_idx: P(correct)}."""
    n_steps = len(history)
    one_hot_input = torch.zeros(1, n_steps, 2 * n_concepts)
    for t, (concept_idx, is_correct) in enumerate(history):
        # Two-hot encoding scheme: index = concept if correct, or
        # concept + n_concepts if incorrect — matches the DKT model's
        # training encoding (see engine/tracing/dataset.py:encode_sequence).
        hot_index = concept_idx if is_correct else concept_idx + n_concepts
        one_hot_input[0, t, hot_index] = 1.0
    with torch.no_grad():
        predictions = model(one_hot_input.to(device))   # (1, T, M)
    last_step_predictions = predictions[0, -1, :].cpu()
    return {i: float(last_step_predictions[i]) for i in range(n_concepts)}


def _simulate(
    policy: str,                      # "fsrs" | "dkt"
    concepts: list[Concept],
    model: torch.nn.Module | None,    # DKT model (loaded weights); None → FSRS-only
    concept_index: dict[str, int],
    n_steps: int,
    rng: random.Random,
    base_accuracy: float = 0.40,
    improvement_rate: float = 0.015,
    device: torch.device | None = None,
) -> list[dict]:
    """Run one simulation; return [{step, readiness, concept_id}]."""
    if device is None:
        device = torch.device("cpu")

    index_to_concept_id = {idx: cid for cid, idx in concept_index.items()}
    n_concepts = len(concept_index)
    concept_ids = [concept.id for concept in concepts]

    skill = _fresh_skill(concept_ids, base_accuracy)
    reps: dict[str, int] = {concept_id: 0 for concept_id in concept_ids}
    answer_history: list[tuple[int, int]] = []   # (concept_idx, is_correct)
    log: list[dict] = []

    # Cache of DKT P(correct) predictions, refreshed periodically below —
    # stale between refreshes, but re-running the LSTM every single step
    # would make a 300-step simulation prohibitively slow.
    p_correct_cache: dict[str, float] = {}

    for step in range(n_steps):
        if policy == "fsrs" or model is None:
            chosen_concept = max(concepts, key=lambda c: _fsrs_score(c, reps))
        else:
            chosen_concept = max(
                concepts, key=lambda c: _dkt_score(c, p_correct_cache, reps)
            )

        concept_id = chosen_concept.id
        concept_idx = concept_index.get(concept_id, 0)

        # Simulate the student's response using their true (oracle) skill —
        # the DKT model never sees this value, only the resulting correct/
        # incorrect outcome, just like in a real practice session.
        p_correct_true = min(0.95, skill[concept_id])
        is_correct = int(rng.random() < p_correct_true)

        answer_history.append((concept_idx, is_correct))
        reps[concept_id] += 1
        skill[concept_id] = min(0.95, skill[concept_id] + improvement_rate)

        # Refresh the DKT prediction cache every 5 steps (and on the very
        # first step) — a speed/freshness tradeoff, since recomputing after
        # every single answer is unnecessary for a slowly-drifting estimate.
        if policy == "dkt" and model is not None and (step % 5 == 0 or step == 0):
            predictions_by_idx = _dkt_predict_batch(
                model, answer_history[-200:], n_concepts, device
            )
            p_correct_cache = {
                index_to_concept_id[idx]: value for idx, value in predictions_by_idx.items()
            }

        # Compute readiness using the student's current true skill as the
        # P(correct) oracle (not the DKT prediction) — this isolates how good
        # the *selection policy* is, independent of DKT prediction accuracy.
        readiness, _ = compute_readiness(concepts, skill)
        log.append({"step": step + 1, "readiness": readiness, "concept_id": concept_id})

    return log


def run_ablation(
    n_warmup_sessions: int = 15,
    n_warmup_steps: int = 25,
    n_trial_steps: int = 300,
    hidden: int = 64,
    n_epochs: int = 30,
    base_accuracy: float = 0.40,
    improvement_rate: float = 0.015,
    seed: int = 42,
) -> dict:
    """Run the full ablation. Returns {fsrs_only, dkt_hybrid, n_warmup_interactions}."""
    from engine.db.connection import init_db
    from engine.db.dao import get_all_concepts
    from engine.tracing.concept_index import load as load_index
    from engine.tracing.model import DKT
    from engine.tracing.train import train as dkt_train

    concept_index = load_index()
    concepts: list[Concept] = []  # populated once the temp DB is seeded, below

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: this constructed Random instance is never assigned or used —
    # looks like leftover/dead code from an earlier version of this function.
    # The actual simulation RNGs are created separately below (seed + 1).
    random.Random(seed)

    with tempfile.TemporaryDirectory() as tmp_dir:
        warmup_db_path = str(Path(tmp_dir) / "warmup.db")
        checkpoint_dir = Path(tmp_dir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Point DB_PATH at an isolated temp DB so this ablation run never
        # touches the real database or its DKT checkpoints.
        previous_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = warmup_db_path
        try:
            init_db()
            from engine.db.seed import load as seed_load
            seed_load(Path("data/concept_graph.seed.json"))
            concepts = get_all_concepts()

            from engine.tracing.synthetic import seed_synthetic
            n_warmup_interactions = seed_synthetic(
                n_sessions=n_warmup_sessions,
                steps_per_session=n_warmup_steps,
                rng_seed=seed,
            )

            # Train a throwaway DKT model on the synthetic warmup data, redirecting
            # the training module's checkpoint paths so this never overwrites the
            # real trained model on disk.
            import engine.tracing.train as train_mod
            previous_checkpoint_dir = train_mod.CHECKPOINT_DIR
            previous_best_checkpoint = train_mod._BEST_CHECKPOINT
            previous_train_log = train_mod._TRAIN_LOG
            train_mod.CHECKPOINT_DIR = checkpoint_dir
            train_mod._BEST_CHECKPOINT = checkpoint_dir / "dkt_best.pt"
            train_mod._TRAIN_LOG = checkpoint_dir / "log.jsonl"
            try:
                training_result = dkt_train(n_epochs=n_epochs, hidden=hidden, seed=seed)
                checkpoint_path = train_mod._BEST_CHECKPOINT
                trained_val_auc = training_result.get("val_auc", float("nan"))

                # If training produced no checkpoint (e.g. AUC was NaN every
                # epoch because warmup data lacked variety), fall back to the
                # FSRS-only policy by leaving model as None.
                model: torch.nn.Module | None = None
                if checkpoint_path.exists():
                    checkpoint = torch.load(
                        checkpoint_path, map_location=device, weights_only=True
                    )
                    n_concepts = len(concept_index)
                    dkt_model = DKT(
                        n_concepts, hidden=checkpoint["hidden"], layers=checkpoint["layers"]
                    ).to(device)
                    dkt_model.load_state_dict(checkpoint["model_state"])
                    dkt_model.eval()
                    model = dkt_model
            finally:
                train_mod.CHECKPOINT_DIR = previous_checkpoint_dir
                train_mod._BEST_CHECKPOINT = previous_best_checkpoint
                train_mod._TRAIN_LOG = previous_train_log

        finally:
            if previous_db_path is None:
                os.environ.pop("DB_PATH", None)
            else:
                os.environ["DB_PATH"] = previous_db_path

    # Run both policies on independent simulated students (pure Python, no DB).
    fsrs_only_log = _simulate(
        "fsrs", concepts, None, concept_index,
        n_trial_steps, random.Random(seed + 1),
        base_accuracy, improvement_rate, device,
    )
    dkt_hybrid_log = _simulate(
        "dkt", concepts, model, concept_index,
        n_trial_steps, random.Random(seed + 1),   # same RNG seed → same student luck
        base_accuracy, improvement_rate, device,
    )

    return {
        "fsrs_only": fsrs_only_log,
        "dkt_hybrid": dkt_hybrid_log,
        "n_warmup_interactions": n_warmup_interactions,
        "warmup_val_auc": trained_val_auc,
    }
