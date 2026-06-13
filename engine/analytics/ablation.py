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


def _fresh_skill(concept_ids: list[str], base: float) -> dict[str, float]:
    return {cid: base for cid in concept_ids}


def _fsrs_score(concept: Concept, reps: dict[str, int]) -> float:
    """Urgency proxy: high-tier and low-rep concepts score highest."""
    return concept.exam_weight_tier / (reps[concept.id] + 1)


def _dkt_score(
    concept: Concept,
    p_correct: dict[str, float],
    reps: dict[str, int],
) -> float:
    """DKT weakness weighted by tier; fall back to urgency proxy if no pred."""
    p = p_correct.get(concept.id, 0.5)
    return (1.0 - p) * concept.exam_weight_tier


def _dkt_predict_batch(
    model: torch.nn.Module,
    history: list[tuple[int, int]],  # (concept_idx, is_correct)
    n_concepts: int,
    device: torch.device,
) -> dict[int, float]:
    """Run one forward pass on history; return {concept_idx: P(correct)}."""
    T = len(history)
    x = torch.zeros(1, T, 2 * n_concepts)
    for t, (c_idx, correct) in enumerate(history):
        hot = c_idx if correct else c_idx + n_concepts
        x[0, t, hot] = 1.0
    with torch.no_grad():
        preds = model(x.to(device))   # (1, T, M)
    last = preds[0, -1, :].cpu()
    return {i: float(last[i]) for i in range(n_concepts)}


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

    idx_to_cid = {v: k for k, v in concept_index.items()}
    M = len(concept_index)
    concept_ids = [c.id for c in concepts]

    skill = _fresh_skill(concept_ids, base_accuracy)
    reps: dict[str, int] = {cid: 0 for cid in concept_ids}
    history_idx: list[tuple[int, int]] = []   # (concept_idx, is_correct)
    log: list[dict] = []

    p_correct_cache: dict[str, float] = {}   # recomputed every step for DKT

    for step in range(n_steps):
        # Select concept
        if policy == "fsrs" or model is None:
            concept = max(concepts, key=lambda c: _fsrs_score(c, reps))
        else:
            concept = max(concepts, key=lambda c: _dkt_score(c, p_correct_cache, reps))

        cid = concept.id
        c_idx = concept_index.get(cid, 0)

        # Simulate student response
        p = min(0.95, skill[cid])
        is_correct = int(rng.random() < p)

        # Update
        history_idx.append((c_idx, is_correct))
        reps[cid] += 1
        skill[cid] = min(0.95, skill[cid] + improvement_rate)

        # DKT prediction update every 5 steps (speed vs freshness tradeoff)
        if policy == "dkt" and model is not None and (step % 5 == 0 or step == 0):
            idx_preds = _dkt_predict_batch(model, history_idx[-200:], M, device)
            p_correct_cache = {idx_to_cid[i]: v for i, v in idx_preds.items()}

        # Compute readiness using current skill as P(correct) oracle
        readiness, _ = compute_readiness(concepts, skill)
        log.append({"step": step + 1, "readiness": readiness, "concept_id": cid})

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
    concepts: list[Concept] = []  # filled below inside env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.Random(seed)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "warmup.db")
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir()

        # --- warmup: seed synthetic data into temp DB ---
        old_db = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = db_path
        try:
            init_db()
            from engine.db.seed import load as seed_load
            seed_load(Path("data/concept_graph.seed.json"))
            concepts = get_all_concepts()

            from engine.tracing.synthetic import seed_synthetic
            n_warmup = seed_synthetic(
                n_sessions=n_warmup_sessions,
                steps_per_session=n_warmup_steps,
                rng_seed=seed,
            )

            # --- train DKT on warmup data ---
            import engine.tracing.train as train_mod
            old_ckpt_dir = train_mod.CHECKPOINT_DIR
            old_best = train_mod._BEST_CHECKPOINT
            old_log = train_mod._TRAIN_LOG
            train_mod.CHECKPOINT_DIR = ckpt_dir
            train_mod._BEST_CHECKPOINT = ckpt_dir / "dkt_best.pt"
            train_mod._TRAIN_LOG = ckpt_dir / "log.jsonl"
            try:
                result = dkt_train(n_epochs=n_epochs, hidden=hidden, seed=seed)
                ckpt_path = train_mod._BEST_CHECKPOINT
                trained_auc = result.get("val_auc", float("nan"))

                # load model weights
                model: torch.nn.Module | None = None
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                    M = len(concept_index)
                    dkt = DKT(M, hidden=ckpt["hidden"], layers=ckpt["layers"]).to(device)
                    dkt.load_state_dict(ckpt["model_state"])
                    dkt.eval()
                    model = dkt
            finally:
                train_mod.CHECKPOINT_DIR = old_ckpt_dir
                train_mod._BEST_CHECKPOINT = old_best
                train_mod._TRAIN_LOG = old_log

        finally:
            if old_db is None:
                os.environ.pop("DB_PATH", None)
            else:
                os.environ["DB_PATH"] = old_db

    # --- run simulations (pure Python, no DB) ---
    fsrs_log = _simulate(
        "fsrs", concepts, None, concept_index,
        n_trial_steps, random.Random(seed + 1),
        base_accuracy, improvement_rate, device,
    )
    dkt_log = _simulate(
        "dkt", concepts, model, concept_index,
        n_trial_steps, random.Random(seed + 1),   # same RNG seed → same student luck
        base_accuracy, improvement_rate, device,
    )

    return {
        "fsrs_only": fsrs_log,
        "dkt_hybrid": dkt_log,
        "n_warmup_interactions": n_warmup,
        "warmup_val_auc": trained_auc,
    }
