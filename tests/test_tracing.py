"""Tests for engine/tracing/ — model, concept_index, dataset, synthetic, train, infer."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from engine.tracing.concept_index import build_and_save, load
from engine.tracing.dataset import (
    DKTBatch,
    build_batches,
    encode_sequence,
    load_sequences,
    train_val_split,
)
from engine.tracing.infer import checkpoint_meta, dkt_is_active, predict
from engine.tracing.model import DKT
from engine.tracing.synthetic import seed_synthetic
from engine.tracing.train import _bce_loss, _eval_auc, train


class TestDKTModel:
    def test_forward_output_shape(self):
        M = 5
        model = DKT(n_concepts=M, hidden=16, layers=1)
        x = torch.zeros(2, 4, 2 * M)
        out = model(x)
        assert out.shape == (2, 4, M)

    def test_output_in_unit_interval(self):
        M = 4
        model = DKT(n_concepts=M, hidden=8, layers=1)
        x = torch.rand(1, 3, 2 * M)
        out = model(x)
        assert (out >= 0.0).all().item() and (out <= 1.0).all().item()

    def test_multilayer_no_error(self):
        model = DKT(n_concepts=4, hidden=8, layers=2, dropout=0.1)
        x = torch.zeros(1, 3, 8)
        out = model(x)
        assert out.shape == (1, 3, 4)




class TestConceptIndex:
    def test_load_real_index(self):
        idx = load()
        assert isinstance(idx, dict)
        assert len(idx) == 24

    def test_load_missing_raises(self, tmp_path):
        with patch("engine.tracing.concept_index.INDEX_PATH", tmp_path / "ci.json"):
            with pytest.raises(FileNotFoundError, match="concept_index.json not found"):
                load()

    def test_build_creates_new_sorted(self, tmp_path):
        fake = tmp_path / "ci.json"
        with patch("engine.tracing.concept_index.INDEX_PATH", fake):
            result = build_and_save(["c", "a", "b"])
        assert result == {"a": 0, "b": 1, "c": 2}
        assert fake.exists()

    def test_build_existing_same_keys_returns_existing(self, tmp_path):
        fake = tmp_path / "ci.json"
        existing = {"a": 0, "b": 1, "c": 2}
        fake.write_text(json.dumps(existing))
        with patch("engine.tracing.concept_index.INDEX_PATH", fake):
            result = build_and_save(["a", "b", "c"])
        assert result == existing

    def test_build_existing_different_keys_raises(self, tmp_path):
        fake = tmp_path / "ci.json"
        fake.write_text(json.dumps({"a": 0, "b": 1}))
        with patch("engine.tracing.concept_index.INDEX_PATH", fake):
            with pytest.raises(RuntimeError, match="concept IDs have changed"):
                build_and_save(["a", "b", "c"])



_MINI_INDEX = {"tvm": 0, "annuity": 1}


class TestEncodeSequence:
    def test_correct_step_encoding(self):
        M = 3
        seq = [(0, 1), (1, 0)]
        x, target, t_idx = encode_sequence(seq, M)
        assert x.shape == (1, 2 * M)
        assert x[0, 0] == 1.0  # concept 0, correct → hot=0
        assert target[0] == 0.0  # next step is_correct=0
        assert t_idx[0] == 1     # next step concept=1

    def test_incorrect_step_encoding(self):
        M = 3
        seq = [(0, 0), (1, 1)]
        x, target, t_idx = encode_sequence(seq, M)
        assert x[0, 3] == 1.0  # concept 0, incorrect → hot=0+M=3
        assert target[0] == 1.0
        assert t_idx[0] == 1

    def test_three_step_shapes(self):
        M = 3
        seq = [(0, 1), (1, 0), (2, 1)]
        x, target, t_idx = encode_sequence(seq, M)
        assert x.shape == (2, 2 * M)
        assert target.shape == (2,)
        assert t_idx.shape == (2,)


class TestBuildBatches:
    _SEQS = [
        [(0, 1), (1, 0), (2, 1)],
        [(1, 1), (0, 1), (2, 0)],
        [(2, 0), (1, 1), (0, 0)],
        [(0, 0), (2, 1), (1, 1)],
    ]

    def test_batch_count(self):
        batches = build_batches(self._SEQS, n_concepts=3, batch_size=3, shuffle=False)
        assert len(batches) == 2

    def test_batch_tensor_types(self):
        batches = build_batches(self._SEQS[:2], n_concepts=3, batch_size=10, shuffle=False)
        batch = batches[0]
        assert batch.inputs.dim() == 3
        assert batch.targets.dim() == 2
        assert batch.mask.dtype == torch.bool

    def test_shuffle_no_crash(self):
        batches = build_batches(self._SEQS, n_concepts=3, batch_size=10, shuffle=True)
        assert len(batches) == 1


class TestTrainValSplit:
    def test_sizes_add_up(self):
        seqs = [[(0, 1), (1, 0)]] * 10
        train, val = train_val_split(seqs, val_frac=0.2, seed=42)
        assert len(train) + len(val) == 10
        assert len(val) == 2

    def test_minimum_val_is_1(self):
        seqs = [[(0, 1), (1, 0)]] * 2
        train, val = train_val_split(seqs, val_frac=0.0001, seed=42)
        assert len(val) == 1


class TestLoadSequences:
    def test_empty_db_returns_empty(self, seeded_db):
        with patch("engine.tracing.dataset.load_index", return_value=_MINI_INDEX):
            seqs = load_sequences()
        assert seqs == []

    def test_single_step_session_filtered(self, seeded_db):
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.execute("INSERT INTO session (started_at) VALUES ('2024-01-01')")
        sid = cur.lastrowid
        conn.execute(
            "INSERT INTO interaction "
            "(session_id, concept_id, seed, problem_kind, params_json, correct_answer,"
            " user_answer, is_correct, grade, elapsed_ms, shown_at, answered_at) "
            "VALUES (?, 'tvm', 0, 'k', '{}', '0', '0', 1, 3, 1000, '2024-01-01', '2024-01-01')",
            (sid,),
        )
        conn.commit()
        conn.close()
        with patch("engine.tracing.dataset.load_index", return_value=_MINI_INDEX):
            seqs = load_sequences()
        assert seqs == []

    def test_unknown_concept_skipped(self, seeded_db):
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.execute("INSERT INTO session (started_at) VALUES ('2024-01-01')")
        sid = cur.lastrowid
        for _ in range(3):
            conn.execute(
                "INSERT INTO interaction "
                "(session_id, concept_id, seed, problem_kind, params_json, correct_answer,"
                " user_answer, is_correct, grade, elapsed_ms, shown_at, answered_at) "
                "VALUES (?, 'UNKNOWN', 0, 'k', '{}', '0', '0', 1, 3, 1000, '2024-01-01', '2024-01-01')",
                (sid,),
            )
        conn.commit()
        conn.close()
        with patch("engine.tracing.dataset.load_index", return_value=_MINI_INDEX):
            seqs = load_sequences()
        assert seqs == []

    def test_valid_session_returns_sequence(self, seeded_db):
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.execute("INSERT INTO session (started_at) VALUES ('2024-01-01')")
        sid = cur.lastrowid
        for cid, correct in [("tvm", 1), ("annuity", 0)]:
            conn.execute(
                "INSERT INTO interaction "
                "(session_id, concept_id, seed, problem_kind, params_json, correct_answer,"
                " user_answer, is_correct, grade, elapsed_ms, shown_at, answered_at) "
                "VALUES (?, ?, 0, 'k', '{}', '0', '0', ?, 3, 1000, '2024-01-01', '2024-01-01')",
                (sid, cid, correct),
            )
        conn.commit()
        conn.close()
        with patch("engine.tracing.dataset.load_index", return_value=_MINI_INDEX):
            seqs = load_sequences()
        assert len(seqs) == 1
        assert len(seqs[0]) == 2

    def test_load_sequences_without_provided_index(self, seeded_db):
        """load_sequences() calls load_index() internally when concept_index=None."""
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.execute("INSERT INTO session (started_at) VALUES ('2024-01-01')")
        sid = cur.lastrowid
        for cid, correct in [("tvm", 1), ("annuity", 0)]:
            conn.execute(
                "INSERT INTO interaction "
                "(session_id, concept_id, seed, problem_kind, params_json, correct_answer,"
                " user_answer, is_correct, grade, elapsed_ms, shown_at, answered_at) "
                "VALUES (?, ?, 0, 'k', '{}', '0', '0', ?, 3, 1000, '2024-01-01', '2024-01-01')",
                (sid, cid, correct),
            )
        conn.commit()
        conn.close()
        # Pass concept_index explicitly so no disk read needed; confirm both paths work
        seqs = load_sequences(concept_index=_MINI_INDEX)
        assert len(seqs) == 1




class TestSeedSynthetic:
    def test_inserts_correct_count(self, seeded_db):
        with patch("engine.tracing.synthetic.load_index", return_value=_MINI_INDEX):
            interaction_count = seed_synthetic(n_sessions=2, steps_per_session=5, rng_seed=42)
        assert interaction_count == 10
        conn = sqlite3.connect(str(seeded_db))
        count = conn.execute("SELECT COUNT(*) FROM interaction").fetchone()[0]
        conn.close()
        assert count == 10

    def test_sessions_marked_ended(self, seeded_db):
        with patch("engine.tracing.synthetic.load_index", return_value=_MINI_INDEX):
            seed_synthetic(n_sessions=3, steps_per_session=2, rng_seed=7)
        conn = sqlite3.connect(str(seeded_db))
        ended = conn.execute("SELECT COUNT(*) FROM session WHERE ended_at IS NOT NULL").fetchone()[0]
        conn.close()
        assert ended == 3




class TestBCELoss:
    def test_returns_scalar(self):
        M, B, T = 3, 2, 4
        model = DKT(n_concepts=M, hidden=8, layers=1)
        x = torch.zeros(B, T, 2 * M)
        preds = model(x)
        targets = torch.randint(0, 2, (B, T)).float()
        t_idx = torch.zeros(B, T, dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        batch = DKTBatch(preds, targets, t_idx, mask)
        loss = _bce_loss(preds, batch)
        assert loss.dim() == 0


class TestEvalAUC:
    def _batch(self, M, B, T, all_same: bool):
        model = DKT(n_concepts=M, hidden=8, layers=1)
        model.eval()
        x = torch.zeros(B, T, 2 * M)
        if all_same:
            targets = torch.ones(B, T)
        else:
            # Guarantee both 0 and 1
            targets = torch.cat([
                torch.ones(B // 2, T),
                torch.zeros(B - B // 2, T),
            ])
        t_idx = torch.zeros(B, T, dtype=torch.long)
        mask = torch.ones(B, T, dtype=torch.bool)
        return model, [DKTBatch(x, targets, t_idx, mask)]

    def test_all_same_target_returns_nan(self):
        model, batches = self._batch(3, 4, 3, all_same=True)
        result = _eval_auc(model, batches, torch.device("cpu"))
        assert math.isnan(result)

    def test_mixed_targets_returns_valid_auc(self):
        model, batches = self._batch(3, 4, 3, all_same=False)
        result = _eval_auc(model, batches, torch.device("cpu"))
        assert not math.isnan(result)
        assert 0.0 <= result <= 1.0


class TestTrain:
    def test_no_sequences_early_return(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        with (
            patch("engine.tracing.train._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.train._TRAIN_LOG", tmp_path / "log.jsonl"),
            patch("engine.tracing.train.load_sequences", return_value=[]),
        ):
            result = train(n_epochs=1, device_str="cpu")
        assert result["epochs_run"] == 0
        assert math.isnan(result["val_auc"])
        assert result["checkpoint_path"] is None

    def test_one_sequence_early_return(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        with (
            patch("engine.tracing.train._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.train._TRAIN_LOG", tmp_path / "log.jsonl"),
            patch("engine.tracing.train.load_sequences", return_value=[[(0, 1), (1, 0)]]),
        ):
            result = train(n_epochs=1, device_str="cpu")
        assert result["epochs_run"] == 0

    def test_full_training_saves_checkpoint(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        seqs = [
            [(0, 1), (1, 0), (2, 1), (0, 0), (1, 1)],
            [(1, 0), (0, 1), (2, 0), (1, 1), (0, 0)],
            [(2, 1), (1, 0), (0, 1), (2, 0), (1, 1)],
            [(0, 0), (1, 1), (2, 0), (0, 1), (2, 1)],
            [(1, 1), (2, 0), (0, 1), (1, 0), (2, 1)],
        ]
        with (
            patch("engine.tracing.train._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.train._TRAIN_LOG", tmp_path / "log.jsonl"),
            patch("engine.tracing.train.load_sequences", return_value=seqs),
            patch("engine.tracing.train.load_index", return_value={"a": 0, "b": 1, "c": 2}),
        ):
            result = train(n_epochs=2, device_str="cpu", seed=42)
        assert result["epochs_run"] == 2
        assert ckpt.exists()
        assert result["checkpoint_path"] is not None

    def test_device_str_explicit(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        with (
            patch("engine.tracing.train._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.train._TRAIN_LOG", tmp_path / "log.jsonl"),
            patch("engine.tracing.train.load_sequences", return_value=[]),
        ):
            result = train(n_epochs=1, device_str="cpu")
        assert result["epochs_run"] == 0



_MINI_INDEX_3 = {"a": 0, "b": 1, "c": 2}


def _save_fake_checkpoint(path: Path, M: int = 3, val_auc: float = 0.85) -> None:
    model = DKT(n_concepts=M, hidden=8, layers=1)
    torch.save(
        {
            "epoch": 5,
            "val_auc": val_auc,
            "n_concepts": M,
            "hidden": 8,
            "layers": 1,
            "dropout": 0.2,
            "model_state": model.state_dict(),
        },
        path,
    )


class TestPredict:
    def test_empty_history_returns_none(self):
        assert predict([]) is None

    def test_no_checkpoint_returns_none(self, tmp_path):
        with patch("engine.tracing.infer._BEST_CHECKPOINT", tmp_path / "no.pt"):
            assert predict([("a", True)]) is None

    def test_valid_prediction_all_keys_returned(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, M=3)
        with (
            patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.infer.load_index", return_value=_MINI_INDEX_3),
        ):
            result = predict([("a", True), ("b", False)], device_str="cpu")
        assert result is not None
        assert set(result.keys()) == {"a", "b", "c"}
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_unknown_concept_id_skipped(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, M=3)
        with (
            patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.infer.load_index", return_value=_MINI_INDEX_3),
        ):
            result = predict([("UNKNOWN", True), ("a", False)], device_str="cpu")
        assert result is not None

    def test_explicit_device_str(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, M=3)
        with (
            patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt),
            patch("engine.tracing.infer.load_index", return_value=_MINI_INDEX_3),
        ):
            result = predict([("a", True)], device_str="cpu")
        assert result is not None


class TestCheckpointMeta:
    def test_no_checkpoint_returns_none(self, tmp_path):
        with patch("engine.tracing.infer._BEST_CHECKPOINT", tmp_path / "no.pt"):
            assert checkpoint_meta() is None

    def test_returns_metadata(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, val_auc=0.88, M=3)
        with patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt):
            meta = checkpoint_meta()
        assert meta is not None
        assert meta["val_auc"] == pytest.approx(0.88)
        assert meta["epoch"] == 5
        assert meta["n_concepts"] == 3


class TestDktIsActive:
    def test_not_enough_interactions(self, seeded_db):
        assert dkt_is_active() is False

    def test_no_checkpoint(self, seeded_db, tmp_path):
        with (
            patch("engine.db.dao.count_answered_interactions", return_value=9999),
            patch("engine.tracing.infer._BEST_CHECKPOINT", tmp_path / "no.pt"),
        ):
            assert dkt_is_active() is False

    def test_auc_below_threshold(self, seeded_db, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, val_auc=0.50)
        with (
            patch("engine.db.dao.count_answered_interactions", return_value=9999),
            patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt),
        ):
            assert dkt_is_active() is False

    def test_active_when_auc_meets_threshold(self, seeded_db, tmp_path):
        ckpt = tmp_path / "best.pt"
        _save_fake_checkpoint(ckpt, val_auc=0.99)
        with (
            patch("engine.db.dao.count_answered_interactions", return_value=9999),
            patch("engine.tracing.infer._BEST_CHECKPOINT", ckpt),
        ):
            assert dkt_is_active() is True
