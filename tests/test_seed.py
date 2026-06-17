"""Tests for engine/db/seed.py — concept graph loading."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

_NODES = [
    {
        "id": "alpha",
        "name": "Alpha",
        "category": "interest",
        "exam_weight_tier": 2,
        "summary": "Alpha concept",
        "prerequisites": [],
        "generator": None,
    },
    {
        "id": "beta",
        "name": "Beta",
        "category": "annuity",
        "exam_weight_tier": 1,
        "summary": "Beta concept",
        "prerequisites": ["alpha"],
        "generator": {
            "kind": "annuity_immediate",
            "params": {"ask": ["pv_annuity_imm"], "i_range": [0.04, 0.08], "n_range": [5, 10], "payment_range": [500, 2000]},
        },
    },
]


class TestValidate:
    def test_valid_nodes_passes(self):
        from engine.db.seed import _validate
        _validate(_NODES)  # must not raise

    def test_dangling_prereq_raises(self):
        from engine.db.seed import _validate
        nodes = [{"id": "x", "prerequisites": ["nonexistent"]}]
        with pytest.raises(ValueError, match="dangling prerequisite"):
            _validate(nodes)

    def test_no_prereqs_passes(self):
        from engine.db.seed import _validate
        _validate([{"id": "x", "prerequisites": []}])


class TestAssertNoCycles:
    def test_direct_cycle_raises(self):
        from engine.db.seed import _assert_no_cycles
        nodes = [
            {"id": "a", "prerequisites": ["b"]},
            {"id": "b", "prerequisites": ["a"]},
        ]
        with pytest.raises(ValueError, match="Cycle"):
            _assert_no_cycles(nodes)

    def test_self_loop_raises(self):
        from engine.db.seed import _assert_no_cycles
        nodes = [{"id": "a", "prerequisites": ["a"]}]
        with pytest.raises(ValueError, match="Cycle"):
            _assert_no_cycles(nodes)

    def test_dag_passes(self):
        from engine.db.seed import _assert_no_cycles
        nodes = [
            {"id": "a", "prerequisites": []},
            {"id": "b", "prerequisites": ["a"]},
            {"id": "c", "prerequisites": ["a", "b"]},
        ]
        _assert_no_cycles(nodes)  # must not raise

    def test_already_visited_not_re_explored(self):
        from engine.db.seed import _assert_no_cycles
        # Diamond dependency — a -> b, a -> c, b -> d, c -> d
        nodes = [
            {"id": "a", "prerequisites": ["b", "c"]},
            {"id": "b", "prerequisites": ["d"]},
            {"id": "c", "prerequisites": ["d"]},
            {"id": "d", "prerequisites": []},
        ]
        _assert_no_cycles(nodes)  # must not raise


class TestLoad:
    def test_load_with_theory(self, isolated_db, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps({"nodes": _NODES}), encoding="utf-8")
        theory_file = tmp_path / "theory.json"
        theory_file.write_text(json.dumps({"alpha": "Theory for alpha"}), encoding="utf-8")

        with patch("engine.db.seed.build_and_save"):
            from engine.db import seed as seed_mod
            seed_mod.load(seed_file, theory_file)

        from engine.db.dao import get_all_concepts
        concepts = get_all_concepts()
        ids = {c.id for c in concepts}
        assert "alpha" in ids
        assert "beta" in ids

    def test_load_without_theory(self, isolated_db, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps({"nodes": _NODES}), encoding="utf-8")
        missing_theory = tmp_path / "no_such_theory.json"

        with patch("engine.db.seed.build_and_save"):
            from engine.db import seed as seed_mod
            seed_mod.load(seed_file, missing_theory)

        from engine.db.dao import get_all_concepts
        concepts = get_all_concepts()
        assert len(concepts) == 2

    def test_load_inserts_prereqs(self, isolated_db, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps({"nodes": _NODES}), encoding="utf-8")

        with patch("engine.db.seed.build_and_save"):
            from engine.db import seed as seed_mod
            seed_mod.load(seed_file, Path("/nonexistent/theory.json"))

        from engine.db.dao import get_concept
        beta = get_concept("beta")
        assert "alpha" in beta.prerequisites

    def test_load_calls_build_and_save(self, isolated_db, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps({"nodes": _NODES}), encoding="utf-8")

        with patch("engine.db.seed.build_and_save") as mock_bas:
            from engine.db import seed as seed_mod
            seed_mod.load(seed_file, Path("/nonexistent/theory.json"))

        mock_bas.assert_called_once_with(["alpha", "beta"])
