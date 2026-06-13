"""Stable concept_id → integer mapping required by the DKT model.

The index is written once on first seed and must not change while a trained
checkpoint exists, because the model's input/output dimensions are fixed to M.
"""
import json
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "concept_index.json"


def build_and_save(concept_ids: list[str]) -> dict[str, int]:
    """Build a stable concept_id → integer index sorted by concept_id.

    Written once on first seed. Subsequent calls verify the IDs match — raises
    if they have changed, because reindexing invalidates any trained DKT checkpoint.
    """
    if INDEX_PATH.exists():
        existing: dict[str, int] = json.loads(INDEX_PATH.read_text())
        if set(existing.keys()) == set(concept_ids):
            return existing
        raise RuntimeError(
            "concept_index.json exists but concept IDs have changed. "
            "Delete it only if you are retraining DKT from scratch."
        )
    index = {cid: i for i, cid in enumerate(sorted(concept_ids))}
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def load() -> dict[str, int]:
    """Return the saved concept_id → integer index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "concept_index.json not found. Run `make seed` to generate it."
        )
    return json.loads(INDEX_PATH.read_text())
