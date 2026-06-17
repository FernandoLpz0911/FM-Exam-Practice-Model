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
        existing_index: dict[str, int] = json.loads(INDEX_PATH.read_text())
        if set(existing_index.keys()) == set(concept_ids):
            return existing_index
        raise RuntimeError(
            "concept_index.json exists but concept IDs have changed. "
            "Delete it only if you are retraining DKT from scratch."
        )
    # Sort before assigning indices so the mapping is reproducible across
    # runs regardless of the order concept_ids happens to arrive in.
    new_index = {concept_id: i for i, concept_id in enumerate(sorted(concept_ids))}
    INDEX_PATH.write_text(json.dumps(new_index, indent=2, sort_keys=True))
    return new_index


def load() -> dict[str, int]:
    """Return the saved concept_id → integer index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "concept_index.json not found. Run `make seed` to generate it."
        )
    return json.loads(INDEX_PATH.read_text())
