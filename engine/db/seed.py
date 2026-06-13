"""Load concept graph seed data into SQLite."""
import json
from pathlib import Path

from engine.db.connection import get_connection, init_db
from engine.tracing.concept_index import build_and_save

SEED_PATH = Path("data/concept_graph.seed.json")
THEORY_PATH = Path("data/concept_theory.json")


def load(
    seed_path: Path = SEED_PATH,
    theory_path: Path = THEORY_PATH,
) -> None:
    """Load concept graph into SQLite and generate the DKT concept index.

    Idempotent — safe to re-run after updating the seed file.
    Raises ValueError on cycles or dangling prerequisites.
    """
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    _validate(nodes)
    theory: dict[str, str] = {}
    if theory_path.exists():
        theory = json.loads(theory_path.read_text(encoding="utf-8"))
    init_db()
    _insert(nodes, theory)
    build_and_save([n["id"] for n in nodes])
    print(f"Seeded {len(nodes)} concepts.")


def _validate(nodes: list[dict]) -> None:
    all_ids = {n["id"] for n in nodes}
    for node in nodes:
        for prereq_id in node.get("prerequisites", []):
            if prereq_id not in all_ids:
                raise ValueError(
                    f"Concept '{node['id']}' has dangling "
                    f"prerequisite '{prereq_id}'"
                )
    _assert_no_cycles(nodes)


def _assert_no_cycles(nodes: list[dict]) -> None:
    prereq_map = {
        n["id"]: set(n.get("prerequisites", [])) for n in nodes
    }
    visited: set[str] = set()
    path: set[str] = set()

    def dfs(node_id: str) -> None:
        if node_id in path:
            raise ValueError(
                f"Cycle detected involving concept '{node_id}'"
            )
        if node_id in visited:
            return
        path.add(node_id)
        for prereq_id in prereq_map.get(node_id, set()):
            dfs(prereq_id)
        path.discard(node_id)
        visited.add(node_id)

    for node_id in prereq_map:
        dfs(node_id)


def _insert(nodes: list[dict], theory: dict[str, str]) -> None:
    with get_connection() as conn:
        for node in nodes:
            gen = node.get("generator")
            conn.execute(
                """
                INSERT OR REPLACE INTO concept
                    (id, name, category, exam_weight_tier,
                     summary, generator_json, theory_md)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node["id"],
                    node["name"],
                    node["category"],
                    node["exam_weight_tier"],
                    node.get("summary"),
                    json.dumps(gen) if gen else None,
                    theory.get(node["id"]),
                ),
            )
        conn.execute("DELETE FROM concept_prereq")
        for node in nodes:
            for prereq_id in node.get("prerequisites", []):
                conn.execute(
                    "INSERT INTO concept_prereq "
                    "(concept_id, prereq_id) VALUES (?, ?)",
                    (node["id"], prereq_id),
                )


if __name__ == "__main__":
    load()
