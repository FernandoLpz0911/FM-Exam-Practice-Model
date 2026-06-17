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
    all_ids = {node["id"] for node in nodes}
    for node in nodes:
        for prereq_id in node.get("prerequisites", []):
            if prereq_id not in all_ids:
                raise ValueError(
                    f"Concept '{node['id']}' has dangling "
                    f"prerequisite '{prereq_id}'"
                )
    _assert_no_cycles(nodes)


def _assert_no_cycles(nodes: list[dict]) -> None:
    """Reject seed data with circular prerequisites, which would make the
    concept graph un-orderable — a learner could never "finish" prerequisites
    for a concept that's also one of its own (indirect) prerequisites.
    """
    prereqs_by_id = {
        node["id"]: set(node.get("prerequisites", [])) for node in nodes
    }
    fully_checked: set[str] = set()
    path_in_progress: set[str] = set()   # nodes on the current DFS stack

    def visit(node_id: str) -> None:
        if node_id in path_in_progress:
            raise ValueError(
                f"Cycle detected involving concept '{node_id}'"
            )
        if node_id in fully_checked:
            return
        path_in_progress.add(node_id)
        for prereq_id in prereqs_by_id.get(node_id, set()):
            visit(prereq_id)
        path_in_progress.discard(node_id)
        fully_checked.add(node_id)

    for node_id in prereqs_by_id:
        visit(node_id)


def _insert(nodes: list[dict], theory: dict[str, str]) -> None:
    with get_connection() as conn:
        for node in nodes:
            generator_config = node.get("generator")
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
                    json.dumps(generator_config) if generator_config else None,
                    theory.get(node["id"]),
                ),
            )
        # Re-seeding always wipes and rebuilds prerequisite edges from
        # scratch — simpler than diffing old vs new edges, and safe because
        # this whole function runs inside one connection/transaction.
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
