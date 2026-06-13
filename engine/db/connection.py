"""SQLite connection factory and schema initialisation."""
import os
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Return an open connection with row_factory and foreign-key enforcement."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the database, apply schema, and run additive migrations."""
    path = _db_path()
    db_dir = os.path.dirname(path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(concept)").fetchall()}
    if "theory_md" not in existing:
        conn.execute("ALTER TABLE concept ADD COLUMN theory_md TEXT")


def _db_path() -> str:
    return os.getenv("DB_PATH", "data/app.db")
