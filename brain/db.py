import sqlite3
import uuid
import time
from pathlib import Path
from brain import DB_PATH

HALF_LIVES = {
    "event":   7.0,
    "fact":    21.0,
    "concept": 60.0,
    "insight": 90.0,
    "skill":   180.0,
    "project": 365.0,
    "person":  float("inf"),
}

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS nodes (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    type          TEXT NOT NULL DEFAULT 'concept',
    content       TEXT,
    source        TEXT,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    weight        REAL NOT NULL DEFAULT 1.0,
    confidence    REAL NOT NULL DEFAULT 0.8,
    half_life_days REAL NOT NULL DEFAULT 60.0,
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    relation            TEXT NOT NULL DEFAULT 'relates_to',
    weight              REAL NOT NULL DEFAULT 1.0,
    created_at          REAL NOT NULL,
    last_reinforced     REAL NOT NULL DEFAULT 0,
    reinforcement_count INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id           TEXT PRIMARY KEY,
    raw_text     TEXT,
    source       TEXT,
    ingested_at  REAL NOT NULL,
    nodes_added  TEXT,
    edges_added  TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_name    ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_type    ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_weight  ON nodes(weight);
CREATE INDEX IF NOT EXISTS idx_edges_source  ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target  ON edges(target_id);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn):
    """Add columns introduced after initial schema without breaking existing DBs."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(edges)")}
    for col, defn in [
        ("last_reinforced", "REAL NOT NULL DEFAULT 0"),
        ("reinforcement_count", "INTEGER NOT NULL DEFAULT 1"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE edges ADD COLUMN {col} {defn}")
    # backfill last_reinforced = created_at where still 0
    conn.execute("UPDATE edges SET last_reinforced = created_at WHERE last_reinforced = 0")


def new_id():
    return str(uuid.uuid4())


def now():
    return time.time()


# ── Nodes ──────────────────────────────────────────────────────────────────

def add_node(conn, name, type_="concept", content="", source="", confidence=0.8):
    half_life = HALF_LIVES.get(type_, 60.0)
    node_id = new_id()
    t = now()
    conn.execute(
        """INSERT INTO nodes
           (id, name, type, content, source, created_at, last_accessed,
            weight, confidence, half_life_days)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?)""",
        (node_id, name, type_, content, source, t, t, confidence, half_life),
    )
    return node_id


def get_node(conn, node_id):
    return conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()


def get_node_by_name(conn, name):
    return conn.execute(
        "SELECT * FROM nodes WHERE lower(name) = lower(?)", (name,)
    ).fetchone()


def all_nodes(conn, include_archived=False, min_weight=0.0):
    q = "SELECT * FROM nodes WHERE weight >= ?"
    params = [min_weight]
    if not include_archived:
        q += " AND archived = 0"
    return conn.execute(q, params).fetchall()


def touch_node(conn, node_id):
    """Mark as accessed — resets decay."""
    conn.execute(
        """UPDATE nodes SET last_accessed = ?, access_count = access_count + 1,
           weight = 1.0 WHERE id = ?""",
        (now(), node_id),
    )


def archive_node(conn, node_id):
    conn.execute("UPDATE nodes SET archived = 1 WHERE id = ?", (node_id,))


def delete_node(conn, node_id):
    conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))


def ensure_identity_anchor(conn, name: str):
    """Create the user's identity node if it doesn't exist. Never decays."""
    existing = get_node_by_name(conn, name)
    if not existing:
        add_node(conn, name=name, type_="person", content=f"The owner of this brain.", confidence=1.0)
        conn.commit()


def search_nodes(conn, query, min_weight=0.0):
    q = query.lower()
    return conn.execute(
        """SELECT * FROM nodes
           WHERE (lower(name) LIKE ? OR lower(content) LIKE ?)
             AND archived = 0 AND weight >= ?
           ORDER BY weight DESC""",
        (f"%{q}%", f"%{q}%", min_weight),
    ).fetchall()


# ── Edges ──────────────────────────────────────────────────────────────────

def add_edge(conn, source_id, target_id, relation="relates_to", weight=1.0):
    existing = conn.execute(
        "SELECT id FROM edges WHERE source_id=? AND target_id=? AND relation=?",
        (source_id, target_id, relation),
    ).fetchone()
    if existing:
        # Hebbian reinforcement: strengthen the connection and reset decay clock
        conn.execute(
            """UPDATE edges
               SET weight = min(1.0, weight + 0.15),
                   last_reinforced = ?,
                   reinforcement_count = reinforcement_count + 1
               WHERE id = ?""",
            (now(), existing["id"]),
        )
        return existing["id"]
    edge_id = new_id()
    t = now()
    conn.execute(
        """INSERT INTO edges
           (id, source_id, target_id, relation, weight, created_at, last_reinforced, reinforcement_count)
           VALUES (?,?,?,?,?,?,?,1)""",
        (edge_id, source_id, target_id, relation, weight, t, t),
    )
    return edge_id


def edges_for_node(conn, node_id):
    return conn.execute(
        "SELECT * FROM edges WHERE source_id = ? OR target_id = ?",
        (node_id, node_id),
    ).fetchall()


def all_edges(conn):
    return conn.execute("SELECT * FROM edges").fetchall()


# ── Ingestion log ──────────────────────────────────────────────────────────

def log_ingestion(conn, raw_text, source, node_ids, edge_ids):
    import json
    conn.execute(
        "INSERT INTO ingestion_log (id, raw_text, source, ingested_at, nodes_added, edges_added) VALUES (?,?,?,?,?,?)",
        (new_id(), raw_text[:2000], source, now(), json.dumps(node_ids), json.dumps(edge_ids)),
    )


# ── Stats ──────────────────────────────────────────────────────────────────

def stats(conn):
    n_total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    n_active = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    n_archived = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=1").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    by_type = conn.execute(
        "SELECT type, COUNT(*) as n FROM nodes WHERE archived=0 GROUP BY type"
    ).fetchall()
    avg_weight = conn.execute(
        "SELECT AVG(weight) FROM nodes WHERE archived=0"
    ).fetchone()[0] or 0.0
    return {
        "total": n_total,
        "active": n_active,
        "archived": n_archived,
        "edges": n_edges,
        "by_type": {r["type"]: r["n"] for r in by_type},
        "avg_weight": round(avg_weight, 3),
    }
