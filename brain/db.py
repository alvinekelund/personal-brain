import json
import re
import sqlite3
import uuid
import time
from pathlib import Path
from brain import DB_PATH

HALF_LIVES = {
    "task":         5.0,      # actionable items — fade fast once irrelevant
    "event":        7.0,
    "fact":         21.0,
    "artifact":     30.0,     # documents, slides, files — temporary references
    "concept":      60.0,
    "insight":      90.0,
    "skill":        180.0,
    "project":      365.0,
    "person":       float("inf"),
    "organization": float("inf"),  # institutions don't expire
    "category":     float("inf"),  # structural hierarchy nodes never decay
}

# Controlled edge-relation vocabulary. Every edge is normalized to one of these
# (see normalize_relation), so extraction and synthesis stay consistent and the
# graph never accumulates free-form labels like "relies on" / "is a key part of".
RELATIONS = (
    "relates_to", "builds_on", "requires", "contradicts", "part_of",
    "studied_by", "created_by", "used_in", "assigned_to", "attended_by",
    "works_at", "member_of", "located_at",
)


def normalize_relation(relation: str) -> str:
    """Map a free-form relation label onto the controlled vocabulary.

    Exact match wins; otherwise a few keyword heuristics catch verbose LLM
    phrasings ("is a key component of" -> part_of); unknowns fall back to
    relates_to so an edge is never dropped for a bad label.
    """
    r = (relation or "").strip().lower().replace("-", "_").replace(" ", "_")
    if r in RELATIONS:
        return r
    if "part" in r or "component" in r or "belong" in r:
        return "part_of"
    if "requir" in r or "relies" in r or "rely" in r or "depend" in r or "need" in r:
        return "requires"
    if "build" in r or "extend" in r:
        return "builds_on"
    if "contradict" in r or "conflict" in r:
        return "contradicts"
    if "creat" in r or "made" in r or "author" in r or "wrote" in r:
        return "created_by"
    if "work" in r or "employ" in r:
        return "works_at"
    if "studi" in r or "study" in r or "learn" in r:
        return "studied_by"
    if "attend" in r:
        return "attended_by"
    if "member" in r:
        return "member_of"
    if "assign" in r or "request" in r:
        return "assigned_to"
    if "use" in r:
        return "used_in"
    if "locat" in r:
        return "located_at"
    return "relates_to"


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
    importance    REAL NOT NULL DEFAULT 0.5,
    half_life_days REAL NOT NULL DEFAULT 60.0,
    archived      INTEGER NOT NULL DEFAULT 0,
    embedding     TEXT
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
    # nodes.embedding (semantic search) — added later than the initial schema
    node_cols = {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}
    if "embedding" not in node_cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN embedding TEXT")
    if "importance" not in node_cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN importance REAL NOT NULL DEFAULT 0.5")


def new_id():
    return str(uuid.uuid4())


def now():
    return time.time()


# ── Nodes ──────────────────────────────────────────────────────────────────

def add_node(conn, name, type_="concept", content="", source="", confidence=0.8, importance=0.5):
    half_life = HALF_LIVES.get(type_, 60.0)
    node_id = new_id()
    t = now()
    conn.execute(
        """INSERT INTO nodes
           (id, name, type, content, source, created_at, last_accessed,
            weight, confidence, importance, half_life_days)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?, ?)""",
        (node_id, name, type_, content, source, t, t, confidence, importance, half_life),
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


PROPAGATION_DECAY = 0.6      # boost given to a parent, falling off per level
PROPAGATION_MAX_DEPTH = 3    # how far up the spine a touch reaches


def _reinforce_ancestors(conn, node_id, depth=1, seen=None):
    """Walk up the part_of spine, giving each ancestor a diminishing freshness
    boost — so using a child keeps its category/topic ancestors alive too."""
    if depth > PROPAGATION_MAX_DEPTH:
        return
    seen = seen if seen is not None else set()
    parents = conn.execute(
        "SELECT target_id FROM edges WHERE source_id = ? AND relation = 'part_of'",
        (node_id,),
    ).fetchall()
    boost = PROPAGATION_DECAY ** depth
    for p in parents:
        pid = p["target_id"]
        if pid in seen:
            continue
        seen.add(pid)
        conn.execute(
            "UPDATE nodes SET last_accessed = ?, weight = max(weight, ?) WHERE id = ?",
            (now(), boost, pid),
        )
        _reinforce_ancestors(conn, pid, depth + 1, seen)


def touch_node(conn, node_id):
    """Mark as accessed — resets decay and propagates a freshness boost up the
    part_of hierarchy so ancestors (topic, category) stay alive too."""
    conn.execute(
        """UPDATE nodes SET last_accessed = ?, access_count = access_count + 1,
           weight = 1.0 WHERE id = ?""",
        (now(), node_id),
    )
    _reinforce_ancestors(conn, node_id)


def set_embedding(conn, node_id, vector):
    """Store a node's embedding vector (JSON-encoded) for semantic search."""
    conn.execute(
        "UPDATE nodes SET embedding = ? WHERE id = ?", (json.dumps(vector), node_id)
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


_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "is", "are", "be", "about", "my", "i", "me", "this", "that", "it",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem_eq(a: str, b: str) -> bool:
    """Loose token match tolerant of simple plurals/suffixes:
    'transformer' ~ 'transformers', 'architecture' ~ 'architectures'."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long.startswith(short) and len(long) - len(short) <= 3


def search_nodes(conn, query, min_weight=0.0):
    """Token-based ranked search.

    A whole-phrase substring match ("machine learning" must appear verbatim) is
    far too brittle for a knowledge graph — the brain stores "Data Science" and
    "transformers", not the literal query phrase. Instead, split the query into
    tokens and rank each node by how many distinct query tokens appear in its
    name or content (whole-phrase matches get a bonus), then by weight.
    """
    q = query.lower().strip()
    if not q:
        return []
    tokens = {t for t in _TOKEN_RE.findall(q) if len(t) > 1 and t not in _STOPWORDS}
    if not tokens:
        tokens = {q}

    rows = conn.execute(
        "SELECT * FROM nodes WHERE archived = 0 AND weight >= ?", (min_weight,)
    ).fetchall()

    scored = []
    for r in rows:
        hay = f"{r['name']} {r['content'] or ''}".lower()
        hay_tokens = set(_TOKEN_RE.findall(hay))
        hits = sum(1 for t in tokens if any(_stem_eq(t, h) for h in hay_tokens))
        if not hits:
            continue
        if q in hay:  # exact-phrase match is a strong signal
            hits += len(tokens)
        scored.append((hits, r["weight"], r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored]


# ── Edges ──────────────────────────────────────────────────────────────────

def add_edge(conn, source_id, target_id, relation="relates_to", weight=1.0):
    relation = normalize_relation(relation)  # keep the graph on the controlled vocab
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


def merge_nodes(conn, keep_id, drop_id) -> bool:
    """Merge drop_id into keep_id: re-point drop's edges onto keep, then delete
    drop (its leftover edges cascade away). Self-loops are skipped and add_edge
    dedups/reinforces, so merging never creates loops or duplicate edges.
    Returns False if either node is missing or the ids are the same.
    """
    if keep_id == drop_id or not get_node(conn, keep_id) or not get_node(conn, drop_id):
        return False
    for edge in edges_for_node(conn, drop_id):
        src = keep_id if edge["source_id"] == drop_id else edge["source_id"]
        tgt = keep_id if edge["target_id"] == drop_id else edge["target_id"]
        if src != tgt:
            add_edge(conn, src, tgt, edge["relation"], edge["weight"])
    delete_node(conn, drop_id)
    touch_node(conn, keep_id)
    conn.commit()
    return True


# ── Ingestion log ──────────────────────────────────────────────────────────

def log_ingestion(conn, raw_text, source, node_ids, edge_ids):
    import json
    conn.execute(
        "INSERT INTO ingestion_log (id, raw_text, source, ingested_at, nodes_added, edges_added) VALUES (?,?,?,?,?,?)",
        (new_id(), raw_text[:2000], source, now(), json.dumps(node_ids), json.dumps(edge_ids)),
    )


def clear(conn) -> dict:
    """Delete ALL nodes, edges, and ingestion history. Returns counts removed."""
    counts = {
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "log": conn.execute("SELECT COUNT(*) FROM ingestion_log").fetchone()[0],
    }
    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM nodes")
    conn.execute("DELETE FROM ingestion_log")
    conn.commit()
    return counts


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
