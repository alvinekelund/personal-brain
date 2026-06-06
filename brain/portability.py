"""Export/import the brain to a portable JSON document — for backup and migration.

Export dumps every node and edge (including archived) with all fields, so a
restore into a fresh database is exact. Import is idempotent and merge-friendly:
existing ids and same-name nodes are skipped, and edges to deduped nodes are
remapped, so re-importing or merging two brains never duplicates.
"""
import json
from brain import db

SCHEMA_VERSION = 1


def export_brain(conn) -> dict:
    nodes = [dict(r) for r in conn.execute("SELECT * FROM nodes").fetchall()]
    edges = [dict(r) for r in conn.execute("SELECT * FROM edges").fetchall()]
    return {"schema_version": SCHEMA_VERSION, "nodes": nodes, "edges": edges}


def export_to_file(conn, path) -> dict:
    data = export_brain(conn)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def import_brain(conn, data: dict) -> tuple:
    """Insert nodes/edges from an export dict. Returns (nodes_added, edges_added)."""
    existing_ids = {r["id"] for r in conn.execute("SELECT id FROM nodes")}
    existing_names = {(r["name"] or "").lower() for r in conn.execute("SELECT name FROM nodes")}
    id_map = {}  # imported node id -> resulting id in this DB (for edge remap)
    n_added = 0

    for n in data.get("nodes", []):
        nid, name = n.get("id"), (n.get("name") or "").strip()
        if not nid or not name:
            continue
        key = name.lower()
        if nid in existing_ids or key in existing_names:
            existing = db.get_node_by_name(conn, name)
            if existing:
                id_map[nid] = existing["id"]
            continue
        node_type = n.get("type", "concept")
        conn.execute(
            "INSERT INTO nodes (id,name,type,content,source,created_at,last_accessed,"
            "access_count,weight,confidence,half_life_days,archived) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, name, node_type, n.get("content", ""), n.get("source", ""),
             n.get("created_at", db.now()), n.get("last_accessed", db.now()),
             int(n.get("access_count", 0)), float(n.get("weight", 1.0)),
             float(n.get("confidence", 0.8)),
             float(n.get("half_life_days", db.HALF_LIVES.get(node_type, 60.0))),
             int(n.get("archived", 0))),
        )
        existing_ids.add(nid)
        existing_names.add(key)
        id_map[nid] = nid
        n_added += 1

    existing_edges = {
        (r["source_id"], r["target_id"], r["relation"])
        for r in conn.execute("SELECT source_id,target_id,relation FROM edges")
    }
    e_added = 0
    for e in data.get("edges", []):
        src = id_map.get(e.get("source_id"), e.get("source_id"))
        tgt = id_map.get(e.get("target_id"), e.get("target_id"))
        rel = db.normalize_relation(e.get("relation", "relates_to"))
        if not src or not tgt or src not in existing_ids or tgt not in existing_ids:
            continue
        if (src, tgt, rel) in existing_edges:
            continue
        conn.execute(
            "INSERT INTO edges (id,source_id,target_id,relation,weight,created_at,"
            "last_reinforced,reinforcement_count) VALUES (?,?,?,?,?,?,?,?)",
            (e.get("id") or db.new_id(), src, tgt, rel, float(e.get("weight", 1.0)),
             e.get("created_at", db.now()), e.get("last_reinforced", db.now()),
             int(e.get("reinforcement_count", 1))),
        )
        existing_edges.add((src, tgt, rel))
        e_added += 1

    conn.commit()
    return n_added, e_added


def import_from_file(conn, path) -> tuple:
    with open(path, encoding="utf-8") as f:
        return import_brain(conn, json.load(f))
