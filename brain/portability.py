"""Export/import the brain to a portable JSON document — for backup and migration.

Export dumps every node and edge (including archived) with all fields, so a
restore into a fresh database is exact. Import is idempotent and merge-friendly:
existing ids and same-name nodes are skipped, and edges to deduped nodes are
remapped, so re-importing or merging two brains never duplicates.
"""
import json
from brain import db

SCHEMA_VERSION = 1


def export_brain(conn, lean: bool = False) -> dict:
    """Every node and edge with all fields. Embeddings ride along by default so a
    restore is complete (about 40 KB per node; `lean=True` drops them and a
    restore then needs `brain reindex`, one API call per node)."""
    nodes = []
    for r in conn.execute("SELECT * FROM nodes").fetchall():
        d = dict(r)
        d.pop("path", None)  # recomputed by `brain index`
        if lean:
            d.pop("embedding", None)
        nodes.append(d)
    edges = [dict(r) for r in conn.execute("SELECT * FROM edges").fetchall()]
    # the ingestion log is the audit trail (and what `brain doctor` reads for
    # "last ingest"); small, so a backup carries it
    log = [dict(r) for r in conn.execute("SELECT * FROM ingestion_log").fetchall()]
    return {"schema_version": SCHEMA_VERSION, "nodes": nodes, "edges": edges, "ingestion_log": log}


def export_to_file(conn, path, lean: bool = False) -> dict:
    data = export_brain(conn, lean=lean)
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
        last_accessed = n.get("last_accessed", db.now())
        emb = n.get("embedding")
        conn.execute(
            "INSERT INTO nodes (id,name,type,content,source,created_at,last_accessed,"
            "access_count,weight,confidence,half_life_days,archived,importance,last_decayed,embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, name, node_type, n.get("content", ""), n.get("source", ""),
             n.get("created_at", db.now()), last_accessed,
             int(n.get("access_count", 0)), float(n.get("weight", 1.0)),
             float(n.get("confidence", 0.8)),
             float(n.get("half_life_days", db.HALF_LIVES.get(node_type, 60.0))),
             int(n.get("archived", 0)),
             float(n.get("importance", 0.5)),          # was dropped: every restored node became 0.5
             float(n.get("last_decayed", last_accessed)),
             emb if isinstance(emb, str) else (json.dumps(emb) if emb else None)),
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

    existing_log = {r["id"] for r in conn.execute("SELECT id FROM ingestion_log")}
    for row in data.get("ingestion_log", []):
        rid = row.get("id")
        if not rid or rid in existing_log:
            continue
        conn.execute(
            "INSERT INTO ingestion_log (id, raw_text, source, ingested_at, nodes_added, edges_added)"
            " VALUES (?,?,?,?,?,?)",
            (rid, row.get("raw_text", ""), row.get("source", ""), row.get("ingested_at", db.now()),
             row.get("nodes_added", "[]"), row.get("edges_added", "[]")),
        )
        existing_log.add(rid)

    conn.commit()
    return n_added, e_added


def import_from_file(conn, path) -> tuple:
    with open(path, encoding="utf-8") as f:
        return import_brain(conn, json.load(f))
