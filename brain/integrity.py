"""Graph integrity — the invariants the person-rooted tree must keep, checked and repaired.

The hierarchy is only useful if it stays a tree: every non-category node has
exactly one `part_of` parent, every category hangs directly off the person, and
nothing is orphaned or cyclic. In Sep 2026 two days of ingests, a reorganize and
a decay bug left 12 orphans, 17 multi-parent nodes and one cycle — all silently.
`check` finds those (plus missing embeddings and near-duplicate names, which
degrade retrieval); `repair` fixes the structural ones deterministically.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from brain import db

FALLBACK_CATEGORY = {
    "person": "Relationships", "organization": "Organizations", "skill": "Skills",
    "project": "Projects", "event": "Events", "artifact": "Artifacts",
    "fact": "Knowledge", "insight": "Insights", "concept": "Knowledge",
}
DUP_RATIO = 0.9
_STRIP = re.compile(r"[^a-z0-9 ]+")
_POSSESSIVE = re.compile(r"\b(?:alvin'?s?|my|the)\b")


@dataclass
class Report:
    orphans: list[str] = field(default_factory=list)              # non-category nodes with no part_of parent
    multi_parent: list[tuple[str, list[str]]] = field(default_factory=list)
    unrooted_categories: list[str] = field(default_factory=list)  # category with no parent
    category_bad_parent: list[tuple[str, list[str]]] = field(default_factory=list)  # category whose parent isn't the person
    under_identity: list[str] = field(default_factory=list)       # non-category node hanging off the person
    cycles: list[list[str]] = field(default_factory=list)         # part_of cycles
    missing_embeddings: int = 0
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # suspiciously similar names, same type
    legacy_tasks: list[str] = field(default_factory=list)

    @property
    def structural(self) -> int:
        return (len(self.orphans) + len(self.multi_parent) + len(self.unrooted_categories)
                + len(self.category_bad_parent) + len(self.under_identity) + len(self.cycles))

    @property
    def clean(self) -> bool:
        return self.structural == 0 and not self.duplicates and not self.legacy_tasks

    def summary(self) -> str:
        bits = []
        if self.orphans: bits.append(f"{len(self.orphans)} orphan(s)")
        if self.multi_parent: bits.append(f"{len(self.multi_parent)} multi-parent")
        if self.unrooted_categories: bits.append(f"{len(self.unrooted_categories)} unrooted categor(ies)")
        if self.category_bad_parent: bits.append(f"{len(self.category_bad_parent)} categor(ies) under a non-person")
        if self.under_identity: bits.append(f"{len(self.under_identity)} node(s) directly under the person")
        if self.cycles: bits.append(f"{len(self.cycles)} cycle(s)")
        if self.legacy_tasks: bits.append(f"{len(self.legacy_tasks)} legacy task node(s)")
        if self.duplicates: bits.append(f"{len(self.duplicates)} possible duplicate pair(s)")
        if self.missing_embeddings: bits.append(f"{self.missing_embeddings} node(s) without embeddings (brain reindex)")
        return "; ".join(bits) if bits else "tree intact"


def _norm(name: str) -> str:
    s = _POSSESSIVE.sub(" ", name.lower())
    s = _STRIP.sub(" ", s)
    return " ".join(s.split())


def _parents(conn, nid: str) -> list[str]:
    return [e["target_id"] for e in db.edges_for_node(conn, nid)
            if e["source_id"] == nid and e["relation"] == "part_of"]


def check(conn, user: str = "") -> Report:
    r = Report()
    nodes = {n["id"]: n for n in db.all_nodes(conn)}
    ident = db.get_node_by_name(conn, user) if user else None
    ident_id = ident["id"] if ident else None
    parents = {nid: [p for p in _parents(conn, nid) if p in nodes] for nid in nodes}
    for nid, n in nodes.items():
        if nid == ident_id:
            continue
        ps = parents[nid]
        names = [nodes[p]["name"] for p in ps]
        if n["type"] == "category":
            if not ps:
                r.unrooted_categories.append(n["name"])
            elif ident_id and (len(ps) > 1 or ps[0] != ident_id):
                r.category_bad_parent.append((n["name"], names))
        else:
            if not ps:
                r.orphans.append(n["name"])
            elif len(ps) > 1:
                r.multi_parent.append((n["name"], names))
            if ident_id and ident_id in ps:
                r.under_identity.append(n["name"])
        if n["type"] == "task":
            r.legacy_tasks.append(n["name"])
        emb = n["embedding"] if "embedding" in n.keys() else None
        if not emb:
            r.missing_embeddings += 1
    # cycles along part_of (single-parent walk from every node; multi-parent nodes follow every parent)
    seen_cycles: set[frozenset] = set()
    for start in nodes:
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            for p in parents.get(cur, []):
                if p == start:
                    key = frozenset(path)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        r.cycles.append([nodes[x]["name"] for x in path])
                elif p not in path and len(path) < 40:
                    stack.append((p, path + [p]))
    # near-duplicate names within a type (categories excluded)
    by_type: dict[str, list] = {}
    for n in nodes.values():
        if n["type"] != "category":
            by_type.setdefault(n["type"], []).append(n)
    for items in by_type.values():
        normed = [(_norm(n["name"]), n["name"]) for n in items]
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                a, b = normed[i][0], normed[j][0]
                if not a or not b:
                    continue
                ta, tb = set(a.split()), set(b.split())
                # people: a bare first name and the full name are almost always one person
                subset = items[i]["type"] == "person" and (ta <= tb or tb <= ta)
                if a == b or subset or difflib.SequenceMatcher(None, a, b).ratio() >= DUP_RATIO:
                    r.duplicates.append((normed[i][1], normed[j][1]))
    return r


def _subtree_size(conn, nid: str, kids: dict, seen: set | None = None) -> int:
    seen = seen or set()
    seen.add(nid)
    return 1 + sum(_subtree_size(conn, c, kids, seen) for c in kids.get(nid, []) if c not in seen)


def repair(conn, user: str) -> dict:
    """Fix structural problems in place. Returns counts per fix. Deterministic:
    - multi-parent: keep the most specific parent (a non-category if there is exactly
      one; else the newest edge), drop the rest;
    - categories: exactly one parent, the person;
    - node directly under the person: re-home under its type's fallback category;
    - orphan: attach to its type's fallback category (created and rooted if needed);
    - cycle: cut the part_of edge leaving the node with the larger subtree.
    Never deletes or renames nodes; duplicates and embeddings are reported, not fixed."""
    out = {"multi_parent": 0, "categories": 0, "under_identity": 0, "orphans": 0, "cycles": 0}
    db.ensure_identity_anchor(conn, user)
    ident = db.get_node_by_name(conn, user)["id"]

    def ensure_category(name: str) -> str:
        cat = db.get_node_by_name(conn, name)
        if cat:
            cid = cat["id"]
        else:
            cid = db.add_node(conn, name=name, type_="category", source="repair", importance=0.85)
        if not _parents(conn, cid):
            db.add_edge(conn, cid, ident, "part_of")
        return cid

    def part_of_edges(nid):
        return [e for e in db.edges_for_node(conn, nid) if e["source_id"] == nid and e["relation"] == "part_of"]

    nodes = {n["id"]: n for n in db.all_nodes(conn)}
    # 1. categories → only the person
    for nid, n in nodes.items():
        if n["type"] != "category" or nid == ident:
            continue
        edges = part_of_edges(nid)
        if len(edges) != 1 or edges[0]["target_id"] != ident:
            for e in edges:
                db.delete_edge(conn, e["id"])
            db.add_edge(conn, nid, ident, "part_of")
            out["categories"] += 1
    # 2. non-category nodes: exactly one parent, never the person
    for nid, n in nodes.items():
        if n["type"] == "category" or nid == ident:
            continue
        edges = [e for e in part_of_edges(nid) if e["target_id"] in nodes]
        if any(e["target_id"] == ident for e in edges):
            for e in edges:
                if e["target_id"] == ident:
                    db.delete_edge(conn, e["id"])
            edges = [e for e in edges if e["target_id"] != ident]
            if not edges:
                db.add_edge(conn, nid, ensure_category(FALLBACK_CATEGORY.get(n["type"], "Misc")), "part_of")
                edges = part_of_edges(nid)
            out["under_identity"] += 1
        if len(edges) > 1:
            specific = [e for e in edges if nodes[e["target_id"]]["type"] != "category"]
            keep = specific[0] if len(specific) == 1 else max(edges, key=lambda e: e["created_at"])
            for e in edges:
                if e["id"] != keep["id"]:
                    db.delete_edge(conn, e["id"])
            out["multi_parent"] += 1
        elif not edges:
            db.add_edge(conn, nid, ensure_category(FALLBACK_CATEGORY.get(n["type"], "Misc")), "part_of")
            out["orphans"] += 1
    conn.commit()
    # 3. cycles: cut the edge leaving the bigger subtree (a hub must not be its member's child)
    for cyc in check(conn, user).cycles:
        ids = [db.get_node_by_name(conn, name)["id"] for name in cyc]
        kids: dict = {}
        for e in conn.execute("SELECT source_id, target_id FROM edges WHERE relation='part_of'"):
            kids.setdefault(e[1], []).append(e[0])
        victim = max(ids, key=lambda i: _subtree_size(conn, i, kids))
        for e in part_of_edges(victim):
            if e["target_id"] in ids:
                db.delete_edge(conn, e["id"])
                db.add_edge(conn, victim, ensure_category(FALLBACK_CATEGORY.get(nodes[victim]["type"], "Misc")), "part_of")
                out["cycles"] += 1
                break
    conn.commit()
    return out
