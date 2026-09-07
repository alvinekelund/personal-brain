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
from datetime import date
import re
from dataclasses import dataclass, field

from brain import db

FALLBACK_CATEGORY = {
    "person": "Relationships", "organization": "Organizations", "skill": "Skills",
    "project": "Projects", "event": "Events", "artifact": "Artifacts",
    "fact": "Knowledge", "insight": "Insights", "concept": "Knowledge",
}
DUP_RATIO = 0.9
SEMANTIC_DUP_MIN = 0.89   # cosine between two same-type nodes' stored embeddings that means "the same thing twice"
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
    dangling_edges: int = 0                                       # edges whose source/target node no longer exists
    missing_embeddings: int = 0
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # suspiciously similar names, same type
    semantic_duplicates: list[tuple[str, str, float]] = field(default_factory=list)  # same type, embeddings agree (also in duplicates)
    legacy_tasks: list[str] = field(default_factory=list)
    oversized: list[tuple[str, int]] = field(default_factory=list)   # categories with too many direct children
    flat_lists: list[tuple[str, int]] = field(default_factory=list)  # a non-category node with that many children: a list, not structure
    thin_areas: list[tuple[str, int]] = field(default_factory=list)  # top-level categories with that few descendants
    fact_parents: list[tuple[str, list[str]]] = field(default_factory=list)  # a fact with children: a leaf used as a container

    @property
    def structural(self) -> int:
        return (len(self.orphans) + len(self.multi_parent) + len(self.unrooted_categories)
                + len(self.category_bad_parent) + len(self.under_identity) + len(self.cycles)
                + self.dangling_edges)

    @property
    def clean(self) -> bool:
        return (self.structural == 0 and not self.duplicates and not self.legacy_tasks
                and not self.oversized and not self.flat_lists and not self.thin_areas
                and not self.fact_parents)

    def summary(self) -> str:
        bits = []
        if self.orphans: bits.append(f"{len(self.orphans)} orphan(s)")
        if self.multi_parent: bits.append(f"{len(self.multi_parent)} multi-parent")
        if self.unrooted_categories: bits.append(f"{len(self.unrooted_categories)} unrooted categor(ies)")
        if self.category_bad_parent: bits.append(f"{len(self.category_bad_parent)} categor(ies) under a non-person")
        if self.under_identity: bits.append(f"{len(self.under_identity)} node(s) directly under the person")
        if self.cycles: bits.append(f"{len(self.cycles)} cycle(s)")
        if self.dangling_edges: bits.append(f"{self.dangling_edges} dangling edge(s) to deleted nodes")
        if self.legacy_tasks: bits.append(f"{len(self.legacy_tasks)} legacy task node(s)")
        if self.duplicates: bits.append(f"{len(self.duplicates)} possible duplicate pair(s)")
        if self.oversized:
            bits.append("oversized: " + ", ".join(f"{n} ({c})" for n, c in self.oversized[:4]) + " (brain subgroup)")
        if self.flat_lists:
            bits.append("flat list(s): " + ", ".join(f"{n} ({c})" for n, c in self.flat_lists[:4])
                        + " (one fact naming the list, not one node per name: brain merge / brain forget)")
        if self.fact_parents:
            bits.append(f"{len(self.fact_parents)} fact(s) used as a parent: "
                        + ", ".join(f"{n} ← {', '.join(c[:3])}" for n, c in self.fact_parents[:3])
                        + " (a fact is a leaf: brain move the children to the entity the fact is about)")
        if self.thin_areas:
            bits.append("thin area(s): " + ", ".join(f"{n} ({c})" for n, c in self.thin_areas[:4])
                        + " (a sub-category or a duplicate of a broader area: brain move <area> <broader> / brain merge <broader> <area>)")
        if self.missing_embeddings: bits.append(f"{self.missing_embeddings} node(s) without embeddings (brain reindex)")
        return "; ".join(bits) if bits else "tree intact"


def _norm(name: str) -> str:
    s = _POSSESSIVE.sub(" ", name.lower())
    s = _STRIP.sub(" ", s)
    return " ".join(s.split())


def _parents(conn, nid: str) -> list[str]:
    return [e["target_id"] for e in db.edges_for_node(conn, nid)
            if e["source_id"] == nid and e["relation"] == "part_of"]


# shared by the SELECT in check() and the DELETE in repair(); no table alias —
# SQLite's DELETE does not accept one
_DANGLING_WHERE = (
    "WHERE NOT EXISTS (SELECT 1 FROM nodes WHERE nodes.id = edges.source_id)"
    " OR NOT EXISTS (SELECT 1 FROM nodes WHERE nodes.id = edges.target_id)"
)


THIN_AREA_MAX = 2          # a top-level area with this many descendants or fewer is thin
THIN_AREA_MIN_GRAPH = 20   # ...once the graph has this many non-category nodes (a young brain is just small)


def check(conn, user: str = "", oversized_threshold: int | None = None) -> Report:
    r = Report()
    if oversized_threshold is None:
        from brain.extract import SUBGROUP_THRESHOLD
        oversized_threshold = SUBGROUP_THRESHOLD
    # edges left behind by a deleted node (the schema has no ON DELETE CASCADE):
    # invisible to the tree walk below, which is exactly why they must be counted
    r.dangling_edges = conn.execute(f"SELECT COUNT(*) FROM edges {_DANGLING_WHERE}").fetchone()[0]
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
            # a category hangs off the person or off another category (a
            # sub-category made by `subgroup_categories`); never off a plain node
            if not ps:
                r.unrooted_categories.append(n["name"])
            elif ident_id and (len(ps) > 1 or (ps[0] != ident_id and nodes[ps[0]]["type"] != "category")):
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
    # a category with more direct non-category children than the sub-grouping
    # threshold is a flat list, not structure: say so, and name the cure
    child_count: dict[str, int] = {}
    for nid, ps in parents.items():
        if nodes[nid]["type"] != "category":
            for p in ps:
                child_count[p] = child_count.get(p, 0) + 1
    r.oversized = sorted(((nodes[p]["name"], c) for p, c in child_count.items()
                          if nodes[p]["type"] == "category" and c > oversized_threshold),
                         key=lambda x: -x[1])
    # the same fan-out under an event/org/project is worse: subgroup leaves non-categories
    # alone, and 17 "sponsor of X" organizations are one fact wearing 17 nodes
    r.flat_lists = sorted(((nodes[p]["name"], c) for p, c in child_count.items()
                           if nodes[p]["type"] != "category" and p != ident_id and c > oversized_threshold),
                          key=lambda x: -x[1])
    # the same thing under two names: two same-type nodes whose stored embeddings
    # all but coincide ("Current Semester Start" beside "SM Data Science Start"
    # scored 0.896 on Sep 6 2026 while every name check passed). Parent/child
    # pairs are structure, not duplicates; no model call — the vectors are stored.
    import json
    import math
    unit: dict[str, list[float]] = {}
    for nid, n in nodes.items():
        if n["type"] == "category" or not n["embedding"]:
            continue
        try:
            vec = json.loads(n["embedding"])
            norm = math.sqrt(sum(x * x for x in vec))
        except (TypeError, ValueError):
            continue
        if norm:
            unit[nid] = [x / norm for x in vec]
    by_type: dict[str, list[str]] = {}
    for nid in unit:
        by_type.setdefault(nodes[nid]["type"], []).append(nid)
    listed = {frozenset(p) for p in r.duplicates}
    for group in by_type.values():
        for i in range(len(group)):
            a = group[i]
            for j in range(i + 1, len(group)):
                b = group[j]
                if b in parents.get(a, []) or a in parents.get(b, []):
                    continue
                cos = sum(x * y for x, y in zip(unit[a], unit[b]))
                if cos >= SEMANTIC_DUP_MIN:
                    pair = (nodes[a]["name"], nodes[b]["name"])
                    r.semantic_duplicates.append((pair[0], pair[1], round(cos, 3)))
                    if frozenset(pair) not in listed:
                        r.duplicates.append(pair)
                        listed.add(frozenset(pair))
    r.semantic_duplicates.sort(key=lambda x: -x[2])
    # a fact is an attribute — a leaf by design ("also emit it as a fact node with
    # that entity as parent"); one with children is being used as a container
    # ("Alvin's computer" filed under the fact "Alvin's Residence (US)")
    kids_of: dict[str, list[str]] = {}
    for nid, ps in parents.items():
        for p in ps:
            kids_of.setdefault(p, []).append(nodes[nid]["name"])
    r.fact_parents = sorted(((nodes[p]["name"], sorted(cs)) for p, cs in kids_of.items()
                             if nodes[p]["type"] == "fact"), key=lambda x: x[0])
    # a top-level area with almost nothing under it, in a graph that has had time
    # to fill, is a sub-category that got rooted ("Family" beside "Relationships")
    # or an empty template area ("Health"): sprawl the planner is told to avoid
    if ident_id and sum(1 for n in nodes.values() if n["type"] != "category") >= THIN_AREA_MIN_GRAPH:
        children: dict[str, list[str]] = {}
        for nid, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(nid)

        def descendants(nid: str, seen: set) -> int:
            total = 0
            for c in children.get(nid, []):
                if c not in seen:
                    seen.add(c)
                    total += 1 + descendants(c, seen)
            return total
        r.thin_areas = sorted(((nodes[c]["name"], descendants(c, {c}))
                               for c in children.get(ident_id, []) if nodes[c]["type"] == "category"
                               and descendants(c, {c}) <= THIN_AREA_MAX),
                              key=lambda x: (x[1], x[0]))
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
    # two nodes of one type that `brain index` maps to the same vault file are,
    # by that file's own title + alias list, one entity — the name-ratio check
    # above misses pairs like "Miracle" / "Miracle Oy" (0.82), the vault does not
    by_file: dict[tuple, list] = {}
    for nid, n in nodes.items():
        path = n["path"] if "path" in n.keys() else None
        if path and n["type"] != "category":
            by_file.setdefault((path, n["type"]), []).append(nid)
    seen_pairs = {frozenset(p) for p in r.duplicates}
    for ids in by_file.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in parents.get(b, []) or b in parents.get(a, []):
                    continue  # a parent and its child are two things, however the vault files them
                pair = (nodes[a]["name"], nodes[b]["name"])
                if frozenset(pair) not in seen_pairs:
                    r.duplicates.append(pair)
                    seen_pairs.add(frozenset(pair))
    # the same name captured under two types ("AC 215" concept vs "AC215" event)
    # is one entity twice; the per-type pass never compares them, so match on the
    # space-free normalised name across all non-category nodes
    by_compact: dict[str, list[str]] = {}
    for n in nodes.values():
        if n["type"] != "category":
            key = _norm(n["name"]).replace(" ", "")
            if key:
                by_compact.setdefault(key, []).append(n["name"])
    for names in by_compact.values():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if frozenset((names[i], names[j])) not in seen_pairs:
                    r.duplicates.append((names[i], names[j]))
                    seen_pairs.add(frozenset((names[i], names[j])))
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
    - cycle: cut the part_of edge leaving the node with the larger subtree;
    - dangling edge (its source or target node was deleted): removed.
    Never deletes or renames nodes; duplicates and embeddings are reported, not fixed."""
    out = {"multi_parent": 0, "categories": 0, "under_identity": 0, "orphans": 0, "cycles": 0,
           "dangling": 0}
    db.ensure_identity_anchor(conn, user)
    ident = db.get_node_by_name(conn, user)["id"]
    # 0. edges to nodes that no longer exist
    out["dangling"] = conn.execute(f"DELETE FROM edges {_DANGLING_WHERE}").rowcount

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
    # 1. categories → the person, or one parent category (a sub-category)
    for nid, n in nodes.items():
        if n["type"] != "category" or nid == ident:
            continue
        edges = part_of_edges(nid)
        ok = (len(edges) == 1 and (edges[0]["target_id"] == ident
                                   or nodes.get(edges[0]["target_id"], {"type": ""})["type"] == "category"))
        if not ok:
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


# ── content integrity: claims that stopped being true ─────────────────────────
STALE_CLAIM_DAYS = 30
_PLAN_TENSE = re.compile(r"\b(plans? to|planning to|is currently|currently|intends? to|wants? to|"
                         r"is trying to|is considering|will be|upcoming|soon)\b", re.I)
_DATED = re.compile(r"\bas of\b", re.I)
_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_FUTURE_DATED = re.compile(r"\b(?:will|is going to|is scheduled to|is expected to)\b[^.;]{0,80}?\bon "
                           r"([A-Za-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?(?:,? (\d{4}))?", re.I)


def _passed_date(content: str, today) -> str | None:
    """'will begin classes on September 2' once September 2 has gone by — the
    phrase that dates a future claim, or None. A missing year is this year."""
    m = _FUTURE_DATED.search(content)
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if not month:
        return None
    try:
        when = date(int(m.group(3) or today.year), month, int(m.group(2)))
    except ValueError:
        return None
    return f"{m.group(0).strip()} (passed)" if when < today else None


def stale_claims(conn, days: int = STALE_CLAIM_DAYS, now: float | None = None) -> list[tuple[str, str, int]]:
    """Nodes whose content still speaks in plan or present tense ("plans to
    enroll", "is currently trying to") `days` after they were written. A
    backfill of old mail wrote "Alvin plans to relocate to Boston" and "plans
    to pursue studies" at Harvard three months before anyone read them, and
    ingest appends on re-mention, so the stale sentence keeps standing.
    Returns (name, phrase, age_days), oldest first. A claim dated with "as of"
    is deliberate and skipped — date a long-running plan and it stops nagging."""
    import time
    now = now or time.time()
    today = date.fromtimestamp(now)
    out = []
    for n in db.all_nodes(conn):
        if n["type"] == "category":
            continue
        content = n["content"] or ""
        age = int((now - n["created_at"]) // 86400)
        passed = _passed_date(content, today)       # a dated future claim is stale the day after, whatever its age
        if passed:
            out.append((n["name"], passed, age))
            continue
        if age < days or _DATED.search(content):
            continue
        m = _PLAN_TENSE.search(content)
        if m:
            out.append((n["name"], m.group(0), age))
    return sorted(out, key=lambda x: (-x[2], x[0]))

