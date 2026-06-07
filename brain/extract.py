import json
from brain import llm

SYSTEM = """You extract structured knowledge from text.
Return ONLY valid JSON with this exact shape:
{
  "nodes": [
    {"name": "...", "type": "...", "content": "...", "confidence": 0.0-1.0, "importance": 0.0-1.0, "parent": "..."}
  ],
  "edges": [
    {"source": "...", "target": "...", "relation": "..."}
  ]
}

Node types — pick the most specific one that fits:
  category     — a broad life-area grouping (e.g. "Career", "Hobbies",
                 "Relationships", "Health", "Education") used to organise the tree
  person       — a human being
  organization — a company, university, institution, team
  concept      — an abstract idea, theory, or domain of knowledge
  skill        — a concrete capability someone has or is learning
  project      — an ongoing body of work with a goal
  task         — an OPEN, not-yet-done action the person still needs to do
                 (e.g. "email Heli"). NEVER create a task for an action already
                 completed in the text.
  artifact     — a document, slide deck, codebase, file, or physical object
  fact         — a specific true claim or data point
  insight      — a synthesised understanding or non-obvious conclusion
  event        — a time-bound occurrence (meeting, deadline, semester)

Edge relations — pick the most specific one that fits:
  relates_to | builds_on | requires | contradicts | part_of |
  studied_by | created_by | used_in | assigned_to | attended_by |
  works_at | member_of | located_at

Rules:
- names are short labels (2-5 words max)
- content is 1-3 sentences explaining the node
- edges use node names from the nodes list
- confidence reflects how clearly the text supports this extraction
- importance reflects how central and lasting this is to the person:
    0.8-1.0 = identity, close relationships, core skills, long-term projects/goals
    0.4-0.7 = ongoing interests, general knowledge, active work
    0.1-0.3 = one-off events, errands, ephemeral details (these are meant to fade)
- extract only what is genuinely stated or implied; don't hallucinate
- prefer fewer high-confidence nodes over many uncertain ones
- prefer durable entities (concepts, skills, projects, people) over recording
  one-off actions. A completed action is best captured by what it produced —
  the concept learned, skill practised, or project advanced — not as its own node.
- never emit two nodes for the same underlying thing. A problem and the act of
  solving it are ONE node (keep the thing, e.g. "gradient explosion bug", and
  express the outcome via an edge or its content) — not "X" plus "fixing X".

Hierarchy — organise everything into a tree rooted at the person:
- give every node a "parent": the broader node it belongs under.
- the person's ONLY direct children are broad "category" nodes (Career, Hobbies,
  Relationships, Health, Education, ...). NEVER attach a person/concept/project/
  task/etc. directly to the person — it must go under a category.
  (e.g. a friend's parent is "Relationships" or "People", NOT the person.)
- a specific thing's parent is its category or a more specific node under one
  (parent of "Game on Sunday" is "Football"; parent of "Football" is "Hobbies").
- REUSE existing categories when one genuinely fits; don't invent near-duplicates.
- categories are DISTINCT, non-overlapping life-areas. Put each node in the single
  best-fitting one; if none fits well, create a more specific category rather than
  forcing it into a loosely-related one. Do NOT overload one category as a catch-all:
    * a coding skill / side project → "Skills" or "Projects" (NOT "Education")
    * a book, topic, or interest you follow → "Learning" or "Interests"
    * a job / employer / work → "Career"
    * studies / degree / university → "Education"
- "parent" only sets the backbone; still add cross-links between nodes via edges
  (e.g. a friend relates_to a hobby) — the result is a tree plus cross-edges."""


def _parse_json(raw: str) -> dict:
    return llm.parse_json(raw)


MAX_CHUNK = 4000


def _chunk_text(text: str, size: int = MAX_CHUNK) -> list[str]:
    """Split text into <=size pieces on paragraph boundaries (hard-splitting any
    single oversized paragraph). Long input is chunked rather than silently
    truncated, so ingesting an article or book chapter keeps all of it."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        if len(para) > size:
            if cur.strip():
                chunks.append(cur.strip())
                cur = ""
            for i in range(0, len(para), size):
                chunks.append(para[i:i + size].strip())
            continue
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def _extract_chunk(text: str, source: str = "", existing_names: list[str] | None = None,
                   user: str = "", categories: list[str] | None = None) -> dict:
    """Call Gemini once on a single (already size-bounded) chunk of text."""
    prompt = f"Extract knowledge from this text:\n\n{text[:MAX_CHUNK]}"
    if source:
        prompt += f"\n\n(Source: {source})"
    if user:
        prompt += (
            f"\n\nIMPORTANT: The person whose brain this is, and the author of all "
            f"first-person statements ('I', 'me', 'my', 'the user', 'the speaker'), is: {user}. "
            f"Always use the name '{user}' for this person — never 'User', 'Speaker', 'Me', etc. "
            f"Top-level category nodes should have '{user}' as their parent."
        )
    if categories:
        prompt += (
            "\n\nExisting categories — REUSE one of these as a node's parent when it "
            "fits, instead of inventing a near-duplicate:\n" + ", ".join(categories[:40])
        )
    if existing_names:
        prompt += (
            f"\n\nExisting nodes already in the graph — reuse these exact names "
            f"if the text refers to the same entity:\n"
            + ", ".join(existing_names[:60])
        )

    result = _parse_json(llm.generate(prompt, system=SYSTEM, response_json=True))
    return result if isinstance(result, dict) else {"nodes": [], "edges": []}


def extract(text: str, source: str = "", existing_names: list[str] | None = None,
            user: str = "", categories: list[str] | None = None) -> dict:
    """Extract nodes/edges from text. Long input is chunked and merged so nothing
    past the model's input window is dropped."""
    chunks = _chunk_text(text)
    if not chunks:
        return {"nodes": [], "edges": []}
    if len(chunks) == 1:
        return _extract_chunk(chunks[0], source, existing_names, user, categories)

    merged: dict = {"nodes": [], "edges": []}
    seen: set[str] = set()
    for chunk in chunks:
        part = _extract_chunk(chunk, source, existing_names, user, categories)
        for n in part.get("nodes", []):
            key = (n.get("name") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged["nodes"].append(n)
        merged["edges"].extend(part.get("edges", []))
    return merged


def link_entities(new_nodes: list, existing_nodes: list) -> dict:
    """
    Second pass: ask Gemini which new nodes are the same entity as an existing one.
    Returns {new_name: existing_name} for confirmed matches.
    """
    if not new_nodes or not existing_nodes:
        return {}

    new_names = [n["name"] for n in new_nodes]
    existing_names = [n["name"] for n in existing_nodes[:60]]

    prompt = (
        f"New entities just extracted from text: {json.dumps(new_names)}\n"
        f"Entities already in the knowledge graph: {json.dumps(existing_names)}\n\n"
        f"For each new entity, decide if it refers to the exact same real-world entity as one already in the graph.\n"
        f"Be conservative — only match when you are confident it is the same entity, not just related.\n"
        f'Return ONLY JSON: {{"new_name": "existing_name"}} for matches. Empty object {{}} if none match.'
    )

    try:
        return _parse_json(llm.generate(prompt, response_json=True))
    except Exception:
        return {}


# type → fallback category, used when a node would otherwise hang off the person
FALLBACK_CATEGORY = {
    "person": "Relationships", "organization": "Organizations", "skill": "Skills",
    "project": "Projects", "task": "Tasks", "event": "Events",
    "artifact": "Artifacts", "fact": "Knowledge", "insight": "Insights",
    "concept": "Knowledge",
}


def _attach_parents(conn, db, extracted: dict, name_to_id: dict, source: str, user: str):
    """Build the hierarchy spine and enforce that it's a real tree:

    1. wire each node to its parent via `part_of` (unknown parents become categories);
    2. enforce the person's only direct children are categories — re-route any
       non-category node hanging directly off the person (or orphaned) under a
       type-based fallback category;
    3. root every category at the identity node.
    Returns (new_node_ids, new_edge_ids).
    """
    new_nodes, new_edges = [], []
    identity = db.get_node_by_name(conn, user) if user else None

    def ensure_category(name):
        cat = db.get_node_by_name(conn, name)
        if cat:
            cid = cat["id"]
        else:
            cid = db.add_node(conn, name=name, type_="category", source=source, importance=0.85)
            name_to_id[name] = cid
            new_nodes.append(cid)
        return cid

    # 1. explicit parents
    for n in extracted.get("nodes", []):
        child = (n.get("name") or "").strip()
        parent = (n.get("parent") or "").strip()
        if not child or not parent or parent.lower() == child.lower():
            continue
        child_id = name_to_id.get(child)
        if not child_id:
            continue
        parent_id = name_to_id.get(parent)
        if not parent_id:
            existing = db.get_node_by_name(conn, parent)
            parent_id = existing["id"] if existing else ensure_category(parent)
        if parent_id and parent_id != child_id:
            new_edges.append(db.add_edge(conn, child_id, parent_id, "part_of"))

    if not identity:
        return new_nodes, new_edges

    # 2. enforce person → categories only (re-route direct-to-person / orphan nodes)
    for n in extracted.get("nodes", []):
        nid = name_to_id.get((n.get("name") or "").strip())
        if not nid:
            continue
        node = db.get_node(conn, nid)
        if not node or node["type"] == "category" or nid == identity["id"]:
            continue
        part_edges = [e for e in db.edges_for_node(conn, nid)
                      if e["source_id"] == nid and e["relation"] == "part_of"]
        # the person cannot be a direct parent of a non-category node
        for e in part_edges:
            if e["target_id"] == identity["id"]:
                db.delete_edge(conn, e["id"])
        non_person_parents = [e for e in part_edges if e["target_id"] != identity["id"]]
        if not non_person_parents:  # orphan or was only under the person → give it a category
            cat_id = ensure_category(FALLBACK_CATEGORY.get(node["type"], "Misc"))
            if cat_id != nid:
                new_edges.append(db.add_edge(conn, nid, cat_id, "part_of"))

    # 3. root every category at the identity
    for nid in set(name_to_id.values()):
        node = db.get_node(conn, nid)
        if not node or node["type"] != "category" or nid == identity["id"]:
            continue
        if not any(e["source_id"] == nid and e["relation"] == "part_of"
                   for e in db.edges_for_node(conn, nid)):
            new_edges.append(db.add_edge(conn, nid, identity["id"], "part_of"))
    return new_nodes, new_edges


def plan_hierarchy(nodes: list, user: str, categories: list | None = None) -> list:
    """Ask the LLM to place existing (flat) nodes into the person-rooted tree.

    Returns [{name, parent, importance}] for retrofitting an old graph. Used by
    `brain reorganize`. Best-effort: returns [] on any failure.
    """
    if not nodes:
        return []
    catalog = [{"name": n["name"], "type": n["type"]} for n in nodes]
    prompt = (
        f"The person is {user}. Organise these existing knowledge-graph nodes into a "
        f"tree rooted at {user}. For each node give a 'parent' (the broader node or "
        f"category it belongs under) and an 'importance' (0.0-1.0).\n"
        f"- {user}'s ONLY direct children are broad category nodes (Career, Hobbies, "
        f"Relationships, Health, Education, ...); never attach a node directly to {user}.\n"
        f"- reuse these existing categories when they fit: "
        f"{', '.join(categories or []) or '(none yet)'}\n"
        f"- importance: identity/close people/core skills/long-term → 0.8-1.0; "
        f"general interests → 0.4-0.7; one-off details → 0.1-0.3.\n\n"
        f"Nodes:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
        f'Return ONLY JSON: {{"nodes": [{{"name": "...", "parent": "...", "importance": 0.0-1.0}}]}}'
    )
    try:
        data = llm.parse_json(llm.generate(prompt, response_json=True))
        return data.get("nodes", []) if isinstance(data, dict) else []
    except Exception:
        return []


SEMANTIC_DEDUP_THRESHOLD = 0.90   # cosine above which a same-type node is "the same"


def _cosine(a, b):
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_and_find_dupe(conn, db, name, content, type_):
    """Embed a candidate node and look for a near-identical existing node of the
    same type (catches paraphrases the name-based linker misses). Returns
    (existing_id_or_None, vector). No-ops (returns (None, None)) without a key."""
    if not llm.have_key():
        return None, None
    try:
        vec = llm.embed(f"{name}. {content}")
    except Exception:
        return None, None
    best_id, best = None, 0.0
    for r in db.all_nodes(conn):
        emb = r["embedding"] if "embedding" in r.keys() else None
        if r["type"] != type_ or not emb:
            continue
        try:
            sim = _cosine(vec, json.loads(emb))
        except (TypeError, ValueError):
            continue
        if sim > best:
            best, best_id = sim, r["id"]
    return (best_id if best >= SEMANTIC_DEDUP_THRESHOLD else None), vec


def merge_into_db(conn, extracted: dict, source: str, raw_text: str,
                  entity_links: dict | None = None, user: str = ""):
    """Write extracted nodes/edges into the DB, deduplicating by name, entity
    links, and semantic similarity (same-type, high cosine)."""
    from brain import db

    entity_links = entity_links or {}
    name_to_id = {}
    node_ids = []
    edge_ids = []

    for n in extracted.get("nodes", []):
        # defensive: a malformed node (no name) shouldn't abort the whole ingest
        name = (n.get("name") or "").strip()
        if not name:
            continue
        # resolve through entity linker first, then fall back to name match
        canonical = entity_links.get(name, name)
        vec = None
        existing = db.get_node_by_name(conn, canonical)
        if not existing:
            # semantic dedup: a same-type near-identical node counts as the same entity
            dupe_id, vec = _embed_and_find_dupe(
                conn, db, canonical, n.get("content", ""), n.get("type", "concept"))
            if dupe_id:
                existing = db.get_node(conn, dupe_id)
        if existing:
            db.touch_node(conn, existing["id"])
            name_to_id[name] = existing["id"]
            name_to_id[canonical] = existing["id"]
            node_ids.append(existing["id"])
        else:
            nid = db.add_node(
                conn,
                name=canonical,
                type_=n.get("type", "concept"),
                content=n.get("content", ""),
                source=source,
                confidence=n.get("confidence", 0.8),
                importance=n.get("importance", 0.5),
            )
            if vec:  # reuse the embedding we just computed (skip re-embedding later)
                db.set_embedding(conn, nid, vec)
            name_to_id[name] = nid
            name_to_id[canonical] = nid
            node_ids.append(nid)

    for e in extracted.get("edges", []):
        src_id = name_to_id.get((e.get("source") or "").strip())
        tgt_id = name_to_id.get((e.get("target") or "").strip())
        if src_id and tgt_id and src_id != tgt_id:
            eid = db.add_edge(conn, src_id, tgt_id, e.get("relation", "relates_to"))
            edge_ids.append(eid)

    # hierarchy spine (parent → part_of edges, emergent category nodes, root at user)
    h_nodes, h_edges = _attach_parents(conn, db, extracted, name_to_id, source, user)
    node_ids.extend(h_nodes)
    edge_ids.extend(h_edges)

    conn.commit()
    db.log_ingestion(conn, raw_text, source, node_ids, edge_ids)
    conn.commit()

    return node_ids, edge_ids


def embed_nodes(conn, node_ids) -> int:
    """Best-effort: compute & store embeddings for nodes that lack one.

    Lets semantic search work right after `brain add` without a manual reindex.
    Embeddings are an optimization, not required for ingestion, so per-node
    failures (offline, API error) are swallowed. Returns the count embedded.
    """
    from brain import db

    if not llm.have_key():
        return 0
    done = 0
    for nid in node_ids:
        node = db.get_node(conn, nid)
        if not node or node["embedding"]:
            continue
        try:
            db.set_embedding(conn, nid, llm.embed(f"{node['name']}. {node['content'] or ''}"))
            done += 1
        except Exception:
            continue
    if done:
        conn.commit()
    return done


def reorganize(conn, user: str):
    """Retrofit existing flat nodes into the person-rooted hierarchy and re-score
    importance (LLM plans parents). Returns (hierarchy_edges, importance_updates).
    Shared by `brain reorganize` and the web view."""
    from brain import db

    if user:
        db.ensure_identity_anchor(conn, user)
    nodes = [n for n in db.all_nodes(conn)
             if n["type"] != "category" and n["name"].lower() != (user or "").lower()]
    if not nodes:
        return (0, 0)
    categories = [n["name"] for n in db.all_nodes(conn) if n["type"] == "category"]
    plan = plan_hierarchy(nodes, user, categories)
    existing = {n["name"].strip().lower() for n in nodes}
    plan = [p for p in plan if (p.get("name") or "").strip().lower() in existing]
    if not plan:
        return (0, 0)
    _, edge_ids = merge_into_db(conn, {"nodes": plan, "edges": []}, "reorganize", "", user=user)
    name_to_id = {n["name"].lower(): n["id"] for n in db.all_nodes(conn)}
    rescored = 0
    for item in plan:
        nid = name_to_id.get((item.get("name") or "").strip().lower())
        if nid and item.get("importance") is not None:
            try:
                conn.execute("UPDATE nodes SET importance=? WHERE id=?",
                             (float(item["importance"]), nid))
                rescored += 1
            except (TypeError, ValueError):
                pass
    conn.commit()
    subgroup_categories(conn)  # split any bloated category into sub-categories
    return (len(edge_ids), rescored)


SUBGROUP_THRESHOLD = 12   # categories with more direct children than this get split


def _subgroup_one(conn, db, cat, children):
    names = [c["name"] for c in children]
    prompt = (
        f'The category "{cat["name"]}" has too many items:\n{json.dumps(names, ensure_ascii=False)}\n\n'
        f'Cluster them into 2-5 coherent sub-categories; each item goes in exactly one. '
        f'Return ONLY JSON: {{"groups": [{{"name": "sub-category", "members": ["item", ...]}}]}}'
    )
    try:
        data = llm.parse_json(llm.generate(prompt, response_json=True))
    except Exception:
        return 0
    groups = data.get("groups", []) if isinstance(data, dict) else []
    by_name = {c["name"].lower(): c for c in children}
    moved = 0
    for g in groups:
        gname = (g.get("name") or "").strip()
        members = [by_name.get((m or "").strip().lower()) for m in g.get("members", [])]
        members = [m for m in members if m]
        if not gname or gname.lower() == cat["name"].lower() or len(members) < 2:
            continue
        existing = db.get_node_by_name(conn, gname)
        sub_id = existing["id"] if existing else db.add_node(
            conn, name=gname, type_="category", source="subgroup", importance=0.8)
        db.add_edge(conn, sub_id, cat["id"], "part_of")  # sub-category under the category
        for m in members:
            for e in db.edges_for_node(conn, m["id"]):
                if (e["source_id"] == m["id"] and e["target_id"] == cat["id"]
                        and e["relation"] == "part_of"):
                    db.delete_edge(conn, e["id"])
            db.add_edge(conn, m["id"], sub_id, "part_of")
            moved += 1
    return moved


def subgroup_categories(conn, threshold: int = SUBGROUP_THRESHOLD) -> int:
    """Split oversized categories into LLM-clustered sub-categories, so a big area
    gets real sub-structure instead of a flat list. Returns nodes re-parented.
    No-ops without an API key (keeps tests hermetic)."""
    from brain import db

    if not llm.have_key():
        return 0
    moved = 0
    for cat in [n for n in db.all_nodes(conn) if n["type"] == "category"]:
        child_edges = [e for e in db.edges_for_node(conn, cat["id"])
                       if e["target_id"] == cat["id"] and e["relation"] == "part_of"]
        children = [db.get_node(conn, e["source_id"]) for e in child_edges]
        children = [c for c in children if c and c["type"] != "category"]  # leave sub-cats alone
        if len(children) > threshold:
            moved += _subgroup_one(conn, db, cat, children)
    if moved:
        conn.commit()
    return moved


def ingest(conn, raw: str, source: str = "", user: str = ""):
    """Full ingestion pipeline shared by `brain add` and the web view:
    ensure identity → extract → entity-link → merge (with hierarchy) → embed.
    Returns (node_ids, edge_ids)."""
    from brain import db

    if user:
        db.ensure_identity_anchor(conn, user)
    existing = db.all_nodes(conn)
    categories = [n["name"] for n in existing if n["type"] == "category"]
    ex = extract(raw, source=source, existing_names=[n["name"] for n in existing],
                 user=user, categories=categories)
    links = link_entities(ex.get("nodes", []), existing)
    node_ids, edge_ids = merge_into_db(conn, ex, source, raw, entity_links=links, user=user)
    embed_nodes(conn, node_ids)
    return node_ids, edge_ids
