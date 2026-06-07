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

Hierarchy — organise everything into a shallow tree rooted at the person:
- give every node a "parent": the broader node it belongs under.
- broad life-area "category" nodes (Career, Hobbies, Health, ...) have the
  person's name as their parent.
- a specific thing's parent is its category or a more specific node
  (e.g. parent of "Game on Sunday" is "Football"; parent of "Football" is "Hobbies").
- REUSE existing categories when one fits; don't invent near-duplicates.
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


def _attach_parents(conn, db, extracted: dict, name_to_id: dict, source: str, user: str):
    """Build the hierarchy spine: a `part_of` edge from each node to its parent.

    Unknown parents become `category` nodes (never-decay, high importance), and
    any category left without a parent is rooted at the person's identity node.
    Returns (new_node_ids, new_edge_ids).
    """
    new_nodes, new_edges = [], []
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
            if existing:
                parent_id = existing["id"]
            else:  # emergent grouping → a category node
                parent_id = db.add_node(conn, name=parent, type_="category",
                                        source=source, importance=0.9)
                name_to_id[parent] = parent_id
                new_nodes.append(parent_id)
        if parent_id and parent_id != child_id:
            new_edges.append(db.add_edge(conn, child_id, parent_id, "part_of"))

    # root any orphan category at the identity node
    identity = db.get_node_by_name(conn, user) if user else None
    if identity:
        for nid in set(name_to_id.values()):
            node = db.get_node(conn, nid)
            if not node or node["type"] != "category" or nid == identity["id"]:
                continue
            has_parent = any(
                e["source_id"] == nid and e["relation"] == "part_of"
                for e in db.edges_for_node(conn, nid)
            )
            if not has_parent:
                new_edges.append(db.add_edge(conn, nid, identity["id"], "part_of"))
    return new_nodes, new_edges


def merge_into_db(conn, extracted: dict, source: str, raw_text: str,
                  entity_links: dict | None = None, user: str = ""):
    """Write extracted nodes/edges into the DB, deduplicating by name and entity links."""
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
        existing = db.get_node_by_name(conn, canonical)
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
