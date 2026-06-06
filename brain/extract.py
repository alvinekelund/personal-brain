import json
from brain import llm

SYSTEM = """You extract structured knowledge from text.
Return ONLY valid JSON with this exact shape:
{
  "nodes": [
    {"name": "...", "type": "...", "content": "...", "confidence": 0.0-1.0}
  ],
  "edges": [
    {"source": "...", "target": "...", "relation": "..."}
  ]
}

Node types — pick the most specific one that fits:
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
- extract only what is genuinely stated or implied; don't hallucinate
- prefer fewer high-confidence nodes over many uncertain ones
- prefer durable entities (concepts, skills, projects, people) over recording
  one-off actions. A completed action is best captured by what it produced —
  the concept learned, skill practised, or project advanced — not as its own node.
- never emit two nodes for the same underlying thing. A problem and the act of
  solving it are ONE node (keep the thing, e.g. "gradient explosion bug", and
  express the outcome via an edge or its content) — not "X" plus "fixing X"."""


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


def _extract_chunk(text: str, source: str = "", existing_names: list[str] | None = None, user: str = "") -> dict:
    """Call Gemini once on a single (already size-bounded) chunk of text."""
    prompt = f"Extract knowledge from this text:\n\n{text[:MAX_CHUNK]}"
    if source:
        prompt += f"\n\n(Source: {source})"
    if user:
        prompt += (
            f"\n\nIMPORTANT: The person whose brain this is, and the author of all "
            f"first-person statements ('I', 'me', 'my', 'the user', 'the speaker'), is: {user}. "
            f"Always use the name '{user}' for this person — never 'User', 'Speaker', 'Me', etc."
        )
    if existing_names:
        prompt += (
            f"\n\nExisting nodes already in the graph — reuse these exact names "
            f"if the text refers to the same entity:\n"
            + ", ".join(existing_names[:60])
        )

    result = _parse_json(llm.generate(prompt, system=SYSTEM, response_json=True))
    return result if isinstance(result, dict) else {"nodes": [], "edges": []}


def extract(text: str, source: str = "", existing_names: list[str] | None = None, user: str = "") -> dict:
    """Extract nodes/edges from text. Long input is chunked and merged so nothing
    past the model's input window is dropped."""
    chunks = _chunk_text(text)
    if not chunks:
        return {"nodes": [], "edges": []}
    if len(chunks) == 1:
        return _extract_chunk(chunks[0], source, existing_names, user)

    merged: dict = {"nodes": [], "edges": []}
    seen: set[str] = set()
    for chunk in chunks:
        part = _extract_chunk(chunk, source, existing_names, user)
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


def merge_into_db(conn, extracted: dict, source: str, raw_text: str, entity_links: dict | None = None):
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

    conn.commit()
    db.log_ingestion(conn, raw_text, source, node_ids, edge_ids)
    conn.commit()

    return node_ids, edge_ids
