import json
import os
from google import genai
from google.genai import types

SYSTEM = """You extract structured knowledge from text.
Return ONLY valid JSON with this exact shape:
{
  "nodes": [
    {"name": "...", "type": "concept|skill|project|person|fact|insight|event", "content": "...", "confidence": 0.0-1.0}
  ],
  "edges": [
    {"source": "...", "target": "...", "relation": "relates_to|builds_on|requires|contradicts|part_of|studied_by|created_by|used_in"}
  ]
}

Rules:
- names are short labels (2-5 words max)
- content is 1-3 sentences explaining the node
- edges use node names from the nodes list
- confidence reflects how clearly the text supports this extraction
- extract only what is genuinely stated or implied; don't hallucinate
- prefer fewer high-confidence nodes over many uncertain ones"""


def _client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def extract(text: str, source: str = "", existing_names: list[str] | None = None, user: str = "") -> dict:
    """Call Gemini Flash to extract nodes and edges from raw text."""
    client = _client()

    prompt = f"Extract knowledge from this text:\n\n{text[:4000]}"
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
    )
    return _parse_json(response.text)


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
        response = _client().models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return _parse_json(response.text)
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
        # resolve through entity linker first, then fall back to name match
        canonical = entity_links.get(n["name"], n["name"])
        existing = db.get_node_by_name(conn, canonical)
        if existing:
            db.touch_node(conn, existing["id"])
            name_to_id[n["name"]] = existing["id"]
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
            name_to_id[n["name"]] = nid
            name_to_id[canonical] = nid
            node_ids.append(nid)

    for e in extracted.get("edges", []):
        src_id = name_to_id.get(e["source"])
        tgt_id = name_to_id.get(e["target"])
        if src_id and tgt_id and src_id != tgt_id:
            eid = db.add_edge(conn, src_id, tgt_id, e.get("relation", "relates_to"))
            edge_ids.append(eid)

    conn.commit()
    db.log_ingestion(conn, raw_text, source, node_ids, edge_ids)
    conn.commit()

    return node_ids, edge_ids
