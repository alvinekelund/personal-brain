import json
import os
import google.generativeai as genai

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


def extract(text: str, source: str = "") -> dict:
    """Call Gemini Flash to extract nodes and edges from raw text."""
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM,
    )

    prompt = f"Extract knowledge from this text:\n\n{text[:4000]}"
    if source:
        prompt += f"\n\n(Source: {source})"

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def merge_into_db(conn, extracted: dict, source: str, raw_text: str):
    """Write extracted nodes/edges into the DB, deduplicating by name."""
    from brain import db

    name_to_id = {}
    node_ids = []
    edge_ids = []

    for n in extracted.get("nodes", []):
        existing = db.get_node_by_name(conn, n["name"])
        if existing:
            # touch it so it resets decay
            db.touch_node(conn, existing["id"])
            name_to_id[n["name"]] = existing["id"]
            node_ids.append(existing["id"])
        else:
            nid = db.add_node(
                conn,
                name=n["name"],
                type_=n.get("type", "concept"),
                content=n.get("content", ""),
                source=source,
                confidence=n.get("confidence", 0.8),
            )
            name_to_id[n["name"]] = nid
            node_ids.append(nid)

    for e in extracted.get("edges", []):
        src_id = name_to_id.get(e["source"])
        tgt_id = name_to_id.get(e["target"])
        if src_id and tgt_id:
            eid = db.add_edge(conn, src_id, tgt_id, e.get("relation", "relates_to"))
            edge_ids.append(eid)

    conn.commit()
    db.log_ingestion(conn, raw_text, source, node_ids, edge_ids)
    conn.commit()

    return node_ids, edge_ids
