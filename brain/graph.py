"""Graph traversal and context synthesis."""
import os
from collections import deque
from google import genai
from brain import db


def bfs(conn, start_ids: list[str], depth: int = 3, min_weight: float = 0.2) -> dict:
    """BFS from start_ids, returning {node_id: node_row} for all reachable nodes."""
    visited = {}
    queue = deque((nid, 0) for nid in start_ids)
    seen = set(start_ids)

    while queue:
        nid, d = queue.popleft()
        node = db.get_node(conn, nid)
        if not node or node["archived"] or node["weight"] < min_weight:
            continue
        visited[nid] = node

        if d < depth:
            for edge in db.edges_for_node(conn, nid):
                nbr = edge["target_id"] if edge["source_id"] == nid else edge["source_id"]
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append((nbr, d + 1))

    return visited


def synthesize_context(nodes: dict, topic: str = "") -> str:
    """Call Claude Sonnet to synthesise a context document from a node collection."""
    if not nodes:
        return "No relevant knowledge found."

    # build a compact node dump grouped by type
    by_type: dict[str, list] = {}
    for n in nodes.values():
        by_type.setdefault(n["type"], []).append(n)

    sections = []
    for t, items in sorted(by_type.items()):
        lines = [f"  - {i['name']} (w={i['weight']:.2f}): {i['content']}" for i in items]
        sections.append(f"{t.upper()}S\n" + "\n".join(lines))

    node_dump = "\n\n".join(sections)

    prompt = f"""Here is a personal knowledge graph{' about "' + topic + '"' if topic else ''}.

{node_dump}

Write a structured context document with sections:
## Background
## Active Skills
## Current Focus
## Projects
## Open Questions

Be concise and synthesise — don't just list facts. Write as if briefing someone who needs to understand this person's knowledge state quickly."""

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()


def query_nodes(conn, query: str, min_weight: float = 0.0) -> list:
    """Search nodes and touch them (reinforcing weight)."""
    results = db.search_nodes(conn, query, min_weight=min_weight)
    for r in results:
        db.touch_node(conn, r["id"])
    conn.commit()
    return results
