"""Graph traversal and context synthesis."""
from collections import deque
from brain import db, llm


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


def collect_context_nodes(conn, topic: str = "", depth: int = 3, min_weight: float = 0.2):
    """Gather the node set for a context document.

    Returns (nodes_dict, used_fallback). When a topic is given, seed from the
    (stem-aware) keyword search and BFS-expand for connected context. If keyword
    search finds nothing, fall back to the whole high-weight brain — the
    synthesis step is topic-aware and focuses the document regardless, so this
    is reliable without a brittle LLM seed-picker.
    """
    used_fallback = False
    if topic:
        seeds = db.search_nodes(conn, topic, min_weight=min_weight)
        start_ids = [n["id"] for n in seeds]
        if not start_ids:
            used_fallback = True
            start_ids = [n["id"] for n in db.all_nodes(conn, min_weight=min_weight)]
    else:
        start_ids = [n["id"] for n in db.all_nodes(conn, min_weight=min_weight)]

    if not start_ids:
        return {}, used_fallback
    return bfs(conn, start_ids, depth=depth, min_weight=min_weight), used_fallback


def synthesize_context(nodes: dict, topic: str = "") -> str:
    """Call the LLM to synthesise a context document from a node collection."""
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

    return llm.generate(prompt).strip()


def query_nodes(conn, query: str, min_weight: float = 0.0) -> list:
    """Search nodes and touch them (reinforcing weight)."""
    results = db.search_nodes(conn, query, min_weight=min_weight)
    for r in results:
        db.touch_node(conn, r["id"])
    conn.commit()
    return results
