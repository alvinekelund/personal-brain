"""Graph traversal and context synthesis."""
import json
import math
from collections import deque
from brain import db, llm


def bfs(conn, start_ids: list[str], depth: int = 3, min_weight: float = 0.2,
        hub_degree: int = 8) -> dict:
    """BFS from start_ids, returning {node_id: node_row} for reachable nodes.

    A node whose degree exceeds hub_degree is included when reached but is NOT
    expanded through (unless it's a seed). Otherwise a single high-degree hub —
    the identity node connects to almost everything — would pull the entire graph
    into any topic query. Pass hub_degree<=0 to disable and do a plain BFS.
    """
    adjacency: dict = {}
    for e in db.all_edges(conn):
        adjacency.setdefault(e["source_id"], set()).add(e["target_id"])
        adjacency.setdefault(e["target_id"], set()).add(e["source_id"])

    starts = set(start_ids)
    visited = {}
    queue = deque((nid, 0) for nid in start_ids)
    seen = set(start_ids)

    while queue:
        nid, d = queue.popleft()
        node = db.get_node(conn, nid)
        if not node or node["archived"] or node["weight"] < min_weight:
            continue
        visited[nid] = node

        if d >= depth:
            continue
        # don't traverse THROUGH an incidental hub (seeds always expand)
        if hub_degree > 0 and nid not in starts and len(adjacency.get(nid, ())) > hub_degree:
            continue
        for nbr in adjacency.get(nid, ()):
            if nbr not in seen:
                seen.add(nbr)
                queue.append((nbr, d + 1))

    return visited


def hub_cap(conn, floor: int = 5, factor: float = 2.0) -> int:
    """A scale-relative degree above which a node is treated as a traversal hub.

    Absolute thresholds don't transfer (degree 8 is a hub in a 15-node graph,
    ordinary in a 500-node one), so key it off the mean degree with a floor.
    """
    degrees: dict = {}
    for e in db.all_edges(conn):
        degrees[e["source_id"]] = degrees.get(e["source_id"], 0) + 1
        degrees[e["target_id"]] = degrees.get(e["target_id"], 0) + 1
    if not degrees:
        return floor
    mean = sum(degrees.values()) / len(degrees)
    return max(floor, int(round(mean * factor)))


SEMANTIC_SEED_MIN_SIM = 0.4   # cosine floor for using a node as a context seed


def _semantic_seeds(conn, topic: str, min_weight: float) -> list:
    """Embedding-based seeds for a topic (deterministic cosine ranking). Returns
    node ids above the similarity floor, or [] if embeddings/key are unavailable."""
    try:
        if not llm.have_key():
            return []
        ranked = semantic_search(conn, llm.embed(topic), min_weight=min_weight, limit=6)
    except Exception:
        return []
    return [r["id"] for score, r in ranked if score >= SEMANTIC_SEED_MIN_SIM]


def collect_context_nodes(conn, topic: str = "", depth: int = 3, min_weight: float = 0.2):
    """Gather the node set for a context document.

    Returns (nodes_dict, used_fallback). For a topic: seed from (stem-aware)
    keyword search; if that misses, fall back to embedding-based semantic seeds
    (deterministic cosine); only if that also finds nothing do we dump the whole
    high-weight brain. Seeds are BFS-expanded for connected context.
    """
    used_fallback = False
    if topic:
        seeds = db.search_nodes(conn, topic, min_weight=min_weight)
        start_ids = [n["id"] for n in seeds]
        if not start_ids:
            start_ids = _semantic_seeds(conn, topic, min_weight)  # meaning-based fallback
        if not start_ids:
            used_fallback = True
            start_ids = [n["id"] for n in db.all_nodes(conn, min_weight=min_weight)]
    else:
        start_ids = [n["id"] for n in db.all_nodes(conn, min_weight=min_weight)]

    if not start_ids:
        return {}, used_fallback
    # only guard hubs for topic queries; a no-topic dump wants the whole brain
    cap = hub_cap(conn) if topic and not used_fallback else 0
    return bfs(conn, start_ids, depth=depth, min_weight=min_weight, hub_degree=cap), used_fallback


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


def children_map(conn) -> dict:
    """Return {parent_id: [child_ids]} from the part_of hierarchy edges."""
    m: dict = {}
    for e in db.all_edges(conn):
        if e["relation"] == "part_of":
            m.setdefault(e["target_id"], []).append(e["source_id"])
    return m


def cosine(a: list, b: list) -> float:
    """Cosine similarity between two vectors; 0.0 if either is empty/zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_search(conn, query_vector: list, min_weight: float = 0.0, limit: int = 10) -> list:
    """Rank embedded nodes by cosine similarity to query_vector.

    Returns [(score, node_row), ...] highest-first. Nodes without an embedding
    are skipped (run `brain reindex` to embed them).
    """
    scored = []
    for r in db.all_nodes(conn, min_weight=min_weight):
        emb = r["embedding"] if "embedding" in r.keys() else None
        if not emb:
            continue
        try:
            vec = json.loads(emb)
        except (TypeError, ValueError):
            continue
        scored.append((cosine(query_vector, vec), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def query_nodes(conn, query: str, min_weight: float = 0.0) -> list:
    """Search nodes and touch them (reinforcing weight)."""
    results = db.search_nodes(conn, query, min_weight=min_weight)
    for r in results:
        db.touch_node(conn, r["id"])
    conn.commit()
    return results


def connect_isolated_nodes(conn, max_connect: int = 5, min_weight: float = 0.3) -> list:
    """Find nodes with no edges and ask the LLM to connect each to another node.

    Adds the suggested edges and returns a list of
    {"source", "relation", "target"} dicts for the connections actually made.
    A bad/empty/unparseable suggestion for one node is skipped, not fatal.
    """
    nodes = db.all_nodes(conn, min_weight=min_weight)
    if len(nodes) < 2:
        return []

    connected = set()
    for e in db.all_edges(conn):
        connected.add(e["source_id"])
        connected.add(e["target_id"])
    isolated = [n for n in nodes if n["id"] not in connected]

    made = []
    for iso in isolated[:max_connect]:
        candidates = [n["name"] for n in nodes if n["id"] != iso["id"]][:30]
        if not candidates:
            continue
        prompt = (
            f'The node "{iso["name"]}" ({iso["content"]}) is isolated.\n'
            f'Existing nodes: {", ".join(candidates)}\n\n'
            f'Which existing node does "{iso["name"]}" most relate to, and how? '
            f'Pick the relation from: {", ".join(db.RELATIONS)}.\n'
            f'Reply as JSON: {{"target": "node name", "relation": "relation_label"}} '
            f'or null if none.'
        )
        try:
            suggestion = llm.parse_json(llm.generate(prompt, response_json=True))
        except Exception:
            continue
        if not isinstance(suggestion, dict) or not suggestion.get("target"):
            continue
        target = db.get_node_by_name(conn, suggestion["target"])
        if not target or target["id"] == iso["id"]:
            continue
        relation = suggestion.get("relation") or "relates_to"
        db.add_edge(conn, iso["id"], target["id"], relation)
        made.append({"source": iso["name"], "relation": relation, "target": target["name"]})

    conn.commit()
    return made
