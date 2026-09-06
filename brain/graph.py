"""Graph traversal and context synthesis."""
from pathlib import Path
import json
import math
from collections import deque
from brain import db, decay, llm


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


def synthesize_context(nodes: dict, topic: str = "", file_lines: list | None = None,
                       ledger_lines: list | None = None) -> str:
    """Call the LLM to synthesise a context document from a node collection,
    plus (D-014) excerpts of the vault files and ledger lines that match the
    topic — the graph alone briefed "training" without the race date."""
    if not nodes and not file_lines and not ledger_lines:
        return "No relevant knowledge found."

    def imp(n):
        return n["importance"] if "importance" in n.keys() else 0.5

    # build a compact node dump grouped by type, most-important first within each
    by_type: dict[str, list] = {}
    for n in nodes.values():
        by_type.setdefault(n["type"], []).append(n)

    sections = []
    for t, items in sorted(by_type.items()):
        items = sorted(items, key=lambda i: -imp(i))
        lines = [f"  - {i['name']} (importance {imp(i):.2f}): {i['content']}" for i in items]
        sections.append(f"{t.upper()}S\n" + "\n".join(lines))

    node_dump = "\n\n".join(sections) if sections else "(no graph nodes)"
    extra = ""
    if ledger_lines:
        extra += "\n\nLedgers (settled decisions and open loops — the most reliable facts):\n" + "\n".join(ledger_lines)
    if file_lines:
        extra += ("\n\nFiles (the vault is the source of truth; where a file and the graph disagree, "
                  "the file wins; dates and numbers come from here):\n" + "\n".join(file_lines))

    prompt = f"""Here is a personal knowledge graph{' about "' + topic + '"' if topic else ''}.
Each item has an importance (0-1): how central and lasting it is to the person.

{node_dump}{extra}

Write a structured context document with sections:
## Background
## Active Skills
## Current Focus
## Projects
## Open Questions

Lead with the highest-importance items and weight them most; mention low-importance
details only briefly or omit them. Be concise and synthesise — don't just list
facts. Write as if briefing someone who needs to understand this person quickly."""

    return llm.generate(prompt).strip()


def context_material(conn, topic: str, nodes: dict, root=None) -> tuple[list[str], list[str]]:
    """What a briefing needs beyond graph nodes (D-014): excerpts of the vault
    files and the ledger lines that match the topic. Empty topic → nothing
    (the whole-brain briefing has no query to route). Best-effort: never raises."""
    if not (topic or "").strip():
        return [], []
    qvec = None
    try:
        if llm.have_key():
            qvec = llm.embed(topic)
    except Exception:
        qvec = None
    seeds = list(nodes.values())[:40] if nodes else []
    try:
        _, file_lines = file_context(conn, topic, seeds, root, query_vector=qvec)
    except Exception:
        file_lines = []
    try:
        ledger_lines, _ = ledger_context(topic, root, conn=conn, query_vector=qvec)
    except Exception:
        ledger_lines = []
    return file_lines, ledger_lines


def children_map(conn) -> dict:
    """Return {parent_id: [child_ids]} from the part_of hierarchy edges."""
    m: dict = {}
    for e in db.all_edges(conn):
        if e["relation"] == "part_of":
            m.setdefault(e["target_id"], []).append(e["source_id"])
    return m


def open_loops(loops_root=None) -> list:
    """Open loops from the vault's LOOPS.md as display strings (the graph is
    context, not a to-do list — tasks live in the ledger). `loops_root` overrides
    the configured vault (tests); a missing ledger yields []."""
    from brain import config, loops
    root = Path(loops_root) if loops_root is not None else config.vault_dir()
    try:
        ledger = loops.load(root)
    except OSError:
        return []
    return [f"{l.title} (due {l.due.isoformat()}, {l.owner}) {l.id}"
            for l in sorted(ledger.open, key=lambda l: (l.prio, l.due, l.id))]


def digest(conn, user: str = "", top: int = 6, loops_root=None) -> dict:
    """A quick 'state of your brain': the highest-importance items, open loops
    (from LOOPS.md, not graph nodes), what's fading, and the life-area balance.
    Deterministic (no LLM)."""
    def imp(n):
        return n["importance"] if "importance" in n.keys() else 0.5

    # legacy task nodes (pre-Sep-2026 graphs) are ignored everywhere: tasks are loops
    nodes = [n for n in db.all_nodes(conn)
             if n["type"] not in ("category", "task") and n["name"].lower() != (user or "").lower()]
    top_nodes = sorted(nodes, key=lambda n: (-imp(n), -n["weight"]))[:top]
    return {
        "top": [{"name": n["name"], "type": n["type"], "importance": round(imp(n), 2)}
                for n in top_nodes],
        "tasks": open_loops(loops_root),
        "fading": [f for f in decay.at_risk_nodes(conn) if f.get("type") != "task"],
        "areas": category_breakdown(conn, user),
    }


def category_breakdown(conn, user: str = "") -> list:
    """Return [(category_name, descendant_count)] for top-level life-area
    categories (direct children of the person), largest first."""
    identity = db.get_node_by_name(conn, user) if user else None
    kids = children_map(conn)

    def subtree(nid, seen):
        seen.add(nid)
        return 1 + sum(subtree(c, seen) for c in kids.get(nid, []) if c not in seen)

    result = []
    for n in db.all_nodes(conn):
        if n["type"] != "category":
            continue
        parents = [e["target_id"] for e in db.edges_for_node(conn, n["id"])
                   if e["source_id"] == n["id"] and e["relation"] == "part_of"]
        top_level = (identity and identity["id"] in parents) or not parents
        if top_level:
            result.append((n["name"], subtree(n["id"], set()) - 1))
    return sorted(result, key=lambda x: -x[1])


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


def answer_question(conn, question: str, k: int = 8, min_weight: float = 0.0,
                    history: list | None = None, ledger_root=None) -> dict:
    """Answer a natural-language question from the brain: retrieve the most
    relevant nodes (semantic, falling back to keyword) and have the LLM answer
    using only them. Accessing them reinforces them. Returns {answer, sources}.

    history (optional [{q, a}, ...]) makes it conversational — the prior turn is
    folded into retrieval and the prompt so follow-ups ("...and where?") resolve.
    """
    from brain import index as _index
    history = history or []
    retrieval_q = (history[-1].get("q", "") + " " + question).strip() if history else question
    who = bool(_index._WHO_RE.search(retrieval_q.lower()))

    def prefer_people(rows):
        # a who-question is answered from people: person nodes seed first, so
        # their files get the seed bonus and they appear in the graph context
        if not who:
            return rows[:k]
        people = [r for r in rows if r["type"] == "person"]
        others = [r for r in rows if r["type"] != "person"]
        return (people[:k] + others)[:k]

    seeds, qvec = [], None
    try:
        if llm.have_key():
            qvec = llm.embed(retrieval_q)
            pool = [r for _, r in semantic_search(conn, qvec, min_weight=min_weight,
                                                  limit=k * 4 if who else k)]
            seeds = prefer_people(pool)
    except Exception:
        seeds, qvec = [], None
    if not seeds:
        seeds = prefer_people(db.search_nodes(conn, retrieval_q, min_weight=min_weight)[:k * 4 if who else k])
    ledger_lines, ledger_sources = ledger_context(retrieval_q, ledger_root, conn=conn, query_vector=qvec)
    # D-014: the vault directory is the source of truth — route the question to its files
    files, file_lines = file_context(conn, retrieval_q, seeds, ledger_root, query_vector=qvec)
    if not seeds and not ledger_lines and not files:
        return {"answer": "I don't have anything on that yet.", "sources": [], "files": []}

    # pull in 1-hop neighbors of the matches for connected context (capped)
    seed_ids = {n["id"] for n in seeds}
    neighbors: dict = {}
    for n in seeds:
        for e in db.edges_for_node(conn, n["id"]):
            oid = e["target_id"] if e["source_id"] == n["id"] else e["source_id"]
            if oid not in seed_ids and oid not in neighbors:
                o = db.get_node(conn, oid)
                if o and not o["archived"]:
                    neighbors[oid] = o
        if len(neighbors) >= 12:
            break

    def describe(n) -> str:
        # the tree is the brain's structure: say where a node is filed, so
        # "what sits under Career?" is answered from the spine, not guessed
        parent = next((e["target_id"] for e in db.edges_for_node(conn, n["id"])
                       if e["source_id"] == n["id"] and e["relation"] == "part_of"), None)
        p = db.get_node(conn, parent) if parent else None
        where = f", under {p['name']}" if p else ""
        line = f"- {n['name']} ({n['type']}{where}): {n['content'] or ''}"
        if n["type"] == "category":  # a category is its members: list them, capped
            kids = [db.get_node(conn, e["source_id"]) for e in db.edges_for_node(conn, n["id"])
                    if e["target_id"] == n["id"] and e["relation"] == "part_of"]
            names = sorted(k["name"] for k in kids if k and not k["archived"])
            if names:
                shown = ", ".join(names[:20]) + (f", … ({len(names)} in all)" if len(names) > 20 else "")
                line += f" Contains: {shown}."
        return line

    graph_lines = [describe(n) for n in seeds]
    if neighbors:
        graph_lines.append("Related:")
        graph_lines += [describe(o) for o in list(neighbors.values())[:12]]
    lines = list(ledger_lines)
    if file_lines:
        lines += ["Files (the vault is the source of truth; cite a file by its path):"] + file_lines
    if graph_lines:
        lines += ["Graph:"] + graph_lines
    convo = ""
    if history:
        convo = "Conversation so far:\n" + "\n".join(
            f"Q: {h.get('q', '')}\nA: {h.get('a', '')}" for h in history[-3:]) + "\n\n"
    prompt = (
        "Answer the question using ONLY the following knowledge about the person. "
        "If the answer isn't contained in it, say you don't have that. Be concise and direct. "
        "When a fact comes from a file, mention the file path.\n\n"
        f"{convo}Knowledge:\n{chr(10).join(lines)}\n\nQuestion: {question}"
    )
    for n in seeds:  # asking accesses these memories → reinforce them
        db.touch_node(conn, n["id"])
    conn.commit()
    return {"answer": llm.generate(prompt).strip(),
            "sources": ledger_sources + [f["path"] for f in files] + [n["name"] for n in seeds],
            "files": [f["path"] for f in files]}


def file_context(conn, query: str, seeds: list, root=None, query_vector=None, n: int = 6):
    """The vault files a question should be answered from (D-014), as
    (ranked file dicts, prompt lines with excerpts read from disk). Files linked
    to the retrieved graph nodes rank higher; a missing index yields nothing."""
    from brain import config, index
    root = Path(root) if root is not None else config.vault_dir()
    try:
        files = index.search(conn, query, k=n, seed_node_ids=[s["id"] for s in seeds],
                             query_vector=query_vector)
    except Exception:
        return [], []
    lines = []
    for f in files:
        # a log's opening is its oldest entry: lead with the matching lines, newest first
        ex = index.excerpt(root, f["path"], query, matches_first=(f.get("kind") == "log"))
        if ex:
            lines += [f"### {f['path']} ({f['title']})", ex]
    return files, lines


def ledger_context(query: str, root=None, conn=None, query_vector=None,
                   max_loops: int = 8, max_decisions: int = 6) -> tuple[list[str], list[str]]:
    """Decisions and loops matching the query, as prompt lines + source ids. The
    ledgers hold the most valuable facts (what was settled, what is pending), so
    every answer sees them first. Keyword hits come first; with a connection and
    a query vector, the closest embedded ledger lines (brain index) are added so
    a loop worded differently from the question still reaches the answer.
    Missing vault → nothing."""
    from brain import config, decisions, loops
    root = Path(root) if root is not None else config.vault_dir()
    lines, sources = [], []
    try:
        ds = decisions.search(root, query)
        ls = loops.search(root, query, include_closed=True)
    except Exception:
        return [], []
    if conn is not None and query_vector is not None:
        try:
            from brain import index as _index
            hits = _index.ledger_semantic(conn, query_vector)
            if hits:
                have = {d.id for d in ds} | {l.id for l in ls}
                ledger = loops.load(root)
                by_loop = {l.id: l for l in ledger.open + ledger.closed}
                by_dec = {d.id: d for d in decisions.load(root)[0]}
                for key, _cos in hits:
                    if key in have:
                        continue
                    if key in by_loop and len(ls) < max_loops:
                        ls.append(by_loop[key])
                    elif key in by_dec and len(ds) < max_decisions:
                        ds.append(by_dec[key])
        except Exception:
            pass
    if ds:
        lines.append("Decisions (settled — cite the id):")
        for d in ds:
            lines.append(f"- {d.id} ({d.date}) {d.title}: {d.decision} Why: {d.why}"
                         + (f" Revisit if: {d.revisit}" if d.revisit not in ("", "—") else ""))
            sources.append(d.id)
    if ls:
        lines.append("Loops (open unless marked done):")
        for l in ls:
            state = f"done {l.done}" if l.closed else f"due {l.due}, owner {l.owner}"
            lines.append(f"- {l.id} {l.title} ({state}): next {l.next}")
            sources.append(l.id)
    return lines, sources


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
