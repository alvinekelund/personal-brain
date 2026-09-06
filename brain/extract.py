import json
import re
import time
from brain import llm

# A node name that means nothing out of context — "New Project (Harvard)",
# "the meeting", "Unknown" — is noise in retrieval and a duplicate magnet. The
# prompt forbids them; this guard enforces it when the model slips.
_GENERIC_NOUN = (
    r"(?:projects?|tasks?|meetings?|events?|things?|items?|notes?|documents?|files?|"
    r"person|people|places?|organi[sz]ations?|org|company|courses?|plans?|ideas?|topics?|"
    r"updates?|stuff|misc|miscellaneous|unknown|untitled|n/a|tbd|none|null|entity|nodes?|others?)"
)
_VAGUE_NAME = re.compile(
    rf"^(?:(?:a|an|the|this|that|some|my|our|his|her|their|new|another|random|generic)\s+)*"
    rf"{_GENERIC_NOUN}(?:\s*\([^)]*\))?$",
    re.IGNORECASE,
)


# A node that is *just* a course code ("AC 215", "STAT211", "CS 2881R",
# "MIT 6.4212", "AC 215 course") — optionally with a school in front or the
# word course behind. The letters are upper-case (so "Fall 2026" is not a
# course) and a code without a space needs three of them (so the flight
# "AY1653" is not one). "MIT 6.4212 Petition Approval" does not match: an event.
_COURSE_NAME = re.compile(
    r"^(?:(?i:harvard|mit|aalto|stanford)\s+)?"
    r"(?:[A-Z]{2,5}\s\d{1,4}[A-Z]?|[A-Z]{3,5}\d{1,4}[A-Z]?|\d{1,2}\.[A-Z]?\d{3,4})"
    r"(?:\s+(?i:course|class|module))?$"
)


def is_vague_name(name) -> bool:
    """True for names like "New Project (Harvard)", "the meeting", "Unknown",
    "TBD" — a generic noun with at most determiners in front and an optional
    parenthetical behind. "New York area" or "Meeting with Heli" are fine."""
    return bool(_VAGUE_NAME.match((name or "").strip()))

# Progress + wall-clock control for one ingest (L-061). `brain add` points
# ON_STAGE at stderr so a scheduled task shows which stage a slow Gemini call is
# stuck in; the MCP server, ambient capture and the web view leave it None.
# Past INGEST_DEADLINE seconds the best-effort stages (entity-linking,
# embeddings) are skipped — the fact still lands in the graph, and
# `brain reindex` backfills embeddings later. Override with BRAIN_INGEST_DEADLINE.
ON_STAGE = None
INGEST_DEADLINE = 240.0


def _stage(msg: str, t0: float | None = None):
    """Report an ingest stage to the ON_STAGE hook, stamped with elapsed time."""
    if ON_STAGE is None:
        return
    if t0 is not None:
        msg = f"{msg}  [t+{time.monotonic() - t0:.0f}s]"
    try:
        ON_STAGE(msg)
    except Exception:
        pass

SYSTEM = """You extract structured knowledge from text.
Return ONLY valid JSON with this exact shape:
{
  "nodes": [
    {"name": "...", "type": "...", "content": "...", "confidence": 0.0-1.0, "importance": 0.0-1.0, "parent": "..."}
  ],
  "edges": [
    {"source": "...", "target": "...", "relation": "..."}
  ],
  "tasks": ["..."]
}

"tasks" is NOT part of the graph. It lists OPEN, not-yet-done action items the
person still needs to do, each as one short imperative sentence with enough
context to act on ("email Heli about the AM 207 petition", "book the SSN
appointment"). Never include an action already completed in the text. Empty
list if there are none. Do NOT also create a node for a task.

Node types — pick the most specific one that fits:
  category     — a broad life-area grouping (e.g. "Career", "Hobbies",
                 "Relationships", "Health", "Education") used to organise the tree
  person       — a human being
  organization — a company, university, institution, team
  concept      — an abstract idea, theory, or domain of knowledge (a course or
                 module is a concept; its start, deadline or exam is the event)
  skill        — a concrete capability someone has or is learning
  project      — an ongoing body of work with a goal
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
- names must be specific and self-explanatory out of context: "GCP project
  ac215", not "New Project"; "Heli Helskyaho", not "my boss"; "AC 215", not
  "the course". If the text gives no specific name, leave the node out.
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
  etc. directly to the person — it must go under a category.
  (e.g. a friend's parent is "Relationships" or "People", NOT the person.)
- a specific thing's parent is its category or a more specific node under one
  (parent of "Game on Sunday" is "Football"; parent of "Football" is "Hobbies").
- REUSE existing categories when one genuinely fits; don't invent near-duplicates.
- existing categories may be listed as "Area > Sub-category" (a sub-category under
  an area). Prefer the most specific existing one that fits and give its bare name
  as the parent ("Companies & Organizations", not "Career > Companies & Organizations").
  Create a NEW category only for a genuinely new life-area, never for something that
  fits an existing sub-category.
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
            + ", ".join(existing_names[:HINT_LIMIT])
        )

    result = _parse_json(llm.generate(prompt, system=SYSTEM, response_json=True))
    if not isinstance(result, dict):
        return {"nodes": [], "edges": [], "tasks": []}
    # drop nodes whose names mean nothing out of context; edges and parents that
    # point at them fall away downstream (no id to resolve)
    result["nodes"] = [n for n in (result.get("nodes") or [])
                       if isinstance(n, dict) and not is_vague_name(n.get("name"))]
    for n in result["nodes"]:
        # a course named by its code is a body of study, not a 7-day event —
        # the model keeps typing it "event" because a semester is time-bound
        if (n.get("type") or "").lower() == "event" and _COURSE_NAME.match((n.get("name") or "").strip()):
            n["type"] = "concept"
    return result


def extract(text: str, source: str = "", existing_names: list[str] | None = None,
            user: str = "", categories: list[str] | None = None) -> dict:
    """Extract nodes/edges from text. Long input is chunked and merged so nothing
    past the model's input window is dropped."""
    chunks = _chunk_text(text)
    if not chunks:
        return {"nodes": [], "edges": []}
    if len(chunks) == 1:
        return _extract_chunk(chunks[0], source, existing_names, user, categories)

    merged: dict = {"nodes": [], "edges": [], "tasks": []}
    seen: set[str] = set()
    for chunk in chunks:
        part = _extract_chunk(chunk, source, existing_names, user, categories)
        for n in part.get("nodes", []):
            key = (n.get("name") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged["nodes"].append(n)
        merged["edges"].extend(part.get("edges", []))
        merged["tasks"].extend(part.get("tasks", []) or [])
    return merged


def link_entities(new_nodes: list, existing_nodes: list) -> dict:
    """
    Second pass: ask Gemini which new nodes are the same entity as an existing one.
    Returns {new_name: existing_name} for confirmed matches.
    """
    if not new_nodes or not existing_nodes:
        return {}

    new_names = [n["name"] for n in new_nodes]
    existing_names = [n["name"] for n in existing_nodes[:HINT_LIMIT]]

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
    "project": "Projects", "event": "Events",
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
        if " > " in parent:  # the model echoed an "Area > Sub-category" label: the sub-category is the parent
            parent = parent.rsplit(" > ", 1)[-1].strip()
        if is_vague_name(parent):
            parent = ""  # never let "New Project" become a category; the fallback applies
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
            # one part_of parent per node: the planned parent replaces any other
            child_node = db.get_node(conn, child_id)
            for e in db.edges_for_node(conn, child_id):
                if (e["source_id"] == child_id and e["relation"] == "part_of" and e["target_id"] != parent_id
                        and not (child_node and child_node["type"] == "category")):
                    db.delete_edge(conn, e["id"])
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

    # 3. root every parentless category at the identity — ALL categories in the
    #    graph, not just the ones this batch touched: `brain reorganize` plans only
    #    non-category nodes, so a category detached by decay would otherwise never
    #    be re-rooted (which is exactly what happened in Sep 2026).
    for node in db.all_nodes(conn):
        nid = node["id"]
        if node["type"] != "category" or nid == identity["id"]:
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

    from concurrent.futures import ThreadPoolExecutor

    entity_links = entity_links or {}
    nodes_in = extracted.get("nodes", [])
    name_to_id = {}
    node_ids = []
    edge_ids = []

    # Pre-embed candidates in PARALLEL (semantic dedup) — the single biggest
    # ingest cost was doing one sequential embed per new node here.
    embeds_cache: dict = {}  # canonical.lower() -> vec
    if llm.have_key():
        cands = []
        for n in nodes_in:
            name = (n.get("name") or "").strip()
            if not name:
                continue
            canonical = entity_links.get(name, name)
            if db.get_node_by_name(conn, canonical):
                continue
            cands.append((canonical.lower(), f"{canonical}. {n.get('content', '')}"))
        if cands:
            def _fetch(item):
                key, text = item
                try:
                    return key, llm.embed(text)
                except Exception:
                    return key, None
            with ThreadPoolExecutor(max_workers=min(8, len(cands))) as ex:
                for key, vec in ex.map(_fetch, cands):
                    if vec:
                        embeds_cache[key] = vec

    # Build same-type embedding lookup ONCE (was re-scanned per candidate before).
    existing_embs_by_type: dict = {}
    for r in db.all_nodes(conn):
        emb = r["embedding"] if "embedding" in r.keys() else None
        if not emb:
            continue
        try:
            existing_embs_by_type.setdefault(r["type"], []).append((r["id"], json.loads(emb)))
        except (TypeError, ValueError):
            pass

    for n in nodes_in:
        # defensive: a malformed node (no name) shouldn't abort the whole ingest
        name = (n.get("name") or "").strip()
        if not name:
            continue
        # resolve through entity linker first, then fall back to name match
        canonical = entity_links.get(name, name)
        type_ = n.get("type", "concept")
        existing = db.get_node_by_name(conn, canonical)
        vec = embeds_cache.get(canonical.lower())

        if not existing and vec is not None:
            # semantic dedup: a same-type near-identical node counts as the same entity
            best_id, best = None, 0.0
            for nid, ev in existing_embs_by_type.get(type_, []):
                sim = _cosine(vec, ev)
                if sim > best:
                    best, best_id = sim, nid
            if best >= SEMANTIC_DEDUP_THRESHOLD:
                existing = db.get_node(conn, best_id)

        if existing:
            db.touch_node(conn, existing["id"])
            name_to_id[name] = existing["id"]
            name_to_id[canonical] = existing["id"]
            node_ids.append(existing["id"])
        else:
            nid = db.add_node(
                conn,
                name=canonical,
                type_=type_,
                content=n.get("content", ""),
                source=source,
                confidence=n.get("confidence", 0.8),
                importance=n.get("importance", 0.5),
            )
            if vec:  # reuse the embedding we just computed (skip re-embedding later)
                db.set_embedding(conn, nid, vec)
                # so later same-batch candidates can dedup against this one too
                existing_embs_by_type.setdefault(type_, []).append((nid, vec))
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


def embed_nodes(conn, node_ids, workers: int = 8, deadline: float | None = None) -> int:
    """Best-effort: compute & store embeddings for nodes that lack one, embedding
    in parallel (network-bound) for speed. DB writes stay on the calling thread.

    Embeddings are an optimization, not required for ingestion, so per-node
    failures are swallowed. `deadline` (a time.monotonic() value) shrinks each
    call's budget to the time left and skips calls once it has passed.
    Returns the count embedded.
    """
    from concurrent.futures import ThreadPoolExecutor
    from brain import db

    if not llm.have_key():
        return 0
    todo = []
    for nid in node_ids:
        node = db.get_node(conn, nid)
        if node and not node["embedding"]:
            todo.append((nid, f"{node['name']}. {node['content'] or ''}"))
    if not todo:
        return 0

    def fetch(item):
        nid, text = item
        budget = None
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                return nid, None
            budget = max(5.0, min(llm.EMBED_BUDGET, left))
        try:
            return nid, llm.embed(text, budget=budget)
        except Exception:
            return nid, None

    done = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
        for nid, vec in ex.map(fetch, todo):  # DB writes on the main thread
            if vec:
                db.set_embedding(conn, nid, vec)
                done += 1
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


def divert_tasks(extracted: dict) -> list[str]:
    """Pull every action item out of an extraction so none becomes a graph node.

    Takes the `tasks` list the prompt asks for AND any node the model still
    typed as "task" (belt and braces), drops edges that referenced those nodes,
    and returns the task texts. The graph is context; tasks belong in LOOPS.md.
    """
    tasks = [str(t).strip() for t in (extracted.get("tasks") or []) if str(t).strip()]
    keep, dropped = [], set()
    for n in extracted.get("nodes", []) or []:
        if (n.get("type") or "").strip().lower() == "task":
            name = (n.get("name") or "").strip()
            text = (n.get("content") or "").strip() or name   # content is the actionable sentence
            if text:
                tasks.append(text)
            if name:
                dropped.add(name)
        else:
            keep.append(n)
    extracted["nodes"] = keep
    extracted["edges"] = [e for e in extracted.get("edges", []) or []
                          if (e.get("source") or "").strip() not in dropped
                          and (e.get("target") or "").strip() not in dropped]
    extracted["tasks"] = []
    seen, out = set(), []
    for t in tasks:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def route_tasks(tasks: list[str], source: str = "", inbox_root=None) -> int:
    """Append diverted tasks to the vault's loop inbox. Best-effort: the graph
    write path must never fail because the vault is unavailable."""
    if not tasks or inbox_root is False:
        return 0
    try:
        from brain import config, loops
        root = inbox_root if inbox_root is not None else config.vault_dir()
        return loops.inbox_add(root, tasks, source=source)
    except Exception:
        return 0


def commit_vault_writes(source: str = "", inbox_root=None) -> bool:
    """Commit what an ingest wrote into the vault (LOOPS-INBOX.md and the
    generated views), scoped so it never sweeps up curated files someone is
    mid-editing. The CLI commits its own ledger writes; without this every
    ingest left the vault git-dirty until a session mopped it up. Best-effort:
    ingestion must never fail on a git problem."""
    if inbox_root is False:
        return False
    try:
        from brain import config, loops
        root = inbox_root if inbox_root is not None else config.vault_dir()
        src = " ".join((source or "").split())
        return loops.git_commit_paths(
            root, ["DIGEST.md", "graph", loops.INBOX_FILE, loops.INBOX_SEEN_FILE],
            f"ingest: refresh generated views ({src[:60]})" if src else "ingest: refresh generated views")
    except Exception:
        return False


def category_labels(conn) -> list[str]:
    """The existing categories as the extractor should see them: top-level areas
    by name, sub-categories as "Area > Sub-category" (so the model can file a
    node precisely and never re-creates a sub-category at the top level)."""
    from brain import db
    cats = {n["id"]: n for n in db.all_nodes(conn) if n["type"] == "category"}
    labels = []
    for cid, cat in cats.items():
        parent = next((e["target_id"] for e in db.edges_for_node(conn, cid)
                       if e["source_id"] == cid and e["relation"] == "part_of"), None)
        if parent in cats:
            labels.append((1, f"{cats[parent]['name']} > {cat['name']}"))
        else:
            labels.append((0, cat["name"]))
    return [name for _, name in sorted(labels)]


HINT_LIMIT = 80


def relevant_existing(conn, raw: str, existing: list, limit: int = HINT_LIMIT) -> list:
    """The existing nodes worth showing the extractor and the entity-linker for
    this text: keyword matches first, then semantic matches (with a key), then
    the most important nodes — capped. all_nodes() is insertion-ordered, so the
    old "first 60" hint showed the oldest nodes and hid the relevant ones,
    which is how duplicates of recent nodes kept appearing."""
    from brain import db
    order: list = []
    seen: set[str] = set()

    def take(rows):
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                order.append(r)

    try:
        take(db.search_nodes(conn, raw)[: limit // 2])
    except Exception:
        pass
    if llm.have_key():
        try:
            from brain import graph as _graph
            qvec = llm.embed(raw[:2000])
            take([r for _, r in _graph.semantic_search(conn, qvec, limit=limit // 2)])
        except Exception:
            pass
    by_importance = sorted(existing, key=lambda n: (-(n["importance"] or 0.0), -(n["weight"] or 0.0)))
    take(by_importance)
    return order[:limit]


def ingest(conn, raw: str, source: str = "", user: str = "", inbox_root=None,
           deadline_s: float | None = None):
    """Full ingestion pipeline shared by `brain add`, the MCP server, ambient
    capture and the web view: ensure identity → extract → divert tasks to the
    loop inbox → entity-link → merge (with hierarchy) → embed → refresh the
    markdown vault → commit the vault writes. Returns (node_ids, edge_ids).
    `inbox_root=False` disables task routing and the commit (tests); None means
    the configured vault. `deadline_s` (default BRAIN_INGEST_DEADLINE /
    INGEST_DEADLINE) is the wall-clock point after which the best-effort stages
    are skipped rather than started."""
    from brain import db, vault

    t0 = time.monotonic()
    if deadline_s is None:
        deadline_s = llm._env_float("BRAIN_INGEST_DEADLINE", INGEST_DEADLINE)
    deadline = t0 + deadline_s

    def late() -> bool:
        return time.monotonic() > deadline

    if user:
        db.ensure_identity_anchor(conn, user)
    existing = db.all_nodes(conn)
    categories = category_labels(conn)
    hint = relevant_existing(conn, raw, existing)
    _stage(f"extract: {len(raw)} chars, {len(existing)} existing node(s) ({len(hint)} shown), "
           f"{len(_chunk_text(raw))} chunk(s)", t0)
    ex = extract(raw, source=source, existing_names=[n["name"] for n in hint],
                 user=user, categories=categories)
    _stage(f"extracted {len(ex.get('nodes', []))} node(s), {len(ex.get('edges', []))} edge(s), "
           f"{len(ex.get('tasks', []) or [])} task(s)", t0)
    route_tasks(divert_tasks(ex), source=source, inbox_root=inbox_root)

    if late():
        _stage(f"entity-link: skipped, past the {deadline_s:.0f}s ingest deadline "
               f"(name + semantic dedup still apply)", t0)
        links = {}
    else:
        _stage("entity-link against the existing graph", t0)
        links = link_entities(ex.get("nodes", []), hint)

    _stage("merge into the graph (dedup + hierarchy)", t0)
    node_ids, edge_ids = merge_into_db(conn, ex, source, raw, entity_links=links, user=user)

    if late():
        _stage(f"embed: skipped, past the {deadline_s:.0f}s ingest deadline "
               f"— `brain reindex` backfills", t0)
    else:
        _stage(f"embed {len(node_ids)} node(s)", t0)
        embed_nodes(conn, node_ids, deadline=deadline)

    _stage("render the vault + commit", t0)
    vault.auto_render(conn, user)  # keep the markdown file layer in step with the graph
    commit_vault_writes(source=source, inbox_root=inbox_root)
    _stage(f"done: {len(node_ids)} node(s), {len(edge_ids)} edge(s)", t0)
    return node_ids, edge_ids
