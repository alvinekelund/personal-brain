#!/usr/bin/env python3
"""Generate a large synthetic life-log and ingest it to see how the brain reacts
at scale — category spread, tree depth, hubs, decay. Live Gemini, isolated temp
brain (your real brain is untouched).

    python scripts/generate_dataset.py [N]      # default N=40 ingests

The corpus is generated combinatorially (hundreds of unique statements), so you
can crank N as high as you like — each ingest is one extraction call.
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain.db as db
import brain.extract as extract
import brain.graph as graph
import brain.llm as llm

random.seed(42)
USER = "Alvin"

PEOPLE = ["Bjorn", "Heli", "Astrid", "Sara", "Mikael", "Nina", "Omar", "Priya", "Tom", "Lena"]
SPORTS = ["football", "bouldering", "tennis", "running", "swimming", "cycling", "skiing"]
DAYS = ["Mondays", "Tuesdays", "weekends", "Friday evenings", "Sunday mornings"]
BOOKS = ["Thinking Fast and Slow", "Sapiens", "The Pragmatic Programmer", "Dune", "Antifragile"]
TOPICS = ["cognitive biases", "history", "systems design", "stoicism", "complexity"]
PROJECTS = ["a log parser", "a recommender system", "a personal website", "a trading bot", "a chat app"]
TECH = ["Rust", "Python", "TypeScript", "Go", "PyTorch", "React"]
SKILLS = ["the piano", "Spanish", "public speaking", "sketching", "chess", "cooking Thai food"]
PLACES = ["Lund", "Berlin", "Tokyo", "Lisbon", "the Alps", "Helsinki"]
COURSES = ["reinforcement learning", "distributed systems", "linear algebra", "macroeconomics", "NLP"]
ORGS = ["Aalto University", "Harvard", "Siemens", "a startup called Vexa", "the student union"]

TEMPLATES = [
    lambda: f"I play {random.choice(SPORTS)} with my friend {random.choice(PEOPLE)} on {random.choice(DAYS)}.",
    lambda: f"I'm reading '{random.choice(BOOKS)}' and fascinated by {random.choice(TOPICS)}.",
    lambda: f"I'm building {random.choice(PROJECTS)} in {random.choice(TECH)} this semester.",
    lambda: f"I started learning {random.choice(SKILLS)}.",
    lambda: f"I took a trip to {random.choice(PLACES)} with {random.choice(PEOPLE)}.",
    lambda: f"I'm taking a course on {random.choice(COURSES)} at {random.choice(ORGS)}.",
    lambda: f"I had a long conversation with {random.choice(PEOPLE)} about {random.choice(TOPICS)}.",
    lambda: f"I need to finish my {random.choice(COURSES)} assignment by {random.choice(DAYS)}.",
    lambda: f"My colleague {random.choice(PEOPLE)} at {random.choice(ORGS)} helped me with {random.choice(TECH)}.",
    lambda: f"I went {random.choice(SPORTS)} in {random.choice(PLACES)} last month.",
    lambda: f"I'm mentoring {random.choice(PEOPLE)} on {random.choice(SKILLS)}.",
    lambda: f"I gave a talk on {random.choice(TOPICS)} at {random.choice(ORGS)}.",
]


def corpus(n):
    seen, out = set(), []
    guard = 0
    while len(out) < n and guard < n * 50:
        s = random.choice(TEMPLATES)()
        guard += 1
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def report(conn):
    nodes = db.all_nodes(conn, include_archived=True)
    edges = db.all_edges(conn)
    from collections import Counter
    by_type = Counter(n["type"] for n in nodes)
    kids = graph.children_map(conn)

    # category sizes (direct children count)
    cats = [n for n in nodes if n["type"] == "category"]
    cat_sizes = sorted(((len(kids.get(c["id"], [])), c["name"]) for c in cats), reverse=True)

    # degree (hubs)
    deg = Counter()
    for e in edges:
        deg[e["source_id"]] += 1
        deg[e["target_id"]] += 1
    top_hubs = sorted(((d, db.get_node(conn, nid)) for nid, d in deg.items()
                       if db.get_node(conn, nid)), key=lambda x: -x[0])[:6]

    # depth of the tree from the identity
    root = db.get_node_by_name(conn, USER)
    maxd = [0]
    seen = set()

    def depth(nid, d):
        if nid in seen:
            return
        seen.add(nid)
        maxd[0] = max(maxd[0], d)
        for c in kids.get(nid, []):
            depth(c, d + 1)
    if root:
        depth(root["id"], 0)

    print("\n" + "=" * 64, "\nBRAIN REACTION\n" + "=" * 64)
    print(f"nodes: {len(nodes)}  edges: {len(edges)}  categories: {len(cats)}  tree depth: {maxd[0]}")
    print(f"by type: {dict(by_type)}")
    print("\ncategory sizes (direct children):")
    for sz, name in cat_sizes:
        print(f"  {sz:3d}  {name}")
    print("\ntop hubs (degree):")
    for d, n in top_hubs:
        print(f"  {d:3d}  {n['name']} [{n['type']}]")


def main():
    if not llm.have_key():
        print("No GEMINI_API_KEY — set one in ~/.personal-brain/.env"); sys.exit(1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    db.DB_PATH = os.path.join(tempfile.mkdtemp(), "brain.db")
    conn = db.connect()
    db.ensure_identity_anchor(conn, USER)

    statements = corpus(n)
    print(f"Ingesting {len(statements)} synthetic statements (isolated temp brain)...")
    for i, s in enumerate(statements, 1):
        try:
            extract.ingest(conn, s, source="synthetic", user=USER)
        except Exception as e:
            print(f"  [{i}] failed: {str(e)[:80]}")
        if i % 10 == 0:
            print(f"  ...{i}/{len(statements)}")
    report(conn)


if __name__ == "__main__":
    main()
