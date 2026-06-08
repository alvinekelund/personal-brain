#!/usr/bin/env python3
"""Generate a large synthetic life-log and ingest it to see how the brain reacts
at scale — category spread, tree depth, hubs, decay. Live Gemini.

    python scripts/generate_dataset.py [N] [--into PATH] [--user NAME]

By default ingests into an isolated temp brain (your real brain stays untouched)
and prints a report. Pass --into PATH to write to a real brain.db you can serve
and watch evolve in the browser:

    # populate a separate brain at /tmp/big-brain
    python scripts/generate_dataset.py 100 --into /tmp/big-brain/brain.db

    # then in another terminal, serve it (HOME points at the parent of .personal-brain)
    mkdir -p /tmp/big-brain/.personal-brain && cp /tmp/big-brain/brain.db /tmp/big-brain/.personal-brain/
    cp ~/.personal-brain/.env /tmp/big-brain/.personal-brain/  # share the API key
    HOME=/tmp/big-brain brain serve --port 8001
"""
import argparse
import os
import random
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain.db as db
import brain.extract as extract
import brain.graph as graph
import brain.llm as llm

random.seed(42)

# Larger, more varied corpus → meaningful brain at N=100+ without heavy repetition
PEOPLE = ["Bjorn", "Heli", "Astrid", "Sara", "Mikael", "Nina", "Omar", "Priya",
          "Tom", "Lena", "Sofia", "Andre", "Yuki", "Kasper", "Maria", "Ravi",
          "Elena", "Felix", "Ines", "Karim"]
SPORTS = ["football", "bouldering", "tennis", "running", "swimming", "cycling",
          "skiing", "surfing", "yoga", "boxing", "padel", "kayaking"]
DAYS = ["Mondays", "Tuesdays", "weekends", "Friday evenings", "Sunday mornings",
        "Wednesday nights", "every other week"]
BOOKS = ["Thinking Fast and Slow", "Sapiens", "The Pragmatic Programmer", "Dune",
         "Antifragile", "Designing Data-Intensive Applications", "Atomic Habits",
         "The Selfish Gene", "Gödel Escher Bach", "Zero to One"]
TOPICS = ["cognitive biases", "history", "systems design", "stoicism", "complexity",
          "game theory", "evolution", "macroeconomics", "linguistics", "consciousness",
          "behavioural economics", "category theory"]
PROJECTS = ["a log parser", "a recommender system", "a personal website", "a trading bot",
            "a chat app", "a knowledge graph", "a synth plugin", "a notes app",
            "a habit tracker", "a podcast feed"]
TECH = ["Rust", "Python", "TypeScript", "Go", "PyTorch", "React", "PostgreSQL",
        "Redis", "Docker", "Kubernetes", "WebAssembly", "Three.js"]
SKILLS = ["the piano", "Spanish", "public speaking", "sketching", "chess",
          "cooking Thai food", "guitar", "French", "negotiation", "drawing",
          "improv", "knife skills", "calisthenics"]
PLACES = ["Lund", "Berlin", "Tokyo", "Lisbon", "the Alps", "Helsinki", "Porto",
          "Kyoto", "Amsterdam", "Reykjavík", "Barcelona", "Edinburgh"]
COURSES = ["reinforcement learning", "distributed systems", "linear algebra",
           "macroeconomics", "NLP", "graph theory", "compilers", "statistics",
           "operating systems", "behavioral psychology"]
ORGS = ["Aalto University", "Harvard", "Siemens", "a startup called Vexa",
        "the student union", "ETH", "MIT", "Anthropic", "DeepMind", "Spotify"]
EMOTIONS = ["excited about", "frustrated by", "curious about", "proud of", "worried about"]

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
    lambda: f"I'm {random.choice(EMOTIONS)} my work on {random.choice(PROJECTS)}.",
    lambda: f"{random.choice(PEOPLE)} recommended I read '{random.choice(BOOKS)}'.",
    lambda: f"I joined a {random.choice(SPORTS)} club and met {random.choice(PEOPLE)} there.",
    lambda: f"I'm planning to move to {random.choice(PLACES)} next year for {random.choice(ORGS)}.",
    lambda: f"I started using {random.choice(TECH)} for {random.choice(PROJECTS)}.",
    lambda: f"My family — especially {random.choice(PEOPLE)} — visited me last weekend.",
    lambda: f"I've been thinking about {random.choice(TOPICS)} a lot lately.",
    lambda: f"I bought a new book on {random.choice(TOPICS)} after talking to {random.choice(PEOPLE)}.",
]


def corpus(n):
    seen, out = set(), []
    guard = 0
    while len(out) < n and guard < n * 80:
        s = random.choice(TEMPLATES)()
        guard += 1
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def report(conn, user):
    from collections import Counter
    nodes = db.all_nodes(conn, include_archived=True)
    edges = db.all_edges(conn)
    by_type = Counter(n["type"] for n in nodes)
    kids = graph.children_map(conn)

    cats = [n for n in nodes if n["type"] == "category"]
    cat_sizes = sorted(((len(kids.get(c["id"], [])), c["name"]) for c in cats), reverse=True)

    deg = Counter()
    for e in edges:
        deg[e["source_id"]] += 1
        deg[e["target_id"]] += 1
    top_hubs = sorted(((d, db.get_node(conn, nid)) for nid, d in deg.items()
                       if db.get_node(conn, nid)), key=lambda x: -x[0])[:8]

    root = db.get_node_by_name(conn, user)
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

    print("\n" + "=" * 64 + "\nBRAIN REACTION\n" + "=" * 64)
    print(f"nodes: {len(nodes)}  edges: {len(edges)}  categories: {len(cats)}  tree depth: {maxd[0]}")
    print(f"by type: {dict(by_type)}\n")
    print("category sizes (direct children):")
    for sz, name in cat_sizes:
        print(f"  {sz:3d}  {name}")
    print("\ntop hubs (degree):")
    for d, n in top_hubs:
        print(f"  {d:3d}  {n['name']} [{n['type']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=40, help="number of statements to ingest")
    ap.add_argument("--into", help="path to brain.db (default: isolated temp dir)")
    ap.add_argument("--user", default="Alvin", help="identity name (default: Alvin)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent ingests (default: 4; set to 1 for sequential)")
    args = ap.parse_args()

    if not llm.have_key():
        print("No GEMINI_API_KEY — set one in ~/.personal-brain/.env"); sys.exit(1)

    if args.into:
        Path(args.into).parent.mkdir(parents=True, exist_ok=True)
        db.DB_PATH = args.into
        where = args.into
    else:
        db.DB_PATH = os.path.join(tempfile.mkdtemp(), "brain.db")
        where = db.DB_PATH + "  (isolated temp brain)"

    conn = db.connect()
    db.ensure_identity_anchor(conn, args.user)

    # When running ingests in parallel, pre-create the common top-level categories
    # so multiple workers don't race to create the same one (which would dupe them).
    if args.parallel > 1:
        root = db.get_node_by_name(conn, args.user)
        for cat in ["Career", "Hobbies", "Education", "Relationships", "Skills",
                    "Projects", "Learning", "Life Events", "Health", "Family"]:
            if not db.get_node_by_name(conn, cat):
                cid = db.add_node(conn, name=cat, type_="category",
                                  source="pre-seed", importance=0.85)
                if root:
                    db.add_edge(conn, cid, root["id"], "part_of")
        conn.commit()

    statements = corpus(args.n)
    mode = f"parallel={args.parallel}" if args.parallel > 1 else "sequential"
    print(f"Ingesting {len(statements)} synthetic statements as '{args.user}' "
          f"({mode}) → {where}")
    t0 = time.time()
    ok = failed = 0

    def progress(i):
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed else 0
        eta = (len(statements) - i) / rate if rate else 0
        n_nodes = len(db.all_nodes(conn))
        print(f"  ...{i}/{len(statements)}  ok={ok} fail={failed}  "
              f"nodes={n_nodes}  rate={rate:.2f}/s  eta={eta:.0f}s")

    if args.parallel > 1:
        # Each worker thread opens its own sqlite connection (connections are
        # thread-affine). WAL mode lets readers and one writer at a time work
        # concurrently — fine for our handful of workers.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def ingest_one(s):
            wconn = db.connect()
            try:
                extract.ingest(wconn, s, source="synthetic", user=args.user)
                return None
            except Exception as e:
                return str(e)[:80]
            finally:
                wconn.close()

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(ingest_one, s) for s in statements]
            for i, f in enumerate(as_completed(futs), 1):
                err = f.result()
                if err:
                    failed += 1
                    print(f"  failed: {err}")
                else:
                    ok += 1
                if i % 5 == 0 or i == len(statements):
                    progress(i)
    else:
        for i, s in enumerate(statements, 1):
            try:
                extract.ingest(conn, s, source="synthetic", user=args.user)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"  [{i}] failed: {str(e)[:80]}")
            if i % 5 == 0 or i == len(statements):
                progress(i)

    report(conn, args.user)
    if args.into:
        print(f"\nDone. Serve this brain in your browser:")
        parent = str(Path(args.into).resolve().parent.parent)
        print(f"  HOME={parent} brain serve --port 8001")


if __name__ == "__main__":
    main()
