#!/usr/bin/env python3
"""End-to-end scenario test — exercises the real pipeline (live Gemini) on varied
content and prints the result for inspection. NOT a unit test (needs an API key
and network). Runs against an isolated temp DB so your real brain is untouched.

    python3 scripts/scenario.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain.db as db
import brain.extract as extract
import brain.graph as graph
import brain.llm as llm

USER = "Alvin"
CONTENT = [
    "I'm doing a master's thesis on reinforcement learning at Aalto University, "
    "supervised by Professor Lindgren. It's the most important thing in my life right now.",
    "I play football on Tuesdays and go bouldering on weekends with my friend Bjorn.",
    "My grandmother Astrid turned 90 last month and the whole family gathered in Lund.",
    "I'm learning Rust and built a small CLI tool for log parsing; I need to publish "
    "it to crates.io next week.",
    "Reading 'Thinking, Fast and Slow' — fascinated by cognitive biases, especially anchoring.",
]


def ingest(conn, text):
    existing = db.all_nodes(conn)
    cats = [n["name"] for n in existing if n["type"] == "category"]
    ex = extract.extract(text, source="scenario", existing_names=[n["name"] for n in existing],
                         user=USER, categories=cats)
    links = extract.link_entities(ex.get("nodes", []), existing)
    nids, eids = extract.merge_into_db(conn, ex, "scenario", text, entity_links=links, user=USER)
    extract.embed_nodes(conn, nids)
    return nids, eids


def print_tree(conn):
    kids = graph.children_map(conn)
    seen = set()

    def render(nid, depth):
        if nid in seen:
            return
        seen.add(nid)
        n = db.get_node(conn, nid)
        if not n:
            return
        print("  " * depth + f"- {n['name']} [{n['type']}] imp={n['importance']:.2f} w={n['weight']:.2f}")
        for c in sorted(kids.get(nid, []),
                        key=lambda c: -(db.get_node(conn, c) or {"importance": 0})["importance"]):
            render(c, depth + 1)

    root = db.get_node_by_name(conn, USER)
    render(root["id"], 0)
    has_parent = {e["source_id"] for e in db.all_edges(conn) if e["relation"] == "part_of"}
    orphans = [n for n in db.all_nodes(conn) if n["id"] not in has_parent and n["name"] != USER]
    if orphans:
        print("  ORPHANS (no parent — should be none):")
        for n in orphans:
            print(f"    - {n['name']} [{n['type']}]")


def main():
    if not llm.have_key():
        print("No GEMINI_API_KEY — set one in ~/.personal-brain/.env"); sys.exit(1)
    db.DB_PATH = os.path.join(tempfile.mkdtemp(), "brain.db")
    conn = db.connect()
    db.ensure_identity_anchor(conn, USER)

    print("=" * 70, "\nINGESTING", len(CONTENT), "items...\n" + "=" * 70)
    for t in CONTENT:
        nids, eids = ingest(conn, t)
        print(f"  + {len(nids)} nodes, {len(eids)} edges  <- {t[:60]}...")

    print("\n" + "=" * 70, "\nHIERARCHY\n" + "=" * 70)
    print_tree(conn)

    print("\n" + "=" * 70, "\nCROSS-LINKS (non part_of)\n" + "=" * 70)
    for e in db.all_edges(conn):
        if e["relation"] != "part_of":
            s, t = db.get_node(conn, e["source_id"]), db.get_node(conn, e["target_id"])
            if s and t:
                print(f"  {s['name']} --{e['relation']}--> {t['name']}")

    print("\n" + "=" * 70, "\nSEARCH\n" + "=" * 70)
    for q in ["reinforcement learning", "climbing", "psychology"]:
        kw = [r["name"] for r in db.search_nodes(conn, q)][:3]
        sem = [r["name"] for _, r in graph.semantic_search(conn, llm.embed(q), limit=3)]
        print(f"  {q!r:26} keyword={kw}  semantic={sem}")

    print("\n" + "=" * 70, "\nCONTEXT: 'my studies'\n" + "=" * 70)
    nodes, fb = graph.collect_context_nodes(conn, topic="my studies")
    print(f"  ({'whole-brain' if fb else 'seeded'}, {len(nodes)} nodes)\n")
    print(graph.synthesize_context(nodes, topic="my studies"))


if __name__ == "__main__":
    main()
