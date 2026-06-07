"""Core unit tests for personal-brain.

Stdlib unittest only (no third-party deps) so it runs anywhere:

    python3 -m unittest discover -s tests

Covers the pure logic layer — decay curve, search, dedup/merge, graph traversal,
context seeding, and .env loading. The LLM (Gemini) boundary is mocked, so these
tests are deterministic and need no API key or network.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain
import brain.config as config
import brain.db as db
import brain.decay as decay
import brain.graph as graph
import brain.extract as extract
import brain.llm as llm
import brain.portability as portability

DAY = 86400.0


class BrainTestCase(unittest.TestCase):
    """Base class: each test gets a fresh temp database."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp, "brain.db")
        self.conn = db.connect()

    def tearDown(self):
        self.conn.close()
        db.DB_PATH = self._orig_db_path

    def _age(self, node_id, days):
        """Backdate a node's last_accessed by `days` days."""
        self.conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?",
            (time.time() - days * DAY, node_id),
        )
        self.conn.commit()


class DecayTests(BrainTestCase):
    def test_true_half_life(self):
        """At t == half_life the weight is exactly halved (not 0.368)."""
        w = decay.current_weight(1.0, time.time() - 60 * DAY, 60.0)
        self.assertAlmostEqual(w, 0.5, places=4)

    def test_two_half_lives(self):
        w = decay.current_weight(1.0, time.time() - 120 * DAY, 60.0)
        self.assertAlmostEqual(w, 0.25, places=4)

    def test_person_never_decays(self):
        w = decay.current_weight(1.0, time.time() - 9999 * DAY, float("inf"))
        self.assertEqual(w, 1.0)

    def test_archives_below_threshold(self):
        nid = db.add_node(self.conn, "Ephemeral", type_="event", importance=0.1)  # low importance
        self._age(nid, 45)  # H_eff=7*(1+0.4)=9.8d; 0.5**(45/9.8)≈0.04 < 0.10, floor 0.015
        decay.run_decay(self.conn)
        self.assertEqual(db.get_node(self.conn, nid)["archived"], 1)

    def test_importance_stretches_half_life(self):
        # same elapsed time: important node retains far more weight than trivial one
        t = time.time() - 60 * DAY
        trivial = decay.current_weight(1.0, t, 60.0, importance=0.0)
        important = decay.current_weight(1.0, t, 60.0, importance=1.0)
        self.assertAlmostEqual(trivial, 0.5, places=3)        # one base half-life
        self.assertGreater(important, 0.85)                   # H_eff = 60*5 = 300d

    def test_importance_floor_prevents_archive(self):
        nid = db.add_node(self.conn, "Core skill", type_="event", importance=0.9)  # short base HL
        self._age(nid, 3650)  # 10 years untouched
        decay.run_decay(self.conn)
        row = db.get_node(self.conn, nid)
        self.assertEqual(row["archived"], 0)                  # floor 0.9*0.15=0.135 > 0.10
        self.assertGreaterEqual(row["weight"], 0.135 - 1e-9)

    def test_zero_importance_is_plain_half_life(self):
        self.assertAlmostEqual(
            decay.current_weight(1.0, time.time() - 60 * DAY, 60.0, importance=0.0), 0.5, places=4
        )

    def test_active_node_survives(self):
        nid = db.add_node(self.conn, "Durable", type_="skill")  # half-life 180d
        self._age(nid, 30)
        decay.run_decay(self.conn)
        row = db.get_node(self.conn, nid)
        self.assertEqual(row["archived"], 0)
        self.assertGreater(row["weight"], 0.10)

    def test_deletes_long_archived(self):
        nid = db.add_node(self.conn, "Gone", type_="fact")
        db.archive_node(self.conn, nid)
        self._age(nid, 8)  # archived & untouched > 7 days
        decay.run_decay(self.conn)
        self.assertIsNone(db.get_node(self.conn, nid))

    def test_days_until_archive_math(self):
        # 0.2 * 0.5**(7/7) = 0.1 -> exactly one half-life to the threshold
        self.assertAlmostEqual(decay.days_until_archive(0.2, 7.0), 7.0, places=4)

    def test_days_until_archive_at_or_below_threshold(self):
        self.assertEqual(decay.days_until_archive(0.10, 60.0), 0.0)
        self.assertEqual(decay.days_until_archive(0.05, 60.0), 0.0)

    def test_days_until_archive_immortal_is_inf(self):
        self.assertEqual(decay.days_until_archive(1.0, float("inf")), float("inf"))

    def test_at_risk_lowest_first_excludes_immortal(self):
        a = db.add_node(self.conn, "low", type_="concept")
        b = db.add_node(self.conn, "mid", type_="concept")
        db.add_node(self.conn, "high", type_="concept")
        p = db.add_node(self.conn, "Person", type_="person")  # never decays
        self.conn.execute("UPDATE nodes SET weight=0.15 WHERE id=?", (a,))
        self.conn.execute("UPDATE nodes SET weight=0.50 WHERE id=?", (b,))
        self.conn.execute("UPDATE nodes SET weight=0.12 WHERE id=?", (p,))  # low but immortal
        self.conn.commit()
        names = [r["name"] for r in decay.at_risk_nodes(self.conn, limit=2)]
        self.assertEqual(names, ["low", "mid"])  # lowest decaying first; person excluded


class ClearTests(BrainTestCase):
    def test_clear_empties_all_tables(self):
        a = db.add_node(self.conn, "A", type_="concept")
        b = db.add_node(self.conn, "B", type_="concept")
        db.add_edge(self.conn, a, b, "relates_to")
        db.log_ingestion(self.conn, "raw text", "src", [a, b], [])
        self.conn.commit()
        counts = db.clear(self.conn)
        self.assertEqual(counts["nodes"], 2)
        self.assertEqual(counts["edges"], 1)
        self.assertEqual(counts["log"], 1)
        self.assertEqual(db.all_nodes(self.conn, include_archived=True), [])
        self.assertEqual(db.all_edges(self.conn), [])
        self.assertEqual(db.stats(self.conn)["total"], 0)

    def test_clear_on_empty_brain_is_safe(self):
        self.assertEqual(db.clear(self.conn), {"nodes": 0, "edges": 0, "log": 0})


class SearchTests(BrainTestCase):
    def setUp(self):
        super().setUp()
        db.add_node(self.conn, "transformer architectures", type_="concept",
                    content="attention-based neural nets")
        db.add_node(self.conn, "Data Science", type_="concept", content="ML and stats")
        db.add_node(self.conn, "football", type_="concept", content="a sport")
        self.conn.commit()

    def test_stem_match_plural(self):
        names = [r["name"] for r in db.search_nodes(self.conn, "transformers")]
        self.assertIn("transformer architectures", names)

    def test_unrelated_excluded(self):
        names = [r["name"] for r in db.search_nodes(self.conn, "transformers")]
        self.assertNotIn("football", names)

    def test_ranking_by_overlap(self):
        results = db.search_nodes(self.conn, "data science")
        self.assertEqual(results[0]["name"], "Data Science")

    def test_stopwords_only_returns_nothing_useful(self):
        # "the a of" are stopwords -> falls back to phrase, matches nothing here
        self.assertEqual(db.search_nodes(self.conn, "the of and"), [])


class MergeTests(BrainTestCase):
    def _extracted(self):
        return {
            "nodes": [
                {"name": "Alvin", "type": "person", "content": "owner", "confidence": 1.0},
                {"name": "Transformers", "type": "concept", "content": "nets", "confidence": 0.9},
            ],
            "edges": [{"source": "Alvin", "target": "Transformers", "relation": "studied_by"}],
        }

    def test_creates_nodes_and_edges(self):
        nids, eids = extract.merge_into_db(self.conn, self._extracted(), "src", "raw")
        self.assertEqual(len(nids), 2)
        self.assertEqual(len(eids), 1)

    def test_dedup_by_name_touches_not_duplicates(self):
        db.add_node(self.conn, "Alvin", type_="person")
        self.conn.commit()
        before = len(db.all_nodes(self.conn))
        extract.merge_into_db(self.conn, self._extracted(), "src", "raw")
        after = db.all_nodes(self.conn)
        # Alvin reused (not duplicated): only "Transformers" is new
        self.assertEqual(len(after), before + 1)
        alvins = [n for n in after if n["name"].lower() == "alvin"]
        self.assertEqual(len(alvins), 1)

    def test_entity_links_remap_to_existing(self):
        canonical = db.add_node(self.conn, "attention mechanisms", type_="concept")
        self.conn.commit()
        extracted = {
            "nodes": [{"name": "attention", "type": "concept", "content": "x", "confidence": 0.8}],
            "edges": [],
        }
        extract.merge_into_db(self.conn, extracted, "src", "raw",
                              entity_links={"attention": "attention mechanisms"})
        # no new node created; the existing canonical one was reused
        names = [n["name"] for n in db.all_nodes(self.conn)]
        self.assertEqual(names.count("attention mechanisms"), 1)
        self.assertNotIn("attention", names)


class MergeNodesTests(BrainTestCase):
    def test_repoints_edges_and_deletes_drop(self):
        a = db.add_node(self.conn, "A", type_="concept")
        b = db.add_node(self.conn, "B", type_="concept")
        c = db.add_node(self.conn, "C", type_="concept")
        d = db.add_node(self.conn, "D", type_="concept")
        db.add_edge(self.conn, a, c, "relates_to")
        db.add_edge(self.conn, b, d, "relates_to")
        self.conn.commit()
        self.assertTrue(db.merge_nodes(self.conn, a, b))
        self.assertIsNone(db.get_node(self.conn, b))                 # drop gone
        neighbours = {e["source_id"] for e in db.edges_for_node(self.conn, a)} | \
                     {e["target_id"] for e in db.edges_for_node(self.conn, a)}
        self.assertIn(c, neighbours)                                 # original kept
        self.assertIn(d, neighbours)                                 # re-pointed from b

    def test_skips_self_loop(self):
        a = db.add_node(self.conn, "A", type_="concept")
        b = db.add_node(self.conn, "B", type_="concept")
        db.add_edge(self.conn, a, b, "relates_to")
        self.conn.commit()
        db.merge_nodes(self.conn, a, b)
        for e in db.all_edges(self.conn):
            self.assertNotEqual(e["source_id"], e["target_id"])     # no self-loop created

    def test_dedups_duplicate_edge(self):
        a = db.add_node(self.conn, "A", type_="concept")
        b = db.add_node(self.conn, "B", type_="concept")
        x = db.add_node(self.conn, "X", type_="concept")
        db.add_edge(self.conn, a, x, "relates_to")
        db.add_edge(self.conn, b, x, "relates_to")
        self.conn.commit()
        db.merge_nodes(self.conn, a, b)
        ax = [e for e in db.all_edges(self.conn)
              if {e["source_id"], e["target_id"]} == {a, x}]
        self.assertEqual(len(ax), 1)                                 # reinforced, not duplicated

    def test_missing_node_returns_false(self):
        a = db.add_node(self.conn, "A", type_="concept")
        self.conn.commit()
        self.assertFalse(db.merge_nodes(self.conn, a, "nonexistent"))
        self.assertFalse(db.merge_nodes(self.conn, a, a))            # same id is a no-op


class HierarchyTests(BrainTestCase):
    def _parents_of(self, node_id):
        return {e["target_id"] for e in db.edges_for_node(self.conn, node_id)
                if e["source_id"] == node_id and e["relation"] == "part_of"}

    def test_builds_person_rooted_spine(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        extracted = {
            "nodes": [
                {"name": "Football", "type": "concept", "parent": "Hobbies"},
                {"name": "Game on Sunday", "type": "event", "parent": "Football"},
            ],
            "edges": [],
        }
        extract.merge_into_db(self.conn, extracted, "src", "raw", user="Alvin")
        alvin = db.get_node_by_name(self.conn, "Alvin")
        hobbies = db.get_node_by_name(self.conn, "Hobbies")
        football = db.get_node_by_name(self.conn, "Football")
        sunday = db.get_node_by_name(self.conn, "Game on Sunday")
        # emergent grouping became a category node
        self.assertIsNotNone(hobbies)
        self.assertEqual(hobbies["type"], "category")
        # spine: Sunday -> Football -> Hobbies -> Alvin
        self.assertIn(football["id"], self._parents_of(sunday["id"]))
        self.assertIn(hobbies["id"], self._parents_of(football["id"]))
        self.assertIn(alvin["id"], self._parents_of(hobbies["id"]))

    def test_touch_propagates_up_spine(self):
        cat = db.add_node(self.conn, "Hobbies", type_="category", importance=0.9)
        topic = db.add_node(self.conn, "Football", type_="concept")
        detail = db.add_node(self.conn, "Game Sunday", type_="event")
        db.add_edge(self.conn, topic, cat, "part_of")
        db.add_edge(self.conn, detail, topic, "part_of")
        # let the ancestors decay a bit
        self.conn.execute("UPDATE nodes SET weight=0.2 WHERE id IN (?,?)", (cat, topic))
        self.conn.commit()
        db.touch_node(self.conn, detail)  # touching the leaf
        # parent (depth1) boosted to >= 0.6, grandparent (depth2) to >= 0.36
        self.assertGreaterEqual(db.get_node(self.conn, topic)["weight"], 0.6 - 1e-9)
        self.assertGreaterEqual(db.get_node(self.conn, cat)["weight"], 0.36 - 1e-9)

    def test_touch_no_spine_is_noop_for_others(self):
        a = db.add_node(self.conn, "Lonely", type_="concept")
        b = db.add_node(self.conn, "Other", type_="concept")
        self.conn.execute("UPDATE nodes SET weight=0.3 WHERE id=?", (b,))
        self.conn.commit()
        db.touch_node(self.conn, a)  # a has no parents
        self.assertEqual(db.get_node(self.conn, b)["weight"], 0.3)  # unrelated untouched

    def test_children_map(self):
        cat = db.add_node(self.conn, "Cat", type_="category")
        c1 = db.add_node(self.conn, "C1", type_="concept")
        c2 = db.add_node(self.conn, "C2", type_="concept")
        db.add_edge(self.conn, c1, cat, "part_of")
        db.add_edge(self.conn, c2, cat, "part_of")
        db.add_edge(self.conn, c1, c2, "relates_to")  # non-hierarchy edge ignored
        self.conn.commit()
        m = graph.children_map(self.conn)
        self.assertEqual(set(m[cat]), {c1, c2})

    def test_category_never_decays(self):
        from brain import db as _db
        self.assertEqual(_db.HALF_LIVES["category"], float("inf"))
        cid = db.add_node(self.conn, "Career", type_="category")
        self.assertEqual(db.get_node(self.conn, cid)["half_life_days"], float("inf"))

    def test_reuses_existing_category(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        cat = db.add_node(self.conn, "Hobbies", type_="category", importance=0.9)
        self.conn.commit()
        before = len(db.all_nodes(self.conn))
        extract.merge_into_db(self.conn, {
            "nodes": [{"name": "Chess", "type": "concept", "parent": "Hobbies"}],
            "edges": [],
        }, "src", "raw", user="Alvin")
        # only "Chess" is new; "Hobbies" reused, not duplicated
        self.assertEqual(len([n for n in db.all_nodes(self.conn) if n["name"] == "Hobbies"]), 1)
        chess = db.get_node_by_name(self.conn, "Chess")
        self.assertIn(cat, self._parents_of(chess["id"]))


class GraphTests(BrainTestCase):
    def setUp(self):
        super().setUp()
        self.a = db.add_node(self.conn, "transformer architectures", type_="concept")
        self.b = db.add_node(self.conn, "attention", type_="concept")
        self.c = db.add_node(self.conn, "football", type_="concept")
        db.add_edge(self.conn, self.a, self.b, "relates_to")
        self.conn.commit()

    def test_bfs_reaches_neighbor(self):
        visited = graph.bfs(self.conn, [self.a], depth=1)
        self.assertIn(self.b, visited)
        self.assertNotIn(self.c, visited)

    def test_bfs_depth_zero_only_self(self):
        visited = graph.bfs(self.conn, [self.a], depth=0)
        self.assertEqual(set(visited), {self.a})

    def test_bfs_does_not_expand_through_hub(self):
        # topic -- hub(high degree) -- many unrelated leaves
        topic = db.add_node(self.conn, "layer norm", type_="concept")
        hub = db.add_node(self.conn, "Alvin", type_="person")
        db.add_edge(self.conn, topic, hub, "relates_to")
        leaves = []
        for i in range(10):
            leaf = db.add_node(self.conn, f"unrelated {i}", type_="concept")
            db.add_edge(self.conn, hub, leaf, "relates_to")
            leaves.append(leaf)
        self.conn.commit()
        visited = graph.bfs(self.conn, [topic], depth=3, hub_degree=8)
        self.assertIn(hub, visited)              # hub itself is included
        for leaf in leaves:
            self.assertNotIn(leaf, visited)      # but its other neighbours are not

    def test_bfs_seed_hub_still_expands(self):
        hub = db.add_node(self.conn, "Hub", type_="person")
        leaves = []
        for i in range(10):
            leaf = db.add_node(self.conn, f"leaf {i}", type_="concept")
            db.add_edge(self.conn, hub, leaf, "relates_to")
            leaves.append(leaf)
        self.conn.commit()
        visited = graph.bfs(self.conn, [hub], depth=1, hub_degree=8)
        for leaf in leaves:                      # seed expands even though it's a hub
            self.assertIn(leaf, visited)

    def test_context_keyword_path(self):
        nodes, fb = graph.collect_context_nodes(self.conn, topic="transformers", depth=2)
        self.assertFalse(fb)
        self.assertIn(self.a, nodes)

    def test_context_fallback_to_whole_brain(self):
        nodes, fb = graph.collect_context_nodes(self.conn, topic="quantum chromodynamics")
        self.assertTrue(fb)
        self.assertEqual(set(nodes), {self.a, self.b, self.c})

    def test_hub_cap_floor_and_scaling(self):
        # sparse graph -> floor
        self.assertEqual(graph.hub_cap(self.conn, floor=5), 5)
        # dense clique raises the cap above the floor
        clique = [db.add_node(self.conn, f"c{i}", type_="concept") for i in range(6)]
        for i in range(6):
            for j in range(i + 1, 6):
                db.add_edge(self.conn, clique[i], clique[j], "relates_to")
        self.conn.commit()
        self.assertGreater(graph.hub_cap(self.conn, floor=5), 5)

    def test_context_excludes_unrelated_through_hub(self):
        seed = db.add_node(self.conn, "rust lang", type_="concept", content="systems lang")
        hub = db.add_node(self.conn, "Owner", type_="person")
        db.add_edge(self.conn, seed, hub, "relates_to")
        noise = []
        for i in range(10):
            n = db.add_node(self.conn, f"noise {i}", type_="concept")
            db.add_edge(self.conn, hub, n, "relates_to")
            noise.append(n)
        self.conn.commit()
        nodes, fb = graph.collect_context_nodes(self.conn, topic="rust")
        self.assertFalse(fb)
        self.assertIn(seed, nodes)
        self.assertIn(hub, nodes)              # hub kept
        self.assertTrue(all(n not in nodes for n in noise))  # its other leaves pruned

    def test_context_empty_brain(self):
        for n in db.all_nodes(self.conn):
            db.delete_node(self.conn, n["id"])
        self.conn.commit()
        nodes, _ = graph.collect_context_nodes(self.conn, topic="anything")
        self.assertEqual(nodes, {})


class SynthesizeTests(BrainTestCase):
    def setUp(self):
        super().setUp()
        self.hub = db.add_node(self.conn, "transformer architectures", type_="concept")
        self.b = db.add_node(self.conn, "attention", type_="concept")
        db.add_edge(self.conn, self.hub, self.b, "part_of")  # connected pair
        self.iso = db.add_node(self.conn, "layer normalization", type_="concept",
                               content="stabilises training")  # isolated
        self.conn.commit()
        self._orig_generate = llm.generate

    def tearDown(self):
        llm.generate = self._orig_generate
        super().tearDown()

    def test_connects_isolated_node(self):
        llm.generate = lambda *a, **k: '{"target": "transformer architectures", "relation": "used_in"}'
        made = graph.connect_isolated_nodes(self.conn)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0]["source"], "layer normalization")
        self.assertEqual(made[0]["relation"], "used_in")
        # edge actually persisted
        self.assertTrue(db.edges_for_node(self.conn, self.iso))

    def test_null_suggestion_makes_no_edge(self):
        llm.generate = lambda *a, **k: "null"
        self.assertEqual(graph.connect_isolated_nodes(self.conn), [])
        self.assertEqual(db.edges_for_node(self.conn, self.iso), [])

    def test_unknown_target_skipped(self):
        llm.generate = lambda *a, **k: '{"target": "nonexistent node", "relation": "x"}'
        self.assertEqual(graph.connect_isolated_nodes(self.conn), [])

    def test_garbage_response_is_not_fatal(self):
        llm.generate = lambda *a, **k: "the model rambled with no json at all"
        self.assertEqual(graph.connect_isolated_nodes(self.conn), [])


class RelationVocabTests(BrainTestCase):
    def test_exact_vocab_preserved(self):
        for rel in db.RELATIONS:
            self.assertEqual(db.normalize_relation(rel), rel)

    def test_freeform_mapped_to_vocab(self):
        cases = {
            "relies on": "requires",
            "is a key component of": "part_of",
            "depends on": "requires",
            "works for": "works_at",
            "was created by": "created_by",
            "USED-BY": "used_in",
            "located in": "located_at",
        }
        for raw, expected in cases.items():
            self.assertEqual(db.normalize_relation(raw), expected, raw)

    def test_unknown_defaults_to_relates_to(self):
        self.assertEqual(db.normalize_relation("blah blah"), "relates_to")
        self.assertEqual(db.normalize_relation(""), "relates_to")
        self.assertEqual(db.normalize_relation(None), "relates_to")

    def test_add_edge_normalizes_and_stores_vocab(self):
        a = db.add_node(self.conn, "A", type_="concept")
        b = db.add_node(self.conn, "B", type_="concept")
        db.add_edge(self.conn, a, b, "is a key component of")
        self.conn.commit()
        rel = db.all_edges(self.conn)[0]["relation"]
        self.assertEqual(rel, "part_of")
        self.assertIn(rel, db.RELATIONS)


class ChunkingTests(BrainTestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(extract._chunk_text("hello world"), ["hello world"])

    def test_empty_text(self):
        self.assertEqual(extract._chunk_text("   "), [])

    def test_long_text_splits_within_size(self):
        text = "\n\n".join("para %d %s" % (i, "x" * 500) for i in range(40))  # ~20k
        chunks = extract._chunk_text(text, size=4000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 4000 for c in chunks))

    def test_oversized_paragraph_hard_split(self):
        chunks = extract._chunk_text("y" * 9000, size=4000)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 4000 for c in chunks))

    def test_extract_processes_all_chunks_and_dedups(self):
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return ('{"nodes":[{"name":"Shared","type":"concept"},'
                    '{"name":"Uniq%d","type":"concept"}],"edges":[]}' % calls["n"])

        orig = llm.generate
        llm.generate = fake
        try:
            text = "\n\n".join("p%d %s" % (i, "z" * 1000) for i in range(12))  # multi-chunk
            out = extract.extract(text)
        finally:
            llm.generate = orig

        self.assertGreaterEqual(calls["n"], 2)  # every chunk hit the model
        names = [n["name"] for n in out["nodes"]]
        self.assertEqual(names.count("Shared"), 1)  # deduped across chunks


class SemanticSearchTests(BrainTestCase):
    def test_cosine(self):
        self.assertAlmostEqual(graph.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(graph.cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(graph.cosine([1, 0], [-1, 0]), -1.0)
        self.assertEqual(graph.cosine([], [1, 2]), 0.0)
        self.assertEqual(graph.cosine([1, 2], [1, 2, 3]), 0.0)  # mismatched dims

    def test_set_embedding_roundtrip(self):
        nid = db.add_node(self.conn, "X", type_="concept")
        db.set_embedding(self.conn, nid, [0.1, 0.2, 0.3])
        self.conn.commit()
        import json as _json
        self.assertEqual(_json.loads(db.get_node(self.conn, nid)["embedding"]), [0.1, 0.2, 0.3])

    def test_semantic_search_ranks_by_similarity(self):
        near = db.add_node(self.conn, "neural networks", type_="concept")
        mid = db.add_node(self.conn, "data science", type_="concept")
        far = db.add_node(self.conn, "football", type_="concept")
        db.set_embedding(self.conn, near, [1.0, 0.0, 0.0])
        db.set_embedding(self.conn, mid, [0.7, 0.7, 0.0])
        db.set_embedding(self.conn, far, [0.0, 0.0, 1.0])
        self.conn.commit()
        ranked = graph.semantic_search(self.conn, [1.0, 0.0, 0.0], limit=3)
        names = [r["name"] for _, r in ranked]
        self.assertEqual(names, ["neural networks", "data science", "football"])
        self.assertGreater(ranked[0][0], ranked[1][0])  # scores descend

    def test_semantic_search_skips_unembedded(self):
        db.add_node(self.conn, "no-embedding", type_="concept")
        self.conn.commit()
        self.assertEqual(graph.semantic_search(self.conn, [1.0, 0.0]), [])

    def test_embed_nodes_populates_and_is_searchable(self):
        a = db.add_node(self.conn, "rust", type_="concept", content="systems lang")
        b = db.add_node(self.conn, "go", type_="concept", content="another lang")
        self.conn.commit()
        orig = llm.embed
        llm.embed = lambda text, *a, **k: [1.0, 0.0] if "rust" in text.lower() else [0.0, 1.0]
        try:
            n = extract.embed_nodes(self.conn, [a, b])
        finally:
            llm.embed = orig
        self.assertEqual(n, 2)
        top = graph.semantic_search(self.conn, [1.0, 0.0], limit=1)
        self.assertEqual(top[0][1]["name"], "rust")

    def test_embed_nodes_skips_already_embedded_and_swallows_errors(self):
        a = db.add_node(self.conn, "has-emb", type_="concept")
        b = db.add_node(self.conn, "boom", type_="concept")
        db.set_embedding(self.conn, a, [0.5, 0.5])
        self.conn.commit()

        def boom(*a, **k):
            raise RuntimeError("API down")

        orig = llm.embed
        llm.embed = boom
        try:
            n = extract.embed_nodes(self.conn, [a, b])  # a skipped, b errors -> swallowed
        finally:
            llm.embed = orig
        self.assertEqual(n, 0)  # nothing newly embedded, no exception raised


class PortabilityTests(BrainTestCase):
    def _seed(self):
        a = db.add_node(self.conn, "transformers", type_="concept", content="nets")
        b = db.add_node(self.conn, "Alvin", type_="person", content="owner")
        c = db.add_node(self.conn, "old fact", type_="fact")
        db.archive_node(self.conn, c)
        db.add_edge(self.conn, a, b, "studied_by")
        self.conn.commit()
        return a, b, c

    def _fresh_conn(self):
        path = os.path.join(tempfile.mkdtemp(), "brain.db")
        orig = db.DB_PATH
        db.DB_PATH = path
        conn = db.connect()
        db.DB_PATH = orig
        return conn

    def test_export_includes_everything(self):
        self._seed()
        data = portability.export_brain(self.conn)
        self.assertEqual(data["schema_version"], portability.SCHEMA_VERSION)
        self.assertEqual(len(data["nodes"]), 3)   # archived node included
        self.assertEqual(len(data["edges"]), 1)

    def test_roundtrip_into_fresh_brain(self):
        self._seed()
        data = portability.export_brain(self.conn)
        dest = self._fresh_conn()
        n, e = portability.import_brain(dest, data)
        self.assertEqual(n, 3)
        self.assertEqual(e, 1)
        self.assertEqual(len(db.all_nodes(dest, include_archived=True)), 3)
        self.assertEqual(len(db.all_edges(dest)), 1)
        # an exported node keeps its identity and fields
        t = db.get_node_by_name(dest, "transformers")
        self.assertEqual(t["type"], "concept")

    def test_import_is_idempotent(self):
        self._seed()
        data = portability.export_brain(self.conn)
        dest = self._fresh_conn()
        portability.import_brain(dest, data)
        n, e = portability.import_brain(dest, data)  # second time
        self.assertEqual((n, e), (0, 0))

    def test_import_dedups_by_name_and_remaps_edges(self):
        self._seed()
        data = portability.export_brain(self.conn)
        dest = self._fresh_conn()
        # dest already has "transformers" under a different id
        db.add_node(dest, "transformers", type_="concept")
        db.add_node(dest, "Alvin", type_="person")
        dest.commit()
        n, e = portability.import_brain(dest, data)
        self.assertEqual(n, 1)  # only "old fact" is new; transformers/Alvin deduped
        # edge studied_by remapped onto the existing nodes
        self.assertEqual(len(db.all_edges(dest)), 1)


class ParseJsonTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(llm.parse_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(llm.parse_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_salvage_from_prose(self):
        self.assertEqual(llm.parse_json('here you go: {"a": 1} done'), {"a": 1})

    def test_null_passes_through(self):
        self.assertIsNone(llm.parse_json("null"))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig = config.CONFIG_PATH
        config.CONFIG_PATH = Path(self._tmp) / "config.json"

    def tearDown(self):
        config.CONFIG_PATH = self._orig

    def test_missing_returns_empty(self):
        self.assertEqual(config.load(), {})
        self.assertEqual(config.get_user(), "")

    def test_corrupt_returns_empty_not_crash(self):
        config.CONFIG_PATH.write_text("{ not valid json ,,,")
        self.assertEqual(config.load(), {})
        self.assertEqual(config.get_user(), "")  # must not raise

    def test_non_dict_json_returns_empty(self):
        config.CONFIG_PATH.write_text("[1, 2, 3]")
        self.assertEqual(config.load(), {})

    def test_roundtrip(self):
        config.set_user("Alvin")
        self.assertEqual(config.get_user(), "Alvin")


class DotenvTests(unittest.TestCase):
    def test_overrides_empty_env_var(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / ".env").write_text('BRAIN_TEST_KEY=from-file\n')
        orig_dir, orig_val = brain.DATA_DIR, os.environ.get("BRAIN_TEST_KEY")
        cwd = os.getcwd()
        try:
            os.environ["BRAIN_TEST_KEY"] = ""        # empty shell export
            brain.DATA_DIR = Path(tmp)
            os.chdir(tempfile.mkdtemp())             # isolate from project .env
            brain._load_dotenv()
            self.assertEqual(os.environ["BRAIN_TEST_KEY"], "from-file")
        finally:
            brain.DATA_DIR = orig_dir
            os.chdir(cwd)
            if orig_val is None:
                os.environ.pop("BRAIN_TEST_KEY", None)
            else:
                os.environ["BRAIN_TEST_KEY"] = orig_val

    def test_real_env_var_wins(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / ".env").write_text('BRAIN_TEST_KEY2=from-file\n')
        orig_dir = brain.DATA_DIR
        cwd = os.getcwd()
        try:
            os.environ["BRAIN_TEST_KEY2"] = "from-shell"
            brain.DATA_DIR = Path(tmp)
            os.chdir(tempfile.mkdtemp())
            brain._load_dotenv()
            self.assertEqual(os.environ["BRAIN_TEST_KEY2"], "from-shell")
        finally:
            brain.DATA_DIR = orig_dir
            os.chdir(cwd)
            os.environ.pop("BRAIN_TEST_KEY2", None)


class ExtractMockTests(BrainTestCase):
    def test_extract_parses_mocked_llm(self):
        orig = llm.generate
        llm.generate = lambda *a, **k: (
            '{"nodes":[{"name":"X","type":"concept","content":"c","confidence":0.9}],"edges":[]}'
        )
        try:
            out = extract.extract("some text about X")
            self.assertEqual(out["nodes"][0]["name"], "X")
        finally:
            llm.generate = orig

    def test_extract_strips_code_fences(self):
        raw = '```json\n{"nodes":[],"edges":[]}\n```'
        self.assertEqual(extract._parse_json(raw), {"nodes": [], "edges": []})

    def test_parse_json_salvages_from_prose(self):
        raw = 'Sure! Here is the JSON:\n{"nodes": [], "edges": []}\nHope that helps.'
        self.assertEqual(extract._parse_json(raw), {"nodes": [], "edges": []})


class RobustMergeTests(BrainTestCase):
    def test_skips_nameless_nodes(self):
        extracted = {
            "nodes": [
                {"type": "concept", "content": "no name"},      # malformed
                {"name": "  ", "type": "concept"},               # blank name
                {"name": "Valid", "type": "concept", "content": "ok"},
            ],
            "edges": [],
        }
        nids, _ = extract.merge_into_db(self.conn, extracted, "src", "raw")
        self.assertEqual(len(nids), 1)
        self.assertEqual(db.all_nodes(self.conn)[0]["name"], "Valid")

    def test_tolerates_malformed_edges(self):
        extracted = {
            "nodes": [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}],
            "edges": [
                {"source": "A"},                                  # missing target
                {"relation": "relates_to"},                       # missing both
                {"source": "A", "target": "B", "relation": "relates_to"},
            ],
        }
        nids, eids = extract.merge_into_db(self.conn, extracted, "src", "raw")
        self.assertEqual(len(nids), 2)
        self.assertEqual(len(eids), 1)  # only the well-formed edge


if __name__ == "__main__":
    unittest.main(verbosity=2)
