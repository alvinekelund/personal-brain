"""Core unit tests for personal-brain.

Stdlib unittest only (no third-party deps) so it runs anywhere:

    python3 -m unittest discover -s tests

Covers the pure logic layer — decay curve, search, dedup/merge, graph traversal,
context seeding, and .env loading. The LLM (Gemini) boundary is mocked, so these
tests are deterministic and need no API key or network.
"""
import io
import json
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
        # hermetic by default: no test should hit the network unless it opts in
        # by mocking llm. (_load_dotenv may have loaded a real key into the env.)
        self._orig_have_key = llm.have_key
        llm.have_key = lambda: False

    def tearDown(self):
        self.conn.close()
        db.DB_PATH = self._orig_db_path
        llm.have_key = self._orig_have_key

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

    def test_days_until_archive_importance_aware(self):
        # important node is floored above threshold → never archives
        self.assertEqual(decay.days_until_archive(1.0, 7.0, importance=1.0), float("inf"))
        # importance also stretches the timeline vs a trivial node
        trivial = decay.days_until_archive(1.0, 7.0, importance=0.0)
        modest = decay.days_until_archive(1.0, 7.0, importance=0.2)  # floor 0.03 < 0.10
        self.assertGreater(modest, trivial)

    def test_at_risk_excludes_floored_important(self):
        trivial = db.add_node(self.conn, "errand", type_="task", importance=0.1)
        important = db.add_node(self.conn, "core value", type_="fact", importance=0.9)
        self.conn.execute("UPDATE nodes SET weight=0.3 WHERE id IN (?,?)", (trivial, important))
        self.conn.commit()
        names = [r["name"] for r in decay.at_risk_nodes(self.conn)]
        self.assertIn("errand", names)            # low importance → at risk
        self.assertNotIn("core value", names)     # floored → never archives, excluded

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


class ServerTests(BrainTestCase):
    def test_fingerprint_changes_on_change(self):
        from brain import server
        f0 = server.fingerprint(self.conn)
        nid = db.add_node(self.conn, "X", type_="concept")
        self.conn.commit()
        f1 = server.fingerprint(self.conn)
        self.assertNotEqual(f0, f1)            # add changes it
        db.touch_node(self.conn, nid)
        self.conn.commit()
        self.assertNotEqual(f1, server.fingerprint(self.conn))  # access changes it

    def test_render_page_injects_live_reload(self):
        from brain import server
        import brain.visualize as visualize
        orig = visualize.build_html
        visualize.build_html = lambda conn, **k: "<html><body>GRAPH</body></html>"
        try:
            html = server.render_page(self.conn, interval=3)
        finally:
            visualize.build_html = orig
        self.assertIn("GRAPH", html)
        self.assertIn("/version", html)            # polls for changes
        self.assertIn("location.reload", html)     # reloads on change
        self.assertIn("talk to your brain", html)  # in-page add box
        self.assertIn("/add", html)                # posts new content
        self.assertIn("/query", html)              # full feature set wired in
        self.assertIn("/context", html)
        self.assertIn("/synthesize", html)
        self.assertTrue(html.endswith("</body></html>") or "</body>" in html)


class IngestTests(BrainTestCase):
    def test_ingest_runs_full_pipeline_under_a_category(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        responses = iter([
            '{"nodes":[{"name":"Piano","type":"skill","parent":"Hobbies","importance":0.6}],"edges":[]}',
            '{}',  # link_entities
        ])
        orig = llm.generate
        llm.generate = lambda *a, **k: next(responses, "{}")
        try:
            nids, _ = extract.ingest(self.conn, "I started learning piano", source="web", user="Alvin")
        finally:
            llm.generate = orig
        self.assertTrue(nids)
        piano = db.get_node_by_name(self.conn, "Piano")
        self.assertIsNotNone(piano)
        parents = [e for e in db.edges_for_node(self.conn, piano["id"])
                   if e["source_id"] == piano["id"] and e["relation"] == "part_of"]
        self.assertTrue(parents)  # placed under a category, not floating


class Visualize3DTests(BrainTestCase):
    def test_build_html_3d_inlines_data(self):
        import brain.visualize as visualize
        a = db.add_node(self.conn, "Transformers", type_="concept", content="nets")
        b = db.add_node(self.conn, "Alvin", type_="person")
        db.add_edge(self.conn, a, b, "studied_by")
        self.conn.commit()
        html = visualize.build_html_3d(self.conn)
        self.assertIn("3d-force-graph", html)   # WebGL library
        self.assertIn("ForceGraph3D", html)
        self.assertIn("Transformers", html)     # node data inlined
        self.assertIn('"links"', html)
        self.assertNotIn("__DATA__", html)      # placeholder substituted

    def test_build_html_3d_min_weight_filters(self):
        import brain.visualize as visualize
        hi = db.add_node(self.conn, "HiNode", type_="concept")
        lo = db.add_node(self.conn, "LoNode", type_="concept")
        self.conn.execute("UPDATE nodes SET weight=0.2 WHERE id=?", (lo,))
        self.conn.commit()
        html = visualize.build_html_3d(self.conn, min_weight=0.5)
        self.assertIn("HiNode", html)
        self.assertNotIn("LoNode", html)        # faded node filtered out


class ServerApiTests(BrainTestCase):
    def test_api_status(self):
        from brain import server
        db.add_node(self.conn, "X", type_="concept")
        self.conn.commit()
        out = server.api_status(self.conn)
        self.assertIn("stats", out)
        self.assertIn("fading", out)
        self.assertGreaterEqual(out["stats"]["total"], 1)

    def test_api_query_keyword(self):
        from brain import server
        db.add_node(self.conn, "transformer architectures", type_="concept", content="nets")
        self.conn.commit()
        out = server.api_query(self.conn, "transformers", semantic=False)
        self.assertTrue(any(r["name"] == "transformer architectures" for r in out))

    def test_api_node(self):
        from brain import server
        a = db.add_node(self.conn, "Football", type_="concept", content="a sport", importance=0.6)
        b = db.add_node(self.conn, "Bjorn", type_="person")
        db.add_edge(self.conn, b, a, "relates_to")
        self.conn.commit()
        out = server.api_node(self.conn, a)
        self.assertEqual(out["name"], "Football")
        self.assertEqual(out["importance"], 0.6)
        self.assertEqual(out["content"], "a sport")
        self.assertTrue(any(e["other"] == "Bjorn" for e in out["edges"]))
        self.assertEqual(server.api_node(self.conn, "nope"), {"error": "not found"})

    def test_api_tree(self):
        from brain import server
        db.ensure_identity_anchor(self.conn, "Alvin")
        cat = db.add_node(self.conn, "Hobbies", type_="category")
        n = db.add_node(self.conn, "football", type_="concept")
        db.add_edge(self.conn, n, cat, "part_of")
        db.add_edge(self.conn, cat, db.get_node_by_name(self.conn, "Alvin")["id"], "part_of")
        self.conn.commit()
        text = server.api_tree(self.conn, user="Alvin")
        self.assertIn("Alvin", text)
        self.assertIn("Hobbies", text)
        self.assertIn("football", text)


class ReorganizeTests(BrainTestCase):
    def test_reorganize_builds_hierarchy_and_rescores(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        db.add_node(self.conn, "football", type_="concept")
        db.add_node(self.conn, "Bjorn", type_="person")
        self.conn.commit()
        orig = extract.plan_hierarchy
        extract.plan_hierarchy = lambda nodes, user, cats: [
            {"name": "football", "parent": "Hobbies", "importance": 0.6},
            {"name": "Bjorn", "parent": "Relationships", "importance": 0.8},
        ]
        try:
            edges, rescored = extract.reorganize(self.conn, "Alvin")
        finally:
            extract.plan_hierarchy = orig
        self.assertGreater(edges, 0)
        self.assertEqual(rescored, 2)
        self.assertEqual(db.get_node_by_name(self.conn, "Hobbies")["type"], "category")
        self.assertAlmostEqual(db.get_node_by_name(self.conn, "football")["importance"], 0.6, places=2)


class SubgroupTests(BrainTestCase):
    def test_oversized_category_is_split(self):
        cat = db.add_node(self.conn, "Learning", type_="category")
        for i in range(14):
            k = db.add_node(self.conn, f"topic{i}", type_="concept")
            db.add_edge(self.conn, k, cat, "part_of")
        self.conn.commit()
        resp = json.dumps({"groups": [
            {"name": "Group A", "members": [f"topic{i}" for i in range(7)]},
            {"name": "Group B", "members": [f"topic{i}" for i in range(7, 14)]},
        ]})
        orig_gen, orig_key = llm.generate, llm.have_key
        llm.generate = lambda *a, **k: resp
        llm.have_key = lambda: True
        try:
            moved = extract.subgroup_categories(self.conn, threshold=12)
        finally:
            llm.generate, llm.have_key = orig_gen, orig_key
        self.assertEqual(moved, 14)
        ga = db.get_node_by_name(self.conn, "Group A")
        self.assertEqual(ga["type"], "category")
        # Group A sits under Learning; topic0 sits under Group A, not Learning
        ga_parents = {e["target_id"] for e in db.edges_for_node(self.conn, ga["id"])
                      if e["source_id"] == ga["id"] and e["relation"] == "part_of"}
        self.assertIn(cat, ga_parents)  # cat is the id returned by add_node
        t0 = db.get_node_by_name(self.conn, "topic0")
        t0_parents = {e["target_id"] for e in db.edges_for_node(self.conn, t0["id"])
                      if e["source_id"] == t0["id"] and e["relation"] == "part_of"}
        self.assertIn(ga["id"], t0_parents)
        self.assertNotIn(cat, t0_parents)

    def test_small_category_untouched(self):
        cat = db.add_node(self.conn, "Tiny", type_="category")
        for i in range(3):
            db.add_edge(self.conn, db.add_node(self.conn, f"x{i}", type_="concept"), cat, "part_of")
        self.conn.commit()
        orig = llm.have_key
        llm.have_key = lambda: True  # even with a key, below threshold → no LLM, no change
        try:
            self.assertEqual(extract.subgroup_categories(self.conn, threshold=12), 0)
        finally:
            llm.have_key = orig


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

    def test_person_only_has_category_children(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        # LLM (wrongly) attaches a friend and a concept straight to the person
        extracted = {
            "nodes": [
                {"name": "Bjorn", "type": "person", "parent": "Alvin"},
                {"name": "Chess", "type": "concept", "parent": "Alvin"},
            ],
            "edges": [],
        }
        extract.merge_into_db(self.conn, extracted, "src", "raw", user="Alvin")
        alvin = db.get_node_by_name(self.conn, "Alvin")
        # every direct child of the person must be a category
        child_ids = [e["source_id"] for e in db.edges_for_node(self.conn, alvin["id"])
                     if e["target_id"] == alvin["id"] and e["relation"] == "part_of"]
        for cid in child_ids:
            self.assertEqual(db.get_node(self.conn, cid)["type"], "category")
        # Bjorn was re-routed under a category, not under the person
        bjorn = db.get_node_by_name(self.conn, "Bjorn")
        bjorn_parents = [db.get_node(self.conn, e["target_id"])
                         for e in db.edges_for_node(self.conn, bjorn["id"])
                         if e["source_id"] == bjorn["id"] and e["relation"] == "part_of"]
        self.assertTrue(bjorn_parents)
        self.assertTrue(all(p["type"] == "category" for p in bjorn_parents))
        self.assertNotIn(alvin["id"], [p["id"] for p in bjorn_parents])

    def test_plan_hierarchy_parses_llm(self):
        nodes = [{"name": "football", "type": "concept"}, {"name": "Bjorn", "type": "person"}]
        orig = llm.generate
        llm.generate = lambda *a, **k: (
            '{"nodes":[{"name":"football","parent":"Hobbies","importance":0.7},'
            '{"name":"Bjorn","parent":"Relationships","importance":0.8}]}'
        )
        try:
            plan = extract.plan_hierarchy(nodes, "Alvin", [])
        finally:
            llm.generate = orig
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["parent"], "Hobbies")

    def test_reorganize_plan_builds_tree_from_flat(self):
        # simulate: flat nodes already exist; a plan re-parents them under categories
        db.ensure_identity_anchor(self.conn, "Alvin")
        db.add_node(self.conn, "football", type_="concept")
        db.add_node(self.conn, "Bjorn", type_="person")
        self.conn.commit()
        plan = [
            {"name": "football", "parent": "Hobbies", "importance": 0.7},
            {"name": "Bjorn", "parent": "Relationships", "importance": 0.8},
        ]
        extract.merge_into_db(self.conn, {"nodes": plan, "edges": []}, "reorg", "", user="Alvin")
        hobbies = db.get_node_by_name(self.conn, "Hobbies")
        self.assertEqual(hobbies["type"], "category")
        football = db.get_node_by_name(self.conn, "football")
        parents = {e["target_id"] for e in db.edges_for_node(self.conn, football["id"])
                   if e["source_id"] == football["id"] and e["relation"] == "part_of"}
        self.assertIn(hobbies["id"], parents)

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
        # no embeddings + key disabled → keyword miss falls back to whole brain
        orig = llm.have_key
        llm.have_key = lambda: False
        try:
            nodes, fb = graph.collect_context_nodes(self.conn, topic="quantum chromodynamics")
        finally:
            llm.have_key = orig
        self.assertTrue(fb)
        self.assertEqual(set(nodes), {self.a, self.b, self.c})

    def test_context_semantic_seeding_on_keyword_miss(self):
        # embed nodes; a topic with no keyword overlap should seed semantically,
        # NOT dump the whole brain
        db.set_embedding(self.conn, self.a, [1.0, 0.0])   # transformer architectures
        db.set_embedding(self.conn, self.b, [0.9, 0.1])   # attention
        db.set_embedding(self.conn, self.c, [0.0, 1.0])   # football (far)
        self.conn.commit()
        orig_key, orig_embed = llm.have_key, llm.embed
        llm.have_key = lambda: True
        llm.embed = lambda *a, **k: [1.0, 0.0]            # query ~ a/b, not c
        try:
            nodes, fb = graph.collect_context_nodes(self.conn, topic="neural nets", depth=0)
        finally:
            llm.have_key, llm.embed = orig_key, orig_embed
        self.assertFalse(fb)               # semantic seeds found → not a whole-brain dump
        self.assertIn(self.a, nodes)
        self.assertNotIn(self.c, nodes)    # football excluded (below similarity floor)

    def test_category_breakdown(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        root = db.get_node_by_name(self.conn, "Alvin")["id"]
        edu = db.add_node(self.conn, "Education", type_="category")
        hob = db.add_node(self.conn, "Hobbies", type_="category")
        db.add_edge(self.conn, edu, root, "part_of")
        db.add_edge(self.conn, hob, root, "part_of")
        for i in range(3):
            db.add_edge(self.conn, db.add_node(self.conn, f"e{i}", type_="concept"), edu, "part_of")
        db.add_edge(self.conn, db.add_node(self.conn, "f0", type_="concept"), hob, "part_of")
        self.conn.commit()
        bd = graph.category_breakdown(self.conn, "Alvin")
        self.assertEqual(bd[0], ("Education", 3))   # largest first, descendant count
        self.assertEqual(bd[1], ("Hobbies", 1))

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

    def test_answer_question_retrieves_and_answers(self):
        db.add_node(self.conn, "Football", type_="concept", content="Alvin plays football on weekends")
        self.conn.commit()
        captured = {}
        orig = llm.generate
        llm.generate = lambda p, *a, **k: (captured.__setitem__("p", p), "He plays football.")[1]
        try:
            res = graph.answer_question(self.conn, "what sport does he play")
        finally:
            llm.generate = orig
        self.assertEqual(res["answer"], "He plays football.")
        self.assertIn("Football", res["sources"])       # retrieved the right node
        self.assertIn("Football", captured["p"])         # and put it in the prompt

    def test_answer_question_empty_brain(self):
        res = graph.answer_question(self.conn, "anything?")
        self.assertEqual(res["sources"], [])

    def test_answer_question_includes_neighbors(self):
        f = db.add_node(self.conn, "Football", type_="concept", content="a sport Alvin plays")
        b = db.add_node(self.conn, "Bjorn", type_="person", content="Alvin's friend")
        db.add_edge(self.conn, b, f, "relates_to")  # Bjorn connected to Football
        self.conn.commit()
        captured = {}
        orig = llm.generate
        llm.generate = lambda p, *a, **k: (captured.__setitem__("p", p), "ans")[1]
        try:
            res = graph.answer_question(self.conn, "football")
        finally:
            llm.generate = orig
        self.assertIn("Football", res["sources"])        # the match is a source
        self.assertIn("Bjorn", captured["p"])            # neighbor pulled into context

    def test_synthesize_context_is_importance_ordered(self):
        nodes = {
            "1": {"name": "Trivial detail", "type": "concept", "content": "x",
                  "weight": 1.0, "importance": 0.1},
            "2": {"name": "Crucial thing", "type": "concept", "content": "y",
                  "weight": 1.0, "importance": 0.9},
        }
        captured = {}
        orig = llm.generate
        llm.generate = lambda p, *a, **k: (captured.__setitem__("p", p), "DOC")[1]
        try:
            out = graph.synthesize_context(nodes, topic="t")
        finally:
            llm.generate = orig
        self.assertEqual(out, "DOC")
        p = captured["p"]
        self.assertIn("importance", p)                       # importance surfaced
        self.assertLess(p.index("Crucial thing"), p.index("Trivial detail"))  # important first

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


class NodeTypeVocabTests(BrainTestCase):
    def test_normalize_type(self):
        self.assertEqual(db.normalize_type("concept"), "concept")
        self.assertEqual(db.normalize_type("category"), "category")
        self.assertEqual(db.normalize_type("hobby"), "concept")     # invented type → mapped
        self.assertEqual(db.normalize_type("Place"), "fact")
        self.assertEqual(db.normalize_type("company"), "organization")
        self.assertEqual(db.normalize_type("xyzzy"), "concept")     # unknown → default
        self.assertEqual(db.normalize_type(""), "concept")

    def test_add_node_normalizes_type(self):
        nid = db.add_node(self.conn, "Tennis", type_="hobby")
        row = db.get_node(self.conn, nid)
        self.assertEqual(row["type"], "concept")
        self.assertEqual(row["half_life_days"], db.HALF_LIVES["concept"])


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
        orig_embed, orig_key = llm.embed, llm.have_key
        llm.embed = lambda text, *a, **k: [1.0, 0.0] if "rust" in text.lower() else [0.0, 1.0]
        llm.have_key = lambda: True
        try:
            n = extract.embed_nodes(self.conn, [a, b])
        finally:
            llm.embed, llm.have_key = orig_embed, orig_key
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

        orig_embed, orig_key = llm.embed, llm.have_key
        llm.embed = boom
        llm.have_key = lambda: True
        try:
            n = extract.embed_nodes(self.conn, [a, b])  # a skipped, b errors -> swallowed
        finally:
            llm.embed, llm.have_key = orig_embed, orig_key
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
        self.addCleanup(conn.close)
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


class LLMRetryTests(unittest.TestCase):
    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"ok": 1}'

    def _run(self, fake_urlopen):
        import urllib.request as ur
        orig_open, orig_backoff = ur.urlopen, llm.BACKOFF
        ur.urlopen, llm.BACKOFF = fake_urlopen, 0
        try:
            return llm._request(object(), timeout=1)
        finally:
            ur.urlopen, llm.BACKOFF = orig_open, orig_backoff

    def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}
        def flaky(req, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("dropped")  # transient (OSError)
            return self._FakeResp()
        self.assertEqual(self._run(flaky), {"ok": 1})
        self.assertEqual(calls["n"], 2)  # failed once, retried, succeeded

    def test_4xx_fails_fast_without_retry(self):
        import urllib.error as ue
        calls = {"n": 0}
        def bad_key(req, timeout):
            calls["n"] += 1
            raise ue.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad"))
        with self.assertRaises(ue.HTTPError) as cm:
            self._run(bad_key)
        cm.exception.close()  # avoid ResourceWarning on GC
        self.assertEqual(calls["n"], 1)  # 4xx not retried


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
