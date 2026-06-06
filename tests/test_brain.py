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
import brain.db as db
import brain.decay as decay
import brain.graph as graph
import brain.extract as extract
import brain.llm as llm

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
        nid = db.add_node(self.conn, "Ephemeral", type_="event")  # half-life 7d
        self._age(nid, 30)  # 0.5**(30/7) ≈ 0.05 < 0.10
        decay.run_decay(self.conn)
        self.assertEqual(db.get_node(self.conn, nid)["archived"], 1)

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

    def test_context_keyword_path(self):
        nodes, fb = graph.collect_context_nodes(self.conn, topic="transformers", depth=2)
        self.assertFalse(fb)
        self.assertIn(self.a, nodes)

    def test_context_fallback_to_whole_brain(self):
        nodes, fb = graph.collect_context_nodes(self.conn, topic="quantum chromodynamics")
        self.assertTrue(fb)
        self.assertEqual(set(nodes), {self.a, self.b, self.c})

    def test_context_empty_brain(self):
        for n in db.all_nodes(self.conn):
            db.delete_node(self.conn, n["id"])
        self.conn.commit()
        nodes, _ = graph.collect_context_nodes(self.conn, topic="anything")
        self.assertEqual(nodes, {})


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
