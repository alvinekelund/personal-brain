"""Tests for brain/integrity.py — the tree invariants, checked and repaired."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.db as db
import brain.integrity as integrity

from test_brain import BrainTestCase


class IntegrityTests(BrainTestCase):
    def build_broken(self):
        c = self.conn
        db.ensure_identity_anchor(c, "Alvin")
        me = db.get_node_by_name(c, "Alvin")["id"]
        edu = db.add_node(c, "Education", type_="category"); db.add_edge(c, edu, me, "part_of")
        career = db.add_node(c, "Career", type_="category"); db.add_edge(c, career, me, "part_of")
        loose_cat = db.add_node(c, "Hobbies", type_="category")                       # unrooted
        fact = db.add_node(c, "MIT identity", type_="fact"); db.add_edge(c, fact, edu, "part_of")
        badcat = db.add_node(c, "Life Events", type_="category"); db.add_edge(c, badcat, fact, "part_of")  # category under a fact
        harvard = db.add_node(c, "Harvard", type_="organization"); db.add_edge(c, harvard, edu, "part_of")
        ds = db.add_node(c, "Data Science", type_="concept")
        db.add_edge(c, ds, edu, "part_of"); db.add_edge(c, ds, career, "part_of")       # multi-parent (two categories)
        prog = db.add_node(c, "DS Program", type_="project")
        db.add_edge(c, prog, edu, "part_of"); db.add_edge(c, prog, harvard, "part_of")  # multi-parent (one specific)
        orphan = db.add_node(c, "Boston", type_="concept")                             # orphan
        under = db.add_node(c, "Resume", type_="artifact"); db.add_edge(c, under, me, "part_of")  # directly under person
        a = db.add_node(c, "Triathlon Training", type_="project"); b = db.add_node(c, "Triathlon", type_="event")
        db.add_edge(c, a, b, "part_of"); db.add_edge(c, b, a, "part_of")               # cycle
        db.add_node(c, "Legacy", type_="task")
        db.add_node(c, "Anna Houstecka", type_="person"); db.add_node(c, "Alvin's Girlfriend", type_="person")
        db.add_node(c, "Heli", type_="person"); db.add_node(c, "Heli Korhonen", type_="person")
        c.commit()

    def test_check_reports_every_problem(self):
        self.build_broken()
        r = integrity.check(self.conn, "Alvin")
        self.assertIn("Boston", r.orphans)
        self.assertEqual(sorted(n for n, _ in r.multi_parent), ["DS Program", "Data Science"])
        self.assertIn("Hobbies", r.unrooted_categories)
        self.assertEqual([n for n, _ in r.category_bad_parent], ["Life Events"])
        self.assertEqual(r.under_identity, ["Resume"])
        self.assertEqual(len(r.cycles), 1)
        self.assertEqual(set(r.cycles[0]), {"Triathlon Training", "Triathlon"})
        self.assertEqual(r.legacy_tasks, ["Legacy"])
        self.assertIn(("Heli", "Heli Korhonen"), r.duplicates)
        self.assertGreater(r.missing_embeddings, 0)
        self.assertFalse(r.clean)
        self.assertIn("orphan", r.summary())

    def test_repair_makes_a_tree(self):
        self.build_broken()
        fixed = integrity.repair(self.conn, "Alvin")
        self.assertEqual(fixed["orphans"], 6)          # Boston + the five parentless people/task nodes
        self.assertEqual(fixed["multi_parent"], 2)
        self.assertEqual(fixed["under_identity"], 1)
        self.assertGreaterEqual(fixed["categories"], 2)      # Hobbies rooted, Life Events re-rooted
        self.assertEqual(fixed["cycles"], 1)
        r = integrity.check(self.conn, "Alvin")
        self.assertEqual(r.structural, 0, r.summary())
        c = self.conn
        def parent_names(name):
            nid = db.get_node_by_name(c, name)["id"]
            return sorted(db.get_node(c, e["target_id"])["name"] for e in db.edges_for_node(c, nid)
                          if e["source_id"] == nid and e["relation"] == "part_of")
        self.assertEqual(parent_names("DS Program"), ["Harvard"])        # the specific parent wins
        self.assertEqual(len(parent_names("Data Science")), 1)          # two categories → one kept
        self.assertEqual(parent_names("Hobbies"), ["Alvin"])
        self.assertEqual(parent_names("Life Events"), ["Alvin"])
        self.assertEqual(parent_names("Resume"), ["Artifacts"])          # fallback category, rooted
        self.assertEqual(parent_names("Artifacts"), ["Alvin"])
        self.assertEqual(parent_names("Boston"), ["Knowledge"])
        self.assertEqual(parent_names("Heli"), ["Relationships"])
        # cycle cut: the bigger subtree (Training, which held the race) keeps its member
        self.assertEqual(parent_names("Triathlon"), ["Triathlon Training"])
        self.assertNotEqual(parent_names("Triathlon Training"), ["Triathlon"])
        # repair never deletes nodes; duplicates and legacy tasks are still reported
        self.assertIsNotNone(db.get_node_by_name(c, "Legacy"))
        self.assertTrue(r.duplicates)
        self.assertEqual(integrity.repair(self.conn, "Alvin")["multi_parent"], 0)   # idempotent

    def test_clean_graph(self):
        db.ensure_identity_anchor(self.conn, "Alvin")
        me = db.get_node_by_name(self.conn, "Alvin")["id"]
        edu = db.add_node(self.conn, "Education", type_="category"); db.add_edge(self.conn, edu, me, "part_of")
        h = db.add_node(self.conn, "Harvard", type_="organization"); db.add_edge(self.conn, h, edu, "part_of")
        db.set_embedding(self.conn, h, [0.1, 0.2]); db.set_embedding(self.conn, edu, [0.1, 0.2])
        db.set_embedding(self.conn, me, [0.1, 0.2])
        self.conn.commit()
        r = integrity.check(self.conn, "Alvin")
        self.assertTrue(r.clean, r.summary())
        self.assertEqual(r.summary(), "tree intact")

    def test_norm_ignores_possessives(self):
        self.assertEqual(integrity._norm("Alvin's Girlfriend"), "girlfriend")
        self.assertEqual(integrity._norm("The MIT-identity!"), "mit identity")


if __name__ == "__main__":
    unittest.main()
