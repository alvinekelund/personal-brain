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
        ghost = db.add_node(c, "Ghost", type_="concept"); db.add_edge(c, ghost, edu, "part_of")
        c.execute("DELETE FROM nodes WHERE id = ?", (ghost,))                          # the old non-cascading delete → dangling edge
        # two orgs the vault index maps to one file (title + alias) = one entity; the
        # project on the same file is a different type and must not be paired
        for name, type_ in (("Miracle Consulting Group", "organization"), ("Miracle Oy", "organization"),
                            ("Miracle summer job", "project")):
            nid = db.add_node(c, name, type_=type_); db.add_edge(c, nid, career, "part_of")
            c.execute("UPDATE nodes SET path = ? WHERE id = ?", ("orgs/miracle-consulting-group.md", nid))
        # one course captured under two types — the per-type ratio check never compares them
        for name, type_ in (("AC 215", "concept"), ("AC215", "event")):
            nid = db.add_node(c, name, type_=type_); db.add_edge(c, nid, edu, "part_of")
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
        self.assertIn(("Miracle Consulting Group", "Miracle Oy"), r.duplicates)   # same vault file, same type
        self.assertFalse([p for p in r.duplicates if "Miracle summer job" in p])  # same file, other type: not a dupe
        self.assertIn(("AC 215", "AC215"), r.duplicates)                          # same name, two types
        self.assertGreater(r.missing_embeddings, 0)
        self.assertEqual(r.dangling_edges, 1)
        self.assertIn("1 dangling edge", r.summary())
        self.assertFalse(r.clean)
        self.assertIn("orphan", r.summary())

    def test_repair_makes_a_tree(self):
        self.build_broken()
        fixed = integrity.repair(self.conn, "Alvin")
        self.assertEqual(fixed["dangling"], 1)         # the ghost's edge is gone
        self.assertEqual(integrity.check(self.conn, "Alvin").dangling_edges, 0)
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
