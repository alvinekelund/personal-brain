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
        # a subsidiary filed under the same org file but hanging *under* the org in
        # the tree is a child, not a duplicate ("Harvard" ~ "Harvard SEAS" false positive)
        mcg = db.get_node_by_name(c, "Miracle Consulting Group")["id"]
        sea = db.add_node(c, "Miracle SEA", type_="organization"); db.add_edge(c, sea, mcg, "part_of")
        c.execute("UPDATE nodes SET path = ? WHERE id = ?", ("orgs/miracle-consulting-group.md", sea))
        # a sub-category under a category is legitimate structure (subgroup_categories makes them)
        sub = db.add_node(c, "Courses", type_="category"); db.add_edge(c, sub, edu, "part_of")
        # one course captured under two types — the per-type ratio check never compares them
        for name, type_ in (("AC 215", "concept"), ("AC215", "event")):
            nid = db.add_node(c, name, type_=type_); db.add_edge(c, nid, edu, "part_of")
        # an event wearing its sponsor list as children: a flat list, not structure
        hack = db.add_node(c, "HackMIT", type_="event"); db.add_edge(c, hack, career, "part_of")
        for name in ("Long Lake", "ASUS", "Cursor", "Ramp"):
            nid = db.add_node(c, name, type_="organization"); db.add_edge(c, nid, hack, "part_of")
        # an empty template area and a one-node area beside the broad ones: thin
        health = db.add_node(c, "Health", type_="category"); db.add_edge(c, health, me, "part_of")
        fam = db.add_node(c, "Family", type_="category"); db.add_edge(c, fam, me, "part_of")
        gp = db.add_node(c, "Alvin's Grandfather", type_="person"); db.add_edge(c, gp, fam, "part_of")
        c.commit()

    def test_check_reports_every_problem(self):
        self.build_broken()
        r = integrity.check(self.conn, "Alvin")
        self.assertIn("Boston", r.orphans)
        self.assertEqual(sorted(n for n, _ in r.multi_parent), ["DS Program", "Data Science"])
        self.assertIn("Hobbies", r.unrooted_categories)
        self.assertEqual([n for n, _ in r.category_bad_parent], ["Life Events"])   # "Courses" under Education is fine
        self.assertEqual(r.under_identity, ["Resume"])
        self.assertEqual(len(r.cycles), 1)
        self.assertEqual(set(r.cycles[0]), {"Triathlon Training", "Triathlon"})
        self.assertEqual(r.legacy_tasks, ["Legacy"])
        self.assertIn(("Heli", "Heli Korhonen"), r.duplicates)
        self.assertIn(("Miracle Consulting Group", "Miracle Oy"), r.duplicates)   # same vault file, same type
        self.assertFalse([p for p in r.duplicates if "Miracle summer job" in p])  # same file, other type: not a dupe
        self.assertFalse([p for p in r.duplicates if set(p) == {"Miracle Consulting Group", "Miracle SEA"}])  # parent/child
        self.assertIn(("AC 215", "AC215"), r.duplicates)                          # same name, two types
        self.assertEqual(r.oversized, [])                                          # default threshold (12) not hit
        r3 = integrity.check(self.conn, "Alvin", oversized_threshold=3)
        self.assertTrue(r3.oversized and r3.oversized[0][0] == "Education", r3.oversized)
        self.assertIn("(brain subgroup)", r3.summary())
        self.assertFalse(r3.clean)
        self.assertEqual(r.flat_lists, [])
        self.assertEqual(r3.flat_lists, [("HackMIT", 4)])                          # not in oversized: subgroup can't split it
        self.assertNotIn("HackMIT", [n for n, _ in r3.oversized])
        self.assertIn("flat list(s): HackMIT (4) (one fact naming the list", r3.summary())
        self.assertEqual(r.thin_areas, [("Health", 0), ("Family", 1)])          # Education/Career are broad; Hobbies is unrooted
        self.assertIn("thin area(s): Health (0), Family (1)", r.summary())
        self.assertIn("brain move <area> <broader>", r.summary())
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
        self.assertEqual(r.thin_areas, [])        # one area with one node: a young brain, not sprawl

    def test_stale_claims_are_old_plan_tense_content(self):
        """A June mail backfill wrote "Alvin plans to relocate to Boston" and
        "plans to pursue studies" at Harvard; 91 days later, enrolled and living
        in Cambridge, the graph still said so. Old plan-tense content is listed,
        oldest first; fresh plans and dated ("as of") plans are not."""
        import time
        c = self.conn
        db.ensure_identity_anchor(c, "Alvin")
        now = time.time()
        old = db.add_node(c, "Move to Boston", type_="event", content="Alvin plans to relocate to Boston.")
        older = db.add_node(c, "Harvard", type_="organization", content="A university where Alvin plans to pursue studies.")
        dated = db.add_node(c, "IRONMAN Barcelona", type_="event", content="As of June 2026 Alvin plans to race sub-10.")
        fresh = db.add_node(c, "Apple Cash", type_="concept", content="Alvin is currently trying to set it up.")
        done = db.add_node(c, "AC 215", type_="concept", content="Alvin took AC 215 in fall 2026.")
        cat = db.add_node(c, "Plans", type_="category", content="Things Alvin plans to do.")
        for nid, age in ((old, 91), (older, 95), (dated, 95), (fresh, 3), (done, 95), (cat, 95)):
            c.execute("UPDATE nodes SET created_at = ? WHERE id = ?", (now - age * 86400, nid))
        c.commit()
        rows = integrity.stale_claims(c, now=now)
        self.assertEqual(rows, [("Harvard", "plans to", 95), ("Move to Boston", "plans to", 91)])
        self.assertEqual(integrity.stale_claims(c, days=2, now=now)[-1][0], "Apple Cash")   # a shorter window catches the fresh one
        db.set_content(c, old, "Alvin moved to Cambridge, MA on 24 Aug 2026 for the Harvard SM in Data Science.")
        c.commit()
        self.assertEqual([n for n, _, _ in integrity.stale_claims(c, now=now)], ["Harvard"])  # a restatement clears it

    def test_norm_ignores_possessives(self):
        self.assertEqual(integrity._norm("Alvin's Girlfriend"), "girlfriend")
        self.assertEqual(integrity._norm("The MIT-identity!"), "mit identity")


if __name__ == "__main__":
    unittest.main()
