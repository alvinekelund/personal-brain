"""Tests for the vault index (brain/index.py) — the graph as a retrieval layer
over the directory (D-014). Temp DB, temp vault, no key, no network: the LLM
boundary is mocked or disabled in every test."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.config as config
import brain.db as db
import brain.graph as graph
import brain.index as index
import brain.llm as llm


def _disabled(what):
    return lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"{what} is disabled in tests; mock it"))


def w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


STAT211 = """---
type: course
name: STAT 211 Statistical Inference I
code: STAT 211
aliases: [Stat 211, statistical inference, Austern]
updated: 2026-09-03
---
# STAT 211, Statistical Inference I
- 4-credit PhD-required inference course, co-taught by Morgane Austern.
- Sections Tue and Wed 4-5 pm, office hours 5-6 pm, Maxwell Dworkin 109, from Tue Sep 8.
- Cut rule by Sep 21: if 211 is fine keep it; if it is a wall, drop it for 9.522.
"""
HELI = """---
person: heli
name: Heli Helskyaho
role: Alvin's boss, Group CEO of Miracle Consulting Group
aliases: [Heli]
updated: 2026-09-03
---
# Heli Helskyaho
- Group CEO of Miracle Consulting Group; Alvin's boss since May 2026.
"""
ATHLETE = """---
type: profile
name: The athlete
aliases: [triathlon, Barcelona]
updated: 2026-09-03
---
# The athlete
- Sub-10 goal at Barcelona on Oct 4; swim, bike and run blocks; sleep is the limiter.
"""
EDUCATION = """---
type: profile
name: Education
aliases: [studies, transcript]
updated: 2026-09-03
---
# Education
- Harvard SM in Data Science; Aalto B.Sc. in 2.5 years; NUS exchange. See `courses/stat-211.md`.
"""
HUB = """---
type: hub
name: Alvin Ekelund
aliases: [Alvin, hub]
updated: 2026-09-03
---
# ALVIN.md — the hub
- The person: `profile/education.md`, `profile/athlete.md`. People: `people/heli.md`.
"""


class IndexTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "vault"
        self.root.mkdir()
        self._orig_db = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmp, "brain.db")
        self._orig_cfg = config.CONFIG_PATH
        config.CONFIG_PATH = self.tmp / "config.json"
        config.save({"vault_dir": str(self.root)})
        self._orig_llm = (llm.have_key, llm.generate, llm.embed)
        llm.have_key = lambda: False
        llm.generate = _disabled("llm.generate")
        llm.embed = _disabled("llm.embed")
        self.conn = db.connect()
        w(self.root / "ALVIN.md", HUB)
        w(self.root / "profile/education.md", EDUCATION)
        w(self.root / "profile/athlete.md", ATHLETE)
        w(self.root / "courses/stat-211.md", STAT211)
        w(self.root / "people/heli.md", HELI)
        w(self.root / "people/README.md", "# People\nOne file per person.\n")
        w(self.root / "docs/cv.md", "---\ntype: doc\nname: CV\nkind: cv\n---\n# CV\nAalto, Cadentia, Miracle.\n")
        w(self.root / "log/2026-09-01.md", "- Bain applied at the deadline.\n")
        w(self.root / "topics/loose.md", "# Loose note\nNo front-matter here about grokking.\n")
        # never indexed: generated views, ledgers, the protocol README
        w(self.root / "NOW.md", "# NOW\ngenerated\n")
        w(self.root / "DIGEST.md", "# Digest\n")
        w(self.root / "LOOPS.md", "# LOOPS\n- [ ] L-001 · x\n")
        w(self.root / "DECISIONS.md", "# DECISIONS\n")
        w(self.root / "README.md", "# The vault\n")
        w(self.root / "graph/education.md", "# Education\ngenerated mirror of the graph\n")

    def tearDown(self):
        self.conn.close()
        db.DB_PATH = self._orig_db
        config.CONFIG_PATH = self._orig_cfg
        llm.have_key, llm.generate, llm.embed = self._orig_llm

    def paths(self):
        return {r[0] for r in self.conn.execute("SELECT path FROM vault_files")}


class BuildTests(IndexTestCase):
    def test_walks_the_vault_and_skips_generated_and_ledgers(self):
        s = index.build(self.conn, self.root, embed=False)
        self.assertEqual(s["files"], 9)
        self.assertEqual(s["added"], 9)
        self.assertEqual(self.paths(), {"ALVIN.md", "profile/education.md", "profile/athlete.md",
                                        "courses/stat-211.md", "people/heli.md", "people/README.md",
                                        "docs/cv.md", "log/2026-09-01.md", "topics/loose.md"})
        kinds = dict(self.conn.execute("SELECT path, kind FROM vault_files"))
        self.assertEqual(kinds["ALVIN.md"], "hub")
        self.assertEqual(kinds["courses/stat-211.md"], "course")
        self.assertEqual(kinds["people/heli.md"], "person")        # inferred from the shelf
        self.assertEqual(kinds["people/README.md"], "readme")
        self.assertEqual(kinds["log/2026-09-01.md"], "log")
        self.assertEqual(kinds["topics/loose.md"], "topic")
        titles = dict(self.conn.execute("SELECT path, title FROM vault_files"))
        self.assertEqual(titles["courses/stat-211.md"], "STAT 211 Statistical Inference I")   # front-matter name
        self.assertEqual(titles["topics/loose.md"], "Loose note")                              # first heading
        self.assertEqual(s["no_frontmatter"], ["topics/loose.md"])                             # logs/readmes exempt
        aliases = json.loads(self.conn.execute(
            "SELECT aliases FROM vault_files WHERE path='courses/stat-211.md'").fetchone()[0])
        self.assertIn("STAT 211", aliases)                                                     # `code:` becomes an alias
        self.assertEqual(s["embedded"], 0)                                                     # no key, no network

    def test_incremental_and_idempotent(self):
        index.build(self.conn, self.root, embed=False)
        s = index.build(self.conn, self.root, embed=False)
        self.assertEqual((s["added"], s["updated"], s["removed"], s["unchanged"]), (0, 0, 0, 9))
        w(self.root / "people/heli.md", HELI + "- Based in Helsinki.\n")
        (self.root / "docs/cv.md").unlink()
        w(self.root / "orgs/miracle.md", "---\ntype: org\nname: Miracle Consulting Group\n---\n# Miracle\n")
        s = index.build(self.conn, self.root, embed=False)
        self.assertEqual((s["added"], s["updated"], s["removed"], s["unchanged"]), (1, 1, 1, 7))
        self.assertNotIn("docs/cv.md", self.paths())
        self.assertIn("orgs/miracle.md", self.paths())

    def test_file_links_are_recorded(self):
        index.build(self.conn, self.root, embed=False)
        links = {(r[0], r[1]) for r in self.conn.execute("SELECT path, target FROM vault_file_links")}
        self.assertIn(("ALVIN.md", "profile/education.md"), links)
        self.assertIn(("ALVIN.md", "people/heli.md"), links)
        self.assertIn(("profile/education.md", "courses/stat-211.md"), links)

    def test_nodes_link_to_files_and_get_paths(self):
        heli = db.add_node(self.conn, "Heli", type_="person", content="Alvin's boss.")
        stat = db.add_node(self.conn, "STAT 211", type_="concept", content="Inference course.")
        other = db.add_node(self.conn, "Bain & Company", type_="organization")
        self.conn.commit()
        s = index.build(self.conn, self.root, embed=False)
        self.assertEqual(s["node_links"], 2)
        rows = {(r[0], r[1], r[2]) for r in self.conn.execute("SELECT path, node_id, how FROM vault_file_nodes")}
        self.assertIn(("people/heli.md", heli, "alias"), rows)           # alias "Heli" ↔ node "Heli"
        self.assertIn(("courses/stat-211.md", stat, "alias"), rows)      # code "STAT 211" ↔ node "STAT 211"
        self.assertEqual(db.get_node(self.conn, heli)["path"], "people/heli.md")
        self.assertEqual(db.get_node(self.conn, stat)["path"], "courses/stat-211.md")
        self.assertIsNone(db.get_node(self.conn, other)["path"])
        # re-indexing after the node is renamed drops the stale stamp instead of keeping it forever
        self.conn.execute("UPDATE nodes SET name='Heli H.' WHERE id=?", (heli,))
        self.conn.commit()
        index.build(self.conn, self.root, embed=False)
        self.assertIsNone(db.get_node(self.conn, heli)["path"])

    def test_school_prefixed_node_links_to_the_course_file(self):
        """The graph names courses "MIT 9.522" / "Harvard STAT 211" while a
        course file's alias is the bare code — five real nodes were unlinked
        that way on Sep 6 2026."""
        w(self.root / "courses/mit-9-522.md",
          "---\ntype: course\nname: MIT 9.522 Statistical Reinforcement Learning (Rakhlin)\ncode: 9.522\n"
          "aliases: [9.522, Statistical RL]\nupdated: 2026-09-04\n---\n# 9.522\n- Petition approved Sep 2.\n")
        node = db.add_node(self.conn, "MIT 9.522", type_="concept", content="Fourth-seat candidate.")
        stat = db.add_node(self.conn, "Harvard STAT 211", type_="concept")
        self.conn.commit()
        index.build(self.conn, self.root, embed=False)
        self.assertEqual(db.get_node(self.conn, node)["path"], "courses/mit-9-522.md")
        self.assertEqual(db.get_node(self.conn, stat)["path"], "courses/stat-211.md")

    def test_status_reports_new_stale_and_removed(self):
        s = index.status(self.conn, self.root)
        self.assertEqual(s["indexed"], 0)
        self.assertEqual(s["on_disk"], 9)
        index.build(self.conn, self.root, embed=False)
        s = index.status(self.conn, self.root)
        self.assertEqual((s["new"], s["stale"], s["removed"]), ([], [], []))
        w(self.root / "people/heli.md", HELI + "- changed\n")
        os.utime(self.root / "people/heli.md", (time.time() + 5, time.time() + 5))
        (self.root / "docs/cv.md").unlink()
        w(self.root / "orgs/new.md", "# New\n")
        s = index.status(self.conn, self.root)
        self.assertEqual(s["stale"], ["people/heli.md"])
        self.assertEqual(s["removed"], ["docs/cv.md"])
        self.assertEqual(s["new"], ["orgs/new.md"])


class SearchTests(IndexTestCase):
    def setUp(self):
        super().setUp()
        index.build(self.conn, self.root, embed=False)

    def test_title_and_alias_matches_rank_first(self):
        hits = index.search(self.conn, "STAT 211 sections")
        self.assertEqual(hits[0]["path"], "courses/stat-211.md")
        self.assertIn("title match", hits[0]["why"][0])
        self.assertEqual(index.search(self.conn, "Heli")[0]["path"], "people/heli.md")
        self.assertEqual(index.search(self.conn, "transcript")[0]["path"], "profile/education.md")  # alias only

    def test_body_matches_and_no_match(self):
        hits = index.search(self.conn, "Maxwell Dworkin")
        self.assertEqual(hits[0]["path"], "courses/stat-211.md")
        self.assertTrue(hits[0]["why"][0].startswith("body match"))
        self.assertEqual(index.search(self.conn, "quantum chromodynamics"), [])

    def test_graph_seed_nodes_pull_their_files_up(self):
        stat = db.add_node(self.conn, "STAT 211", type_="concept", content="Inference.")
        self.conn.commit()
        index.build(self.conn, self.root, embed=False)
        hits = index.search(self.conn, "inference", seed_node_ids=[stat])
        self.assertEqual(hits[0]["path"], "courses/stat-211.md")
        self.assertIn("node: STAT 211", hits[0]["why"])

    def test_one_hop_of_file_links_is_followed(self):
        hits = index.search(self.conn, "Alvin hub", k=6)
        self.assertEqual(hits[0]["path"], "ALVIN.md")
        by_path = {h["path"]: h for h in hits}
        self.assertIn("people/heli.md", by_path)
        self.assertIn("linked from ALVIN.md", by_path["people/heli.md"]["why"])

    def test_who_questions_surface_people_files(self):
        """"Who is Alvin's boss at Miracle?" must reach people/heli.md although
        the org file out-scores it on plain keyword hits (a real under-answer:
        "people met at Harvard" returned org and profile files only)."""
        w(self.root / "orgs/miracle.md",
          "---\ntype: org\nname: Miracle Consulting Group\naliases: [Miracle]\nupdated: 2026-09-04\n---\n"
          "# Miracle\n- Consulting group in Finland; Alvin's employer; contracts, Siemens.\n")
        index.build(self.conn, self.root, embed=False)
        plain = index.search(self.conn, "Miracle contracts")
        self.assertEqual(plain[0]["path"], "orgs/miracle.md")             # no boost without a who-question
        who = index.search(self.conn, "who is Alvin's boss at Miracle?")
        self.assertEqual(who[0]["path"], "people/heli.md")
        self.assertIn("who-question: person file", who[0]["why"])

    def test_semantic_ranking_with_fake_embeddings(self):
        """No keyword overlap between 'ironman' and the athlete file — the (fake,
        deterministic) embedding is what finds it. Nothing leaves the process."""
        def fake_embed(text):
            t = text.lower()
            return [1.0 if any(k in t for k in ("ironman", "triathlon", "swim")) else 0.0,
                    1.0 if any(k in t for k in ("stat", "inference")) else 0.0,
                    1.0]
        llm.have_key = lambda: True
        llm.embed = fake_embed
        s = index.build(self.conn, self.root, embed=True)
        self.assertEqual(s["embedded"], 9)
        hits = index.search(self.conn, "ironman", query_vector=fake_embed("ironman"))
        self.assertEqual(hits[0]["path"], "profile/athlete.md")
        self.assertTrue(any(w.startswith("semantic") for w in hits[0]["why"]))
        # a second build re-embeds only what changed
        s = index.build(self.conn, self.root, embed=True)
        self.assertEqual(s["embedded"], 0)

    def test_excerpt_reads_disk_and_prefers_matching_lines(self):
        long_body = "---\nname: Long\n---\n# Long file\n" + "\n".join(
            f"- filler line {i} about nothing in particular" for i in range(120)
        ) + "\n- The office hours are in Maxwell Dworkin 109 on Wednesdays.\n"
        w(self.root / "docs/long.md", long_body)
        ex = index.excerpt(self.root, "docs/long.md", "office hours Wednesdays", max_chars=900)
        self.assertLessEqual(len(ex), 900)
        self.assertTrue(ex.startswith("# Long file"))
        self.assertIn("Maxwell Dworkin 109 on Wednesdays", ex)
        self.assertEqual(index.excerpt(self.root, "docs/missing.md"), "")


class AnswerTests(IndexTestCase):
    def test_answer_question_reads_the_files_first(self):
        index.build(self.conn, self.root, embed=False)
        seen = {}
        llm.generate = lambda prompt, *a, **k: seen.setdefault("prompt", prompt) and "Tue and Wed 4-5 pm (courses/stat-211.md)."
        res = graph.answer_question(self.conn, "when are the STAT 211 sections?")
        self.assertEqual(res["files"][0], "courses/stat-211.md")
        self.assertIn("### courses/stat-211.md", seen["prompt"])
        self.assertIn("Maxwell Dworkin 109", seen["prompt"])            # excerpt read from disk
        self.assertIn("Files (the vault is the source of truth", seen["prompt"])
        self.assertIn("courses/stat-211.md", res["sources"])
        self.assertTrue(res["answer"].startswith("Tue and Wed"))

    def test_files_alone_are_enough_to_answer(self):
        """No graph nodes, no ledgers — the directory still answers (D-014)."""
        index.build(self.conn, self.root, embed=False)
        llm.generate = lambda *a, **k: "Group CEO of Miracle."
        res = graph.answer_question(self.conn, "who is Heli?")
        self.assertEqual(res["files"], ["people/heli.md"] + [f for f in res["files"] if f != "people/heli.md"])
        self.assertEqual(res["sources"][0], "people/heli.md")

    def test_unindexed_vault_answers_from_graph_as_before(self):
        db.add_node(self.conn, "Heli", type_="person", content="Alvin's boss.")
        self.conn.commit()
        llm.generate = lambda *a, **k: "Alvin's boss."
        res = graph.answer_question(self.conn, "who is Heli?")
        self.assertEqual(res["files"], [])
        self.assertEqual(res["sources"], ["Heli"])

    def test_nothing_known_at_all(self):
        res = graph.answer_question(self.conn, "who is Heli?")
        self.assertEqual(res, {"answer": "I don't have anything on that yet.", "sources": [], "files": []})


if __name__ == "__main__":
    unittest.main()
