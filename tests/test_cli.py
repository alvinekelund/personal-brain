"""CLI smoke tests: the curation commands register and do what they say, driven
through click's runner against a temp brain (no key, no network). Five commands
(move, rename, retype, subgroup, reindex) shipped on Sep 6 2026 with only a
manual --help check each; a registration error would surface in a terminal, not
in CI."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from click.testing import CliRunner
    import cli as brain_cli
    HAVE_CLICK = True
except ImportError:  # a bare interpreter without the package installed
    HAVE_CLICK = False

import brain.config as config
import brain.db as db

from test_brain import BrainTestCase


@unittest.skipUnless(HAVE_CLICK, "click (and the CLI) not installed — `pip install -e .`")
class CliTests(BrainTestCase):
    def setUp(self):
        super().setUp()
        config.save({"vault_dir": str(self.vault_tmp), "user": "Alvin"})
        db.ensure_identity_anchor(self.conn, "Alvin")
        self.me = db.get_node_by_name(self.conn, "Alvin")["id"]
        self.hobbies = db.add_node(self.conn, "Hobbies", type_="category")
        db.add_edge(self.conn, self.hobbies, self.me, "part_of")
        self.knowledge = db.add_node(self.conn, "Knowledge", type_="category")
        db.add_edge(self.conn, self.knowledge, self.me, "part_of")
        self.padel = db.add_node(self.conn, "Padel", type_="event", content="A racket sport.")
        db.add_edge(self.conn, self.padel, self.knowledge, "part_of")
        self.conn.commit()
        self.runner = CliRunner()

    def run_cli(self, *args):
        return self.runner.invoke(brain_cli.cli, list(args), catch_exceptions=False)

    def parent_of(self, name):
        nid = db.get_node_by_name(self.conn, name)["id"]
        pid = next((e["target_id"] for e in db.edges_for_node(self.conn, nid)
                    if e["source_id"] == nid and e["relation"] == "part_of"), None)
        return db.get_node(self.conn, pid)["name"] if pid else None

    def test_help_lists_the_curation_commands(self):
        out = self.run_cli("--help").output
        for cmd in ("merge", "move", "rename", "retype", "subgroup", "reindex", "repair"):
            self.assertIn(cmd, out)

    def test_move_rename_retype_by_name(self):
        r = self.run_cli("move", "Padel", "Hobbies")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(self.parent_of("Padel"), "Hobbies")
        r = self.run_cli("rename", "Padel", "Padel (racket sport)")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIsNotNone(db.get_node_by_name(self.conn, "Padel (racket sport)"))
        r = self.run_cli("retype", "Padel (racket sport)", "skill")
        self.assertEqual(r.exit_code, 0, r.output)
        n = db.get_node_by_name(self.conn, "Padel (racket sport)")
        self.assertEqual((n["type"], n["half_life_days"]), ("skill", db.HALF_LIVES["skill"]))
        self.assertIn("event → skill", r.output)

    def test_refusals_exit_1_with_the_reason(self):
        r = self.run_cli("move", "Hobbies", "Padel")            # a category under a plain node
        self.assertEqual(r.exit_code, 1)
        self.assertIn("Not moved", r.output)
        r = self.run_cli("rename", "Padel", "Hobbies")          # name already taken
        self.assertEqual(r.exit_code, 1)
        self.assertIn("merge", r.output)
        r = self.run_cli("retype", "Padel", "banana")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("unknown type", r.output)
        r = self.run_cli("move", "Nope", "Hobbies")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("not found", r.output)

    def test_model_commands_refuse_without_a_key(self):
        for args in (("subgroup",), ("reindex",)):
            r = self.run_cli(*args)
            self.assertEqual(r.exit_code, 1, r.output)
            self.assertIn("GEMINI_API_KEY", r.output)

    def test_merge_by_id_or_name(self):
        dup = db.add_node(self.conn, "Padel duplicate", type_="event")
        db.add_edge(self.conn, dup, self.hobbies, "part_of")
        self.conn.commit()
        r = self.run_cli("merge", self.padel, dup)
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIsNone(db.get_node(self.conn, dup))
        self.assertEqual(self.parent_of("Padel"), "Knowledge")   # the survivor keeps its own parent
        dup2 = db.add_node(self.conn, "Padel (dup)", type_="event")
        self.conn.commit()
        r = self.run_cli("merge", "Padel", "Padel (dup)")          # the doctor's hint pastes straight in
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIsNone(db.get_node(self.conn, dup2))
        r = self.run_cli("merge", "Padel", "Padel")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("same node", r.output)

    def test_describe_replaces_content(self):
        r = self.run_cli("describe", "Padel", "A racket sport Alvin plays on Tuesdays.")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(db.get_node_by_name(self.conn, "Padel")["content"], "A racket sport Alvin plays on Tuesdays.")
        r = self.run_cli("describe", "Padel", "   ")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("empty", r.output)

    def test_today_brief_records_its_line(self):
        """The morning brief left no trace (weekly review W36): `today --brief`
        now appends what it produced to brief.log, which the doctor reads."""
        import tempfile
        import brain.doctor as doctor
        orig = doctor.DATA_DIR
        doctor.DATA_DIR = Path(tempfile.mkdtemp())
        try:
            r = self.run_cli("today", "--brief")
            self.assertEqual(r.exit_code, 0, r.output)
            log = (doctor.DATA_DIR / "brief.log").read_text().strip().splitlines()
            self.assertEqual(len(log), 1)
            self.assertTrue(log[0].endswith(r.output.strip()), log[0])
            self.assertRegex(log[0], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
            self.run_cli("today", "--brief")
            self.assertEqual(len((doctor.DATA_DIR / "brief.log").read_text().strip().splitlines()), 2)
        finally:
            doctor.DATA_DIR = orig

    def test_forget_and_reinforce_by_name_with_tree_guards(self):
        self.conn.execute("UPDATE nodes SET weight = 0.4 WHERE id = ?", (self.padel,))
        self.conn.commit()
        r = self.run_cli("reinforce", "Padel")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(db.get_node(self.conn, self.padel)["weight"], 1.0)
        for target, why in (("Alvin", "cannot be archived"), ("Hobbies", "is a category"), ("Nope", "not found")):
            r = self.run_cli("forget", target)
            self.assertEqual(r.exit_code, 1, target)
            self.assertIn(why, r.output)
        child = db.add_node(self.conn, "Padel racket", type_="artifact")
        db.add_edge(self.conn, child, self.padel, "part_of")
        self.conn.commit()
        r = self.run_cli("forget", "Padel")                              # has a child filed under it
        self.assertEqual(r.exit_code, 1)
        self.assertIn("filed under 'Padel'", r.output)
        r = self.run_cli("forget", "Padel racket")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(db.get_node(self.conn, child)["archived"], 1)
        r = self.run_cli("forget", "Padel")                              # child archived: now allowed
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(db.get_node(self.conn, self.padel)["archived"], 1)

    def test_unlink_removes_cross_links_but_never_the_spine(self):
        """Merging 'Boston area' into the residence fact re-pointed its edges,
        leaving 'Alvin's MIT Identity located_at Alvin's Residence' behind with
        no way to remove one edge short of SQL."""
        db.add_edge(self.conn, self.padel, self.hobbies, "relates_to")
        db.add_edge(self.conn, self.hobbies, self.padel, "located_at")
        self.conn.commit()
        r = self.run_cli("unlink", "Padel", "Knowledge")                  # only part_of between them
        self.assertEqual(r.exit_code, 1)
        self.assertIn("brain move", r.output)
        r = self.run_cli("unlink", "Padel", "Hobbies", "--relation", "located_at")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("removed 1 edge(s) (located_at)", r.output)
        left = {e["relation"] for e in db.edges_for_node(self.conn, self.padel)
                if {e["source_id"], e["target_id"]} == {self.padel, self.hobbies}}
        self.assertEqual(left, {"relates_to"})
        r = self.run_cli("unlink", "Padel", "Hobbies")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(self.parent_of("Padel"), "Knowledge")             # the spine is untouched
        r = self.run_cli("unlink", "Padel", "Hobbies")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("No link", r.output)

    def test_semantic_query_shows_paths_and_files_like_keyword_does(self):
        import brain.llm as llm
        self.conn.execute("UPDATE nodes SET path = 'topics/padel.md' WHERE id = ?", (self.padel,))
        db.set_embedding(self.conn, self.padel, [1.0, 0.0])
        self.conn.commit()
        (self.vault_tmp / "topics").mkdir(exist_ok=True)
        (self.vault_tmp / "topics" / "padel.md").write_text("---\ntype: topic\nname: Padel\nupdated: 2026-09-01\n---\n# Padel\n- racket sport\n")
        import brain.index as index
        index.build(self.conn, self.vault_tmp, embed=False)
        orig = llm.embed
        llm.embed = lambda *a, **k: [1.0, 0.0]
        try:
            r = self.run_cli("query", "racket sport", "--semantic")
        finally:
            llm.embed = orig
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("Padel", r.output)
        self.assertIn("→ topics/padel.md", r.output)                       # the node's file, as the keyword path shows it
        self.assertIn("files (the vault is the source of truth)", r.output)
        self.assertIn("topics/padel.md", r.output.split("files (")[1])

    def test_add_shows_where_each_node_was_filed(self):
        """Six dashboard widgets filed under one project on Sep 4 2026 only
        surfaced in a survey two days later: the add output named the nodes but
        not their parents."""
        import json
        import brain.llm as llm
        orig = (llm.have_key, llm.generate, llm.embed)
        responses = iter([json.dumps({"nodes": [{"name": "Padel Tuesdays", "type": "event", "content": "Weekly game.",
                                                 "parent": "Hobbies", "importance": 0.3}], "edges": []}), "{}"])
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: next(responses, "{}")
        llm.embed = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        try:
            r = self.run_cli("add", "Alvin plays padel on Tuesdays.")
        finally:
            llm.have_key, llm.generate, llm.embed = orig
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("[event] Padel Tuesdays → under Hobbies", r.output)
        self.assertRegex(db.get_node_by_name(self.conn, "Padel Tuesdays")["source"], r"^brain add \d{4}-\d{2}-\d{2} \d{2}:\d{2}$")   # provenance by default

    def test_today_snapshots_when_the_newest_backup_is_stale(self):
        import os, tempfile, time
        import brain.doctor as doctor
        orig = doctor.DATA_DIR
        doctor.DATA_DIR = Path(tempfile.mkdtemp())
        try:
            r = self.run_cli("today", "--no-doctor")
            self.assertEqual(r.exit_code, 0, r.output)
            files = sorted((doctor.DATA_DIR / "backups").glob("brain-*.db"))
            self.assertEqual(len(files), 1)                                   # none existed: one taken
            self.assertIn("-auto", files[0].name)
            self.run_cli("today", "--no-doctor")
            self.assertEqual(len(list((doctor.DATA_DIR / "backups").glob("brain-*.db"))), 1)   # fresh: none taken
            old = time.time() - (doctor.BACKUP_MAX_AGE_D + 1) * 86400
            os.utime(files[0], (old, old))
            self.run_cli("today", "--no-doctor")
            self.assertEqual(len(list((doctor.DATA_DIR / "backups").glob("brain-*.db"))), 2)   # stale: another
        finally:
            doctor.DATA_DIR = orig

    def test_backup_snapshots_and_prunes(self):
        import sqlite3, tempfile
        import brain.doctor as doctor
        orig = doctor.DATA_DIR
        doctor.DATA_DIR = Path(tempfile.mkdtemp())
        try:
            r = self.run_cli("backup", "--label", "pre merge!")
            self.assertEqual(r.exit_code, 0, r.output)
            self.assertIn("Backed up to", r.output)
            files = sorted((doctor.DATA_DIR / "backups").glob("brain-*.db"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith("-pre-merge-.db") or "pre-merge" in files[0].name, files[0].name)
            copy = sqlite3.connect(str(files[0]))
            self.assertEqual(copy.execute("SELECT count(*) FROM nodes WHERE name = 'Padel'").fetchone()[0], 1)
            copy.close()
            for _ in range(3):
                self.run_cli("backup", "--keep", "2")
            self.assertEqual(len(list((doctor.DATA_DIR / "backups").glob("brain-*.db"))), 2)
        finally:
            doctor.DATA_DIR = orig

    def test_nothing_ingests_before_setup(self):
        """A throwaway brain on Sep 6 2026: `brain add` before `brain setup` made
        8 orphans and no error. Now it refuses; `setup --name` is scriptable."""
        config.save({"vault_dir": str(self.vault_tmp)})               # no owner
        r = self.run_cli("add", "Alvin plays padel.")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("brain setup", r.output)
        r = self.run_cli("setup", "--name", "Alvin")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(config.get_user(), "Alvin")
        r = self.run_cli("setup", "--name", "  ")
        self.assertEqual(r.exit_code, 1)

    def test_importance_by_name_with_bounds(self):
        r = self.run_cli("importance", "Padel", "0.2")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("0.50 → 0.20", r.output)
        self.assertAlmostEqual(db.get_node(self.conn, self.padel)["importance"], 0.2)
        r = self.run_cli("importance", "Padel", "7")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("between 0 and 1", r.output)
        r = self.run_cli("importance", "Nope", "0.5")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("not found", r.output)

    def test_stale_lists_old_plan_tense_claims(self):
        import time
        r = self.run_cli("stale")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("No plan-tense claim older than 30 days", r.output)
        self.conn.execute("UPDATE nodes SET content = 'Alvin plans to take up padel.', created_at = ? WHERE id = ?",
                          (time.time() - 45 * 86400, self.padel))
        self.conn.commit()
        r = self.run_cli("stale")
        self.assertIn("45d  Padel  'plans to'", r.output)
        self.assertIn("brain describe", r.output)
        self.assertIn("No plan-tense claim older than 60 days", self.run_cli("stale", "--days", "60").output)

    def test_curation_commands_commit_their_views(self):
        """Regression (Sep 6 2026): merge/move/rename/... re-rendered DIGEST.md
        and graph/ but left them uncommitted, so nine curation commands in a row
        left the vault dirty for the doctor to complain about. Ingest already
        commits its own writes (scoped); the curation commands now do the same."""
        import subprocess
        for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@test"],
                    ["git", "config", "user.name", "t"]):
            subprocess.run(cmd, cwd=self.vault_tmp, check=True, capture_output=True)
        (self.vault_tmp / "areas.md").write_text("curated, mid-edit")

        def porcelain():
            r = subprocess.run(["git", "status", "--porcelain"], cwd=self.vault_tmp, capture_output=True, text=True)
            return [l for l in r.stdout.splitlines() if l.strip()]

        def last():
            return subprocess.run(["git", "log", "-1", "--format=%s"], cwd=self.vault_tmp,
                                  capture_output=True, text=True).stdout.strip()
        r = self.run_cli("move", "Padel", "Hobbies")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(porcelain(), ["?? areas.md"])                       # the mid-edit file is left alone
        self.assertEqual(last(), "move: Padel → under Hobbies")
        self.assertTrue((self.vault_tmp / "graph" / "hobbies.md").is_file())
        self.run_cli("rename", "Padel", "Padel (racket sport)")
        self.assertEqual(last(), "rename: Padel → Padel (racket sport)")
        self.run_cli("forget", "Padel (racket sport)")
        self.assertEqual(last(), "forget: Padel (racket sport)")
        self.assertEqual(porcelain(), ["?? areas.md"])

