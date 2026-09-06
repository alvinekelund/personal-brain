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
