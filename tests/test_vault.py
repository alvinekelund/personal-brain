"""Tests for the markdown vault (brain/vault.py) — the brain's file layer.

Rendering is deterministic and LLM-free, so these run hermetically like the
rest of the suite. Every render targets a temp dir via `dest`; the real vault
directory and real config are never touched.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.config as config
import brain.db as db
import brain.vault as vault

from test_brain import BrainTestCase


class VaultTestCase(BrainTestCase):
    """BrainTestCase + a temp vault dir + config isolated from the real file."""

    def setUp(self):
        super().setUp()
        self.vault_root = Path(tempfile.mkdtemp())
        self._orig_config_load = config.load
        config.load = lambda: {}

    def tearDown(self):
        config.load = self._orig_config_load
        super().tearDown()

    def _build_brain(self):
        """Alvin → Education → {Harvard, Thesis}; Boston loose; Harvard↔Boston."""
        db.ensure_identity_anchor(self.conn, "Alvin")
        ident = db.get_node_by_name(self.conn, "Alvin")
        cat = db.add_node(self.conn, "Education", type_="category", importance=1.0)
        db.add_edge(self.conn, cat, ident["id"], "part_of")
        har = db.add_node(self.conn, "Harvard", type_="organization",
                          content="SM in Data Science", importance=0.9)
        db.add_edge(self.conn, har, cat, "part_of")
        thesis = db.add_node(self.conn, "Thesis", type_="project", importance=0.6)
        db.add_edge(self.conn, thesis, har, "part_of")
        boston = db.add_node(self.conn, "Boston", type_="concept", importance=0.3)
        db.add_edge(self.conn, har, boston, "relates_to")
        self.conn.commit()
        return {"category": cat, "harvard": har, "boston": boston}


class SlugifyTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(vault.slugify("Life Events"), "life-events")

    def test_collapses_junk(self):
        self.assertEqual(vault.slugify("A  --  B!"), "a-b")

    def test_never_empty(self):
        self.assertEqual(vault.slugify("!!!"), "unnamed")


class RenderTests(VaultTestCase):
    def test_empty_brain_renders_index_and_digest(self):
        paths = vault.render(self.conn, "Alvin", dest=self.vault_root)
        names = {p.name for p in paths}
        self.assertIn("DIGEST.md", names)
        self.assertIn("README.md", names)
        self.assertTrue((self.vault_root / "graph" / "README.md").exists())

    def test_category_file_holds_subtree_and_cross_links(self):
        self._build_brain()
        vault.render(self.conn, "Alvin", dest=self.vault_root)
        text = (self.vault_root / "graph" / "education.md").read_text()
        self.assertIn("**Harvard**", text)
        self.assertIn("SM in Data Science", text)
        self.assertIn("**Thesis**", text)          # nested child included
        self.assertIn("relates_to", text)          # cross-link rendered
        self.assertIn("Boston", text)

    def test_loose_nodes_get_their_own_file(self):
        self._build_brain()
        vault.render(self.conn, "Alvin", dest=self.vault_root)
        loose = (self.vault_root / "graph" / "loose-ends.md").read_text()
        self.assertIn("Boston", loose)             # related-to but not part_of anything
        index = (self.vault_root / "graph" / "README.md").read_text()
        self.assertIn("loose-ends.md", index)

    def test_digest_lists_open_loops_from_ledger_not_task_nodes(self):
        import datetime
        import brain.loops as loops
        db.add_node(self.conn, "stale task node", type_="task", importance=0.8)
        self.conn.commit()
        loops.add(self.vault_root, "File the AM 207 petition", "2026-09-09", "alvin", "harvard",
                  "my.harvard step", today=datetime.date(2026, 9, 1), commit=False)
        vault.render(self.conn, "", dest=self.vault_root)
        digest = (self.vault_root / "DIGEST.md").read_text()
        self.assertIn("## Open loops (from LOOPS.md)", digest)
        self.assertIn("AM 207", digest)
        self.assertNotIn("stale task node", digest)
        self.assertNotIn("Open tasks", digest)

    def test_curated_files_survive_stale_generated_are_pruned(self):
        self._build_brain()
        curated = self.vault_root / "NOW.md"
        curated.write_text("my curated state\n")
        gdir = self.vault_root / "graph"
        gdir.mkdir()
        stale = gdir / "old-category.md"
        stale.write_text("left over from a renamed category\n")
        vault.render(self.conn, "Alvin", dest=self.vault_root)
        self.assertEqual(curated.read_text(), "my curated state\n")  # untouched
        self.assertFalse(stale.exists())                             # pruned

    def test_render_is_deterministic(self):
        self._build_brain()
        vault.render(self.conn, "Alvin", dest=self.vault_root)
        first = (self.vault_root / "graph" / "education.md").read_text()
        vault.render(self.conn, "Alvin", dest=self.vault_root)
        self.assertEqual(first, (self.vault_root / "graph" / "education.md").read_text())


class AutoRenderTests(VaultTestCase):
    def test_respects_vault_auto_off(self):
        config.load = lambda: {"vault_auto": False, "vault_dir": str(self.vault_root)}
        self._build_brain()
        vault.auto_render(self.conn, "Alvin")
        self.assertEqual(list(self.vault_root.iterdir()), [])

    def test_renders_into_configured_dir(self):
        config.load = lambda: {"vault_dir": str(self.vault_root)}
        self._build_brain()
        vault.auto_render(self.conn, "Alvin")
        self.assertTrue((self.vault_root / "DIGEST.md").exists())

    def test_swallows_render_failures(self):
        config.load = lambda: {"vault_dir": str(self.vault_root)}
        orig = vault.render
        vault.render = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            vault.auto_render(self.conn, "Alvin")  # must not raise
        finally:
            vault.render = orig


if __name__ == "__main__":
    unittest.main()
