"""Tests for DECISIONS.md (brain/decisions.py) — the append-only decision ledger."""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.decisions as decisions

D1 = date(2026, 8, 31)
D2 = date(2026, 9, 1)


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def add(self, title="Fall core", what="AC 215 + CS 2881R + STAT 211", why="fits", when=D1, **kw):
        return decisions.append(self.root, title, what, why, when=when, commit=False, **kw)

    def test_append_creates_file_and_sequential_ids(self):
        a = self.add()
        b = self.add(title="Fourth seat", what="9.522", why="best fit", when=D2,
                     rejected="6.7960 (clash)", revisit="Protopapas allows late arrival", source="Crimson Track")
        self.assertEqual((a.id, b.id), ("D-001", "D-002"))
        text = decisions.path(self.root).read_text()
        self.assertTrue(text.startswith("# DECISIONS"))
        self.assertIn("## D-002 · 2026-09-01 · Fourth seat", text)
        self.assertIn("- **Revisit if:** Protopapas allows late arrival", text)
        ds, errs = decisions.load(self.root)
        self.assertEqual(errs, [])
        self.assertEqual([d.id for d in ds], ["D-001", "D-002"])
        self.assertEqual(ds[1].rejected, "6.7960 (clash)")
        self.assertEqual(ds[0].rejected, "—")

    def test_append_only_extends_existing_text(self):
        self.add()
        before = decisions.path(self.root).read_text()
        self.add(title="Second", when=D2)
        after = decisions.path(self.root).read_text()
        self.assertTrue(after.startswith(before))

    def test_validation(self):
        with self.assertRaises(decisions.DecisionError):
            self.add(title="")
        with self.assertRaises(decisions.DecisionError):
            self.add(what="two\nlines")

    def test_lint_catches_gaps_and_order(self):
        self.add()
        self.add(title="B", when=D2)
        self.assertEqual(decisions.lint(self.root), [])
        p = decisions.path(self.root)
        p.write_text(p.read_text().replace("D-002", "D-004"))
        self.assertTrue(any("sequential" in e for e in decisions.lint(self.root)))
        p.write_text(p.read_text().replace("D-004", "D-002").replace("2026-09-01", "2026-08-01"))
        self.assertTrue(any("dated before" in e for e in decisions.lint(self.root)))

    def test_lint_catches_missing_fields_and_refuses_append(self):
        self.add()
        p = decisions.path(self.root)
        p.write_text(p.read_text().replace("- **Why:** fits\n", ""))
        self.assertTrue(any("missing Why" in e for e in decisions.lint(self.root)))
        with self.assertRaises(decisions.DecisionError):
            self.add(title="blocked")

    def test_recent(self):
        self.add()
        self.add(title="Newer", when=D2)
        self.assertEqual(decisions.recent(self.root, date(2026, 9, 5), days=4), ["D-002 (Sep 01): Newer"])

    def test_missing_file_lint(self):
        self.assertEqual(decisions.lint(self.root), ["DECISIONS.md missing"])


class PreCommitHookTests(unittest.TestCase):
    """The git hook is the second enforcement layer: it must reject removals but allow appends."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.git = ["git", "-C", str(self.root)]
        try:
            subprocess.run(self.git + ["init", "-q"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("git not available")
        subprocess.run(self.git + ["config", "user.email", "t@t"], check=True)
        subprocess.run(self.git + ["config", "user.name", "t"], check=True)
        subprocess.run(self.git + ["config", "commit.gpgsign", "false"], check=True)
        self.assertTrue(decisions.install_pre_commit(self.root))
        hook = self.root / ".git" / "hooks" / "pre-commit"
        self.assertTrue(hook.stat().st_mode & stat.S_IXUSR)

    def commit(self, msg):
        subprocess.run(self.git + ["add", "-A"], check=True, capture_output=True)
        return subprocess.run(self.git + ["commit", "-q", "-m", msg], capture_output=True, text=True)

    def test_append_allowed_edit_rejected(self):
        decisions.append(self.root, "A", "a", "why", when=D1, commit=False)
        self.assertEqual(self.commit("first").returncode, 0)
        decisions.append(self.root, "B", "b", "why", when=D2, commit=False)
        self.assertEqual(self.commit("append").returncode, 0, "appending must pass the hook")
        p = decisions.path(self.root)
        p.write_text(p.read_text().replace("- **Why:** why\n", "- **Why:** changed\n", 1))
        r = self.commit("edit")
        self.assertNotEqual(r.returncode, 0, "editing an entry must be rejected")
        self.assertIn("append-only", r.stderr)

    def test_install_is_idempotent(self):
        self.assertTrue(decisions.install_pre_commit(self.root))


if __name__ == "__main__":
    unittest.main()
