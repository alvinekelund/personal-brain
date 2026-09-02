"""Tests for `brain doctor` (brain/doctor.py) — the health check that makes a
broken brain announce itself. All paths are injected; nothing real is read."""
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.db as db
import brain.doctor as doctor
import brain.decisions as decisions
import brain.llm as llm
import brain.loops as loops

TODAY = date(2026, 9, 1)
NOW_MD = "# NOW\n## 🔥 Hot right now\n<!-- loops:start -->\nx\n<!-- loops:end -->\n"


def by_name(checks):
    return {c.name: c for c in checks}


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "vault"
        self.root.mkdir()
        (self.root / "NOW.md").write_text(NOW_MD)
        self.bin = self.tmp / "venv" / "bin" / "brain"
        self.bin.parent.mkdir(parents=True)
        self.bin.write_text("#!/bin/sh\n")
        self.bin.chmod(0o755)
        self.db = self.tmp / "brain.db"
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = str(self.db)
        conn = db.connect()
        conn.execute("INSERT INTO ingestion_log (id, raw_text, source, ingested_at, nodes_added, edges_added) "
                     "VALUES ('i1', 'x', 's', ?, '[]', '[]')", (time.time() - 3600,))
        conn.commit()
        conn.close()
        self._orig_have_key = llm.have_key
        llm.have_key = lambda: True
        self.settings = self.tmp / "settings.json"
        self.settings.write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
            {"type": "command", "command": f"{self.bin} today"}]}]}}))
        self.claude_json = self.tmp / "claude.json"
        self.claude_json.write_text(json.dumps({"mcpServers": {"brain": {"command": str(self.bin), "args": ["mcp"]}}}))
        self.tasks = self.tmp / "scheduled-tasks"
        (self.tasks / "nightly").mkdir(parents=True)
        (self.tasks / "nightly" / "SKILL.md").write_text(f"run {self.bin} add \"fact\"\n")

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        llm.have_key = self._orig_have_key

    def run_doctor(self, **kw):
        import urllib.error
        def reachable():
            raise urllib.error.HTTPError("https://x/", 404, "nf", {}, None)
        args = dict(root=self.root, today=TODAY, db_path=self.db, expected_bin=self.bin,
                    settings=self.settings, claude_json=self.claude_json, tasks_dir=self.tasks,
                    api_probe=reachable)
        args.update(kw)
        return doctor.run(**args)

    def test_healthy_setup(self):
        import brain.now as now
        (self.root / "IDENTITY.md").write_text("**Alvin**\n")
        loops.add(self.root, "A", "2026-09-09", "alvin", "jobs", "n", today=TODAY, commit=False)
        decisions.append(self.root, "T", "d", "w", when=TODAY, commit=False)
        now.write(self.root)                      # NOW.md becomes generated → the now.md check applies
        checks = by_name(self.run_doctor())
        for name in ("binary", "graph", "gemini-key", "gemini-api", "vault-activity", "now.md", "loops", "decisions", "hooks", "mcp", "scheduled-tasks"):
            self.assertEqual(checks[name].status, "ok", f"{name}: {checks[name].detail}")
        self.assertEqual(checks["vault-git"].status, "warn")   # not a git repo — a warning, not a failure
        self.assertEqual(doctor.worst(list(checks.values())), "warn")
        self.assertTrue(doctor.brief(list(checks.values())).startswith("brain ⚠"))

    def test_missing_binary_is_loud_everywhere(self):
        self.bin.unlink()
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["binary"].status, "fail")
        self.assertEqual(checks["hooks"].status, "fail")
        self.assertEqual(checks["mcp"].status, "fail")
        self.assertEqual(checks["scheduled-tasks"].status, "fail")
        line = doctor.brief(list(checks.values()))
        self.assertTrue(line.startswith("✗ BRAIN BROKEN"))
        self.assertIn("reinstall", line)
        self.assertEqual(doctor.worst(list(checks.values())), "fail")

    def test_stale_graph_and_now_warn(self):
        old = time.time() - 80 * 3600
        conn = sqlite3.connect(str(self.db))
        conn.execute("UPDATE ingestion_log SET ingested_at = ?", (old,))
        conn.commit()
        conn.close()
        os.utime(self.root / "NOW.md", (old, old))
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["graph"].status, "warn")
        self.assertIn("80h", checks["graph"].detail)
        self.assertEqual(checks["vault-activity"].status, "warn")
        self.assertEqual(checks["now.md"].status, "warn")     # legacy hand-written NOW.md in this fixture

    def test_missing_key_and_ledgers_warn(self):
        llm.have_key = lambda: False
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["gemini-key"].status, "warn")
        self.assertEqual(checks["loops"].status, "warn")
        self.assertEqual(checks["decisions"].status, "warn")

    def test_lint_errors_fail(self):
        loops.add(self.root, "A", "2026-09-09", "alvin", "jobs", "n", today=TODAY, commit=False)
        (self.root / "NOW.md").write_text(NOW_MD)   # hand-edit the rendered block
        self.assertEqual(by_name(self.run_doctor())["loops"].status, "fail")

    def test_missing_db(self):
        self.assertEqual(by_name(self.run_doctor(db_path=self.tmp / "nope.db"))["graph"].status, "fail")

    def test_report_format(self):
        text = doctor.report(self.run_doctor())
        self.assertIn("✓ binary", text)
        self.assertRegex(text, r"[✓⚠✗] graph")

    def test_tilde_paths_are_expanded_not_flagged(self):
        home_rel = "~/" + str(self.bin.relative_to(Path.home())) if str(self.bin).startswith(str(Path.home())) else str(self.bin)
        (self.tasks / "nightly" / "SKILL.md").write_text(f"run {home_rel} add\n")
        self.assertEqual(by_name(self.run_doctor())["scheduled-tasks"].status, "ok")
        self.assertEqual(doctor._paths_in("reinstall: ~/x/brain and /y/python3 and /z/other"),
                         [str(Path("~/x/brain").expanduser()), "/y/python3"])

    def test_api_probe_tls_failure_is_a_failure_with_fix(self):
        import ssl
        def bad_tls():
            raise ssl.SSLCertVerificationError("certificate verify failed: unable to get local issuer certificate")
        c = doctor.check_api(bad_tls)
        self.assertEqual(c.status, "fail")
        self.assertIn("pip install certifi", c.detail)
        def offline():
            raise OSError("Network is unreachable")
        self.assertEqual(doctor.check_api(offline).status, "warn")
        self.assertEqual(by_name(self.run_doctor())["gemini-api"].status, "ok")

    def test_unregistered_mcp_is_a_warning(self):
        self.claude_json.write_text(json.dumps({"mcpServers": {}}))
        self.assertEqual(by_name(self.run_doctor())["mcp"].status, "warn")


if __name__ == "__main__":
    unittest.main()
