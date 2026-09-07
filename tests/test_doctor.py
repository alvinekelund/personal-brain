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
        import brain.config as config                          # the doctor reads the owner from config:
        self._orig_config_path = config.CONFIG_PATH            # CI has no ~/.personal-brain/config.json, so
        config.CONFIG_PATH = self.tmp / "config.json"          # without this the tree checks see no root
        config.save({"user": "Alvin"})
        self.settings = self.tmp / "settings.json"
        self.settings.write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [
            {"type": "command", "command": f"{self.bin} today"}]}]}}))
        self.claude_json = self.tmp / "claude.json"
        self.claude_json.write_text(json.dumps({"mcpServers": {"brain": {"command": str(self.bin), "args": ["mcp"]}}}))
        self.capture_log = self.tmp / "capture.log"
        self.capture_log.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " session abc: ingested 2 node(s)\n")
        self.backups = self.tmp / "backups"
        self.backups.mkdir()
        (self.backups / "brain-2026-09-01-080000.db").write_bytes(b"x")
        self.brief_log = self.tmp / "brief.log"
        self.brief_log.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " Seat lock in 3d: enroll 9.522\n")
        self.tasks = self.tmp / "scheduled-tasks"
        (self.tasks / "nightly").mkdir(parents=True)
        (self.tasks / "nightly" / "SKILL.md").write_text(f"run {self.bin} add \"fact\"\n")

    def tearDown(self):
        import brain.config as config
        config.CONFIG_PATH = self._orig_config_path
        db.DB_PATH = self._orig_db_path
        llm.have_key = self._orig_have_key

    def run_doctor(self, **kw):
        import urllib.error
        def reachable():
            raise urllib.error.HTTPError("https://x/", 404, "nf", {}, None)
        args = dict(root=self.root, today=TODAY, db_path=self.db, expected_bin=self.bin,
                    settings=self.settings, claude_json=self.claude_json, tasks_dir=self.tasks,
                    api_probe=reachable, capture_log=self.capture_log, brief_log=self.brief_log,
                    backups_dir=self.backups)
        args.update(kw)
        return doctor.run(**args)

    def test_healthy_setup(self):
        import brain.now as now
        (self.root / "IDENTITY.md").write_text("**Alvin**\n")
        loops.add(self.root, "A", "2026-09-09", "alvin", "jobs", "n", today=TODAY, commit=False)
        decisions.append(self.root, "T", "d", "w", when=TODAY, commit=False)
        now.write(self.root)                      # NOW.md becomes generated → the now.md check applies
        checks = by_name(self.run_doctor())
        for name in ("binary", "graph", "graph-tree", "claims", "backups", "gemini-key", "gemini-api", "capture", "brief", "vault-activity", "now.md", "loops", "decisions", "hooks", "mcp", "scheduled-tasks"):
            self.assertEqual(checks[name].status, "ok", f"{name}: {checks[name].detail}")
        self.assertEqual(checks["vault-git"].status, "warn")   # not a git repo — a warning, not a failure
        self.assertEqual(doctor.worst(list(checks.values())), "warn")
        self.assertTrue(doctor.brief(list(checks.values())).startswith("brain ⚠"))

    def test_vault_index_check(self):
        import brain.index as index
        (self.root / "profile").mkdir()
        (self.root / "profile" / "background.md").write_text("---\ntype: profile\nname: Background\n---\n# Background\n")
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["vault-index"].status, "warn")
        self.assertIn("none indexed", checks["vault-index"].detail)
        conn = db.connect()
        index.build(conn, self.root, embed=False)
        conn.close()
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["vault-index"].status, "ok", checks["vault-index"].detail)
        self.assertIn("ledger lines", checks["vault-index"].detail)
        # a loop written after the last index is matched by nothing until `brain index` runs
        loops.add(self.root, "Collect the repayments", "2026-09-21", "alvin", "life", "chase", today=TODAY, commit=False)
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["vault-index"].status, "warn")
        self.assertIn("ledger line(s) changed since the last index", checks["vault-index"].detail)
        self.assertIn("L-001", checks["vault-index"].detail)
        conn = db.connect()
        index.build(conn, self.root, embed=False)
        conn.close()
        self.assertEqual(by_name(self.run_doctor())["vault-index"].status, "ok")
        (self.root / "profile" / "background.md").write_text("---\ntype: profile\nname: Background\n---\n# Background\n- moved to Boston\n")
        os.utime(self.root / "profile" / "background.md", (time.time() + 5, time.time() + 5))
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["vault-index"].status, "warn")
        self.assertIn("changed since the last index", checks["vault-index"].detail)

    def test_brief_check_reads_the_recorded_push(self):
        """Weekly review W36: the morning brief left no trace, so no review could
        verify a single push. `brain today --brief` records its line; the doctor
        shows the last one and warns when a day and a bit has passed without one."""
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["brief"].status, "ok", checks["brief"].detail)
        self.assertIn("Seat lock in 3d", checks["brief"].detail)
        old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 40 * 3600))
        self.brief_log.write_text(old + " Protopapas email due today\n")
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["brief"].status, "warn")
        self.assertIn("did not run or did not record", checks["brief"].detail)
        self.brief_log.unlink()
        checks = by_name(self.run_doctor())
        self.assertEqual(checks["brief"].status, "warn")
        self.assertIn("no brief.log yet", checks["brief"].detail)
        self.assertNotIn("brief", by_name(self.run_doctor(brief_log=None)))   # disabled by callers that want to

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

    def test_relative_vault_paths_are_not_read_as_absolute(self):
        (self.tasks / "nightly" / "SKILL.md").write_text(
            "edit areas/brain.md and docs/reviews/2026-W36.md, then run brain index\n")
        self.assertEqual(by_name(self.run_doctor())["scheduled-tasks"].status, "ok")
        self.assertEqual(doctor._paths_in("areas/brain.md · log/brain.md"), [])

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

    def test_graph_tree_and_capture_checks(self):
        conn = db.connect()
        db.add_node(conn, "Loose fact", type_="fact")          # orphan → structural failure
        conn.commit(); conn.close()
        self.assertEqual(by_name(self.run_doctor())["graph-tree"].status, "fail")
        self.assertIn("orphan", by_name(self.run_doctor())["graph-tree"].detail)
        self.capture_log.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " error: RuntimeError: boom\n")
        self.assertEqual(by_name(self.run_doctor())["capture"].status, "fail")
        self.capture_log.unlink()
        self.assertEqual(by_name(self.run_doctor())["capture"].status, "warn")

    def test_backups_check_wants_a_recent_snapshot(self):
        self.assertEqual(by_name(self.run_doctor())["backups"].status, "ok")      # fixture: a fresh file
        old = time.time() - 9 * 86400
        os.utime(self.backups / "brain-2026-09-01-080000.db", (old, old))
        c = by_name(self.run_doctor())["backups"]
        self.assertEqual(c.status, "warn")
        self.assertIn("9d old", c.detail)
        (self.backups / "brain-2026-09-01-080000.db").unlink()
        c = by_name(self.run_doctor())["backups"]
        self.assertEqual(c.status, "warn")
        self.assertIn("no backup yet", c.detail)
        self.assertNotIn("backups", by_name(self.run_doctor(backups_dir=None)))

    def test_stale_claims_warn_with_the_cure(self):
        conn = db.connect()
        nid = db.add_node(conn, "Move to Boston", type_="event", content="Alvin plans to relocate to Boston.")
        conn.execute("UPDATE nodes SET created_at = ? WHERE id = ?", (time.time() - 91 * 86400, nid))
        conn.commit(); conn.close()
        c = by_name(self.run_doctor())["claims"]
        self.assertEqual(c.status, "warn")
        self.assertIn("Move to Boston (91d, 'plans to')", c.detail)
        self.assertIn("brain stale", c.detail)

    def test_loops_warning_detail_is_ids_and_reasons_not_titles(self):
        """Three overdue titles made the session-start line 320 characters; the
        card lists the titles right below, so the health line keeps ids only."""
        for i in range(5):
            loops.add(self.root, f"A long loop title number {i} that says a lot", "2026-08-30", "alvin", "jobs", "n", today=TODAY, commit=False)   # touched today: overdue only
        c = by_name(self.run_doctor())["loops"]
        self.assertEqual(c.status, "warn")
        self.assertIn("5 open · 5 warning(s): L-001 overdue by 2d, L-002 overdue by 2d, L-003 overdue by 2d, +2 more", c.detail)
        self.assertNotIn("long loop title", c.detail)

    def test_capture_line_tallies_the_week(self):
        """The weekly review counted '14 sessions ingested, 25 skipped' by hand
        from capture.log; the doctor line now carries the 7-day tally."""
        stamp = lambda h: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - h * 3600))
        self.capture_log.write_text("\n".join([
            f"{stamp(300)} session old: ingested 1 node(s)",                       # outside the window
            f"{stamp(100)} session a: ingested 2 node(s), 3 edge(s)",
            f"{stamp(50)} session b: nothing durable",
            f"{stamp(40)} session c: skipped (automation session)",
            f"{stamp(30)} session d: error: RuntimeError: boom",
            f"{stamp(1)} session e: skipped (0 new chars of user text)",
        ]) + "\n")
        c = by_name(self.run_doctor())["capture"]
        self.assertEqual(c.status, "ok")
        self.assertIn("7d: 1 ingested, 1 nothing durable, 2 skipped, 1 failed", c.detail)

    def test_category_cross_links_warn_not_pass(self):
        """8 'relates_to <category>' edges showed under a green graph-tree line
        on Sep 6 2026: the summary named them, the status did not."""
        conn = db.connect()
        db.ensure_identity_anchor(conn, "Alvin")
        me = db.get_node_by_name(conn, "Alvin")["id"]
        edu = db.add_node(conn, "Education", type_="category"); db.add_edge(conn, edu, me, "part_of")
        h = db.add_node(conn, "Harvard", type_="organization"); db.add_edge(conn, h, edu, "part_of")
        db.add_edge(conn, h, edu, "used_in")                    # any cross-link to a category is noise
        for nid in (me, edu, h):
            db.set_embedding(conn, nid, [0.1, 0.2])
        conn.commit(); conn.close()
        c = by_name(self.run_doctor())["graph-tree"]
        self.assertEqual(c.status, "warn")
        self.assertIn("1 cross-link(s) to a category (brain repair)", c.detail)

    def test_unregistered_mcp_is_a_warning(self):
        self.claude_json.write_text(json.dumps({"mcpServers": {}}))
        self.assertEqual(by_name(self.run_doctor())["mcp"].status, "warn")


if __name__ == "__main__":
    unittest.main()
