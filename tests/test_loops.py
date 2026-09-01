"""Tests for LOOPS.md (brain/loops.py) — the open-loop ledger.

Everything runs against a temp vault dir with a fixed `today`, so results are
deterministic and no real vault, git repo, or clock is touched.
"""
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.loops as loops

TODAY = date(2026, 9, 1)
NOW_TEMPLATE = """# NOW
intro line

## 🔥 Hot right now
<!-- loops:start -->
(stale)
<!-- loops:end -->

## Other
keep me
"""


class LoopsTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "NOW.md").write_text(NOW_TEMPLATE, encoding="utf-8")

    def add(self, title="Lock fourth seat", due="2026-09-09", owner="alvin", area="harvard",
            next_="click Enroll Selected", **kw):
        return loops.add(self.root, title, due, owner, area, next_, today=TODAY, commit=False, **kw)


class GrammarTests(unittest.TestCase):
    def test_roundtrip(self):
        line = ('- [ ] L-003 · Lock fourth seat · due 2026-09-09 · owner alvin · area harvard · prio 1'
                ' · since 2026-08-31 · touched 2026-09-01 · next: click "Enroll Selected" in the cart')
        l = loops.parse_line(line)
        self.assertEqual((l.id, l.title, l.due, l.owner, l.area, l.prio),
                         ("L-003", "Lock fourth seat", date(2026, 9, 9), "alvin", "harvard", 1))
        self.assertEqual(l.next, 'click "Enroll Selected" in the cart')
        self.assertEqual(l.to_line(), line)

    def test_next_is_last_and_may_contain_anything_but_separator(self):
        l = loops.parse_line("- [ ] L-001 · T · due 2026-09-09 · owner alvin · area life · next: a: b, c; d — e (f)")
        self.assertEqual(l.next, "a: b, c; d — e (f)")

    def test_closed_line(self):
        l = loops.parse_line("- [x] L-002 · T · due 2026-09-01 · owner alvin · area jobs · done 2026-09-01 · next: n")
        self.assertTrue(l.closed)
        self.assertEqual(l.done, date(2026, 9, 1))

    def test_checkbox_done_mismatch(self):
        with self.assertRaises(loops.LoopError):
            loops.parse_line("- [x] L-002 · T · due 2026-09-01 · owner alvin · area jobs · next: n")

    def test_missing_fields(self):
        for bad in ("- [ ] L-001 · T · owner alvin · area jobs · next: n",          # no due
                    "- [ ] L-001 · T · due 2026-09-01 · area jobs · next: n",       # no owner
                    "- [ ] L-001 · T · due 2026-09-01 · owner alvin · area jobs"):  # no next
            with self.assertRaises(loops.LoopError, msg=bad):
                loops.parse_line(bad)

    def test_invalid_values(self):
        base = "- [ ] L-001 · T · due {due} · owner {owner} · area {area} · prio {prio} · next: n"
        ok = dict(due="2026-09-01", owner="alvin", area="jobs", prio="2")
        loops.parse_line(base.format(**ok))
        for k, v in (("due", "Sep 9"), ("owner", "bob"), ("owner", "waiting:"), ("area", "school"), ("prio", "5")):
            with self.assertRaises(loops.LoopError, msg=f"{k}={v}"):
                loops.parse_line(base.format(**{**ok, k: v}))

    def test_waiting_owner(self):
        l = loops.parse_line("- [ ] L-001 · T · due 2026-09-07 · owner waiting:boaz-barak · area harvard · next: nudge")
        self.assertEqual(l.waiting_on, "boaz-barak")

    def test_separator_forbidden_in_text(self):
        with self.assertRaises(loops.LoopError):
            loops.validate(loops.Loop("L-001", "a · b", date(2026, 9, 1), "alvin", "jobs", "n"))

    def test_parse_file_sections_and_errors(self):
        text = ("# LOOPS\n## Open\n- [ ] L-001 · A · due 2026-09-02 · owner alvin · area jobs · next: n\n"
                "- [ ] L-002 · broken\n## Closed\n"
                "- [x] L-003 · B · due 2026-08-30 · owner alvin · area jobs · done 2026-08-31 · next: n\n")
        ledger = loops.parse(text)
        self.assertEqual([l.id for l in ledger.open], ["L-001"])
        self.assertEqual([l.id for l in ledger.closed], ["L-003"])
        self.assertEqual(len(ledger.errors), 1)
        self.assertIn("line 4", ledger.errors[0])
        self.assertEqual(ledger.next_id(), "L-004")

    def test_serialize_is_sorted_and_deterministic(self):
        ledger = loops.Ledger(open=[
            loops.Loop("L-002", "later", date(2026, 9, 9), "alvin", "jobs", "n", prio=2),
            loops.Loop("L-001", "urgent", date(2026, 9, 3), "alvin", "jobs", "n", prio=1),
            loops.Loop("L-003", "soon", date(2026, 9, 2), "alvin", "jobs", "n", prio=2),
        ])
        body = loops.serialize(ledger)
        ids = [l.split()[3] for l in body.splitlines() if l.startswith("- [")]
        self.assertEqual(ids, ["L-001", "L-003", "L-002"])
        self.assertEqual(body, loops.serialize(loops.parse(body)))


class OperationTests(LoopsTestCase):
    def test_add_assigns_ids_and_writes_file(self):
        a = self.add()
        b = self.add(title="Second", area="jobs")
        self.assertEqual((a.id, b.id), ("L-001", "L-002"))
        self.assertEqual((a.since, a.touched), (TODAY, TODAY))
        ledger = loops.load(self.root)
        self.assertEqual([l.id for l in ledger.open], ["L-001", "L-002"])
        self.assertEqual(ledger.errors, [])

    def test_add_validates(self):
        with self.assertRaises(loops.LoopError):
            self.add(due="next week")
        with self.assertRaises(loops.LoopError):
            self.add(owner="someone")
        self.assertEqual(loops.load(self.root).open, [])

    def test_add_with_backdated_since(self):
        l = self.add(since="2026-08-31")
        self.assertEqual(l.since, date(2026, 8, 31))
        self.assertEqual(l.touched, TODAY)

    def test_done_moves_to_closed(self):
        self.add()
        closed = loops.done(self.root, "L-001", note="enrolled 9.522", today=TODAY, commit=False)
        self.assertEqual(closed.done, TODAY)
        ledger = loops.load(self.root)
        self.assertEqual(ledger.open, [])
        self.assertEqual(ledger.closed[0].note, "enrolled 9.522")
        with self.assertRaises(loops.LoopError):
            loops.done(self.root, "L-001", today=TODAY, commit=False)
        with self.assertRaises(loops.LoopError):
            loops.done(self.root, "L-999", today=TODAY, commit=False)

    def test_edit_bumps_touched_and_validates(self):
        self.add()
        later = date(2026, 9, 3)
        l = loops.edit(self.root, "L-001", today=later, commit=False, due="2026-09-10", next_="enroll")
        self.assertEqual((l.due, l.next, l.touched), (date(2026, 9, 10), "enroll", later))
        with self.assertRaises(loops.LoopError):
            loops.edit(self.root, "L-001", today=later, commit=False, area="nope")
        with self.assertRaises(loops.LoopError):
            loops.edit(self.root, "L-001", today=later, commit=False)

    def test_ids_never_reused_after_close(self):
        self.add()
        loops.done(self.root, "L-001", today=TODAY, commit=False)
        self.assertEqual(self.add(title="New").id, "L-002")


class RenderTests(LoopsTestCase):
    def test_now_block_rendered_and_rest_untouched(self):
        self.add()
        self.add(title="Ask Boaz", due="2026-09-07", owner="waiting:boaz-barak", next_="nudge")
        text = (self.root / "NOW.md").read_text()
        self.assertIn("intro line", text)
        self.assertIn("keep me", text)
        self.assertNotIn("(stale)", text)
        self.assertIn("**Lock fourth seat** — due Sep 9", text)
        self.assertIn("⏳ **Waiting on:** boaz-barak", text)
        self.assertIn("`L-001`", text)

    def test_render_is_idempotent_and_date_free(self):
        self.add()
        before = (self.root / "NOW.md").read_text()
        self.assertFalse(loops.render_now(self.root))
        self.assertEqual(before, (self.root / "NOW.md").read_text())
        self.assertNotIn(TODAY.isoformat(), loops.render_block(loops.load(self.root)))

    def test_no_markers_leaves_now_alone(self):
        (self.root / "NOW.md").write_text("# NOW\nno markers\n")
        self.add()
        self.assertEqual((self.root / "NOW.md").read_text(), "# NOW\nno markers\n")
        errors, _ = loops.lint(self.root, TODAY)
        self.assertTrue(any("markers" in e for e in errors))


class LintTests(LoopsTestCase):
    def test_clean(self):
        self.add()
        self.assertEqual(loops.lint(self.root, TODAY), ([], []))

    def test_detects_hand_edited_now(self):
        self.add()
        p = self.root / "NOW.md"
        p.write_text(p.read_text().replace("Lock fourth seat", "Lock 4th seat"))
        errors, _ = loops.lint(self.root, TODAY)
        self.assertTrue(any("out of date" in e for e in errors))

    def test_detects_hand_edited_loops(self):
        self.add()
        p = self.root / "LOOPS.md"
        p.write_text(p.read_text() + "- [ ] L-001 · dup · due 2026-09-09 · owner alvin · area jobs · next: n\n")
        errors, _ = loops.lint(self.root, TODAY)
        self.assertTrue(any("duplicate id L-001" in e for e in errors))
        with self.assertRaises(loops.LoopError):
            self.add(title="blocked while broken")

    def test_warnings(self):
        self.add(title="Late", due="2026-08-30")
        self.add(title="Old", due="2026-12-01", since="2026-08-01")
        self.add(title="Wait", due="2026-09-20", owner="waiting:someone")
        _, warnings = loops.lint(self.root, date(2026, 9, 10))
        joined = "\n".join(warnings)
        self.assertIn("L-001 overdue by 11d", joined)
        self.assertIn("untouched for 9d", joined)
        self.assertIn("waiting on someone for 9d", joined)


class TodayTests(LoopsTestCase):
    def test_report_sections_and_countdowns(self):
        self.add(title="Seat", due="2026-09-09", prio=1, next_="enroll")
        self.add(title="Late", due="2026-08-31", next_="fix")
        self.add(title="Far", due="2026-10-15", next_="later")
        self.add(title="Boaz", due="2026-09-07", owner="waiting:boaz-barak", next_="nudge", since="2026-08-25")
        self.add(title="Draft", due="2026-09-04", owner="claude", next_="draft the email")
        kw = dict(horizon=10, doctor_line="brain ✓ ok", decisions=["D-001 (Sep 01): x"])
        r = loops.today_report(self.root, TODAY, **kw)
        self.assertIn("Tue 2026-09-01", r)
        self.assertIn("brain ✓ ok", r)
        self.assertIn("OVERDUE 1d", r)
        self.assertIn("8d  L-001 P1 Seat → enroll", r)
        self.assertNotIn("Far", r.split("TOP 3")[0])
        self.assertIn("boaz-barak 0d  L-004 Boaz (by Sep 07)", r)   # age counts from last touch
        self.assertNotIn("← nudge", r)
        self.assertIn("CLAUDE CAN DO NOW:\n  L-005 Draft → draft the email", r)
        self.assertIn("1. enroll", r)
        self.assertIn("D-001 (Sep 01): x", r)
        self.assertIn("5 open loop(s), 1 beyond the horizon", r)
        self.assertEqual(r, loops.today_report(self.root, TODAY, **kw))
        nag = loops.today_report(self.root, date(2026, 9, 8))
        self.assertIn("boaz-barak 7d", nag)
        self.assertIn("← nudge", nag)

    def test_brief_fits_a_push(self):
        for i in range(6):
            self.add(title=f"Loop number {i} with a long title", due="2026-09-03",
                     next_="do the thing that needs doing")
        b = loops.brief(self.root, TODAY)
        self.assertLessEqual(len(b), 200)
        self.assertIn("(2d)", b)
        self.assertEqual(loops.brief(Path(tempfile.mkdtemp()), TODAY), "No open loops.")


class InboxTests(LoopsTestCase):
    def test_add_list_dedup(self):
        n = loops.inbox_add(self.root, ["email Heli", " email  Heli ", "book SSN · slot"], source="brain add · x", today=TODAY)
        self.assertEqual(n, 2)
        items = loops.inbox_list(self.root)
        self.assertEqual([i["text"] for i in items], ["email Heli", "book SSN - slot"])   # separator sanitised
        self.assertEqual(items[0], {"date": "2026-09-01", "text": "email Heli", "source": "brain add - x"})
        self.assertEqual(loops.inbox_add(self.root, ["EMAIL HELI"], today=TODAY), 0)        # case-insensitive dedup
        self.assertTrue((self.root / "LOOPS-INBOX.md").read_text().startswith("# LOOPS-INBOX"))

    def test_drop_and_clear(self):
        loops.inbox_add(self.root, ["a", "b", "c"], today=TODAY)
        self.assertEqual(loops.inbox_drop(self.root, 2)["text"], "b")
        self.assertEqual([i["text"] for i in loops.inbox_list(self.root)], ["a", "c"])
        with self.assertRaises(loops.LoopError):
            loops.inbox_drop(self.root, 5)
        self.assertEqual(loops.inbox_clear(self.root), 2)
        self.assertEqual(loops.inbox_list(self.root), [])
        self.assertEqual(loops.inbox_clear(self.root), 0)

    def test_missing_vault_dir_is_a_noop(self):
        self.assertEqual(loops.inbox_add(Path(tempfile.mkdtemp()) / "nope", ["x"], today=TODAY), 0)

    def test_today_surfaces_inbox(self):
        self.add()
        self.assertNotIn("INBOX", loops.today_report(self.root, TODAY))
        loops.inbox_add(self.root, ["email Heli"], today=TODAY)
        self.assertIn("INBOX: 1 untriaged action item(s)", loops.today_report(self.root, TODAY))


if __name__ == "__main__":
    unittest.main()
