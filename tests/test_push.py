"""The phone channel: `brain push` sends an iMessage to Alvin's own number via
Messages.app (osascript). Nothing here talks to Messages: the runner is faked."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.config as config
import brain.push as push


class FakeRun:
    def __init__(self, rc=0, out="sent", err=""):
        self.rc, self.out, self.err, self.calls = rc, out, err, []

    def __call__(self, args, **kw):
        self.calls.append(args)

        class R:
            returncode, stdout, stderr = self.rc, self.out, self.err
        return R()


class PushTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = config.CONFIG_PATH
        config.CONFIG_PATH = self.tmp / "config.json"
        config.save({"vault_dir": str(self.tmp / "vault")})
        self.log = self.tmp / "push.log"

    def tearDown(self):
        config.CONFIG_PATH = self._orig

    def test_send_needs_a_recipient_and_text(self):
        self.assertIn("brain push --set-to", push.send("hi", runner=FakeRun(), log_path=self.log))
        push.set_handle("+13392221646")
        self.assertEqual(push.send("   ", runner=FakeRun(), log_path=self.log), "nothing to send")
        with self.assertRaises(ValueError):
            push.set_handle("  ")

    def test_send_passes_handle_and_text_as_arguments_and_logs(self):
        push.set_handle("+13392221646")
        run = FakeRun()
        self.assertIsNone(push.send("Seat lock in 3d\n enroll 9.522", runner=run, log_path=self.log))
        args = run.calls[0]
        self.assertEqual(args[0], "osascript")
        self.assertEqual(args[-2:], ["+13392221646", "Seat lock in 3d enroll 9.522"])   # argv, never interpolated into the script
        self.assertIn("service type = iMessage", " ".join(args))
        self.assertIn("sent → +13392221646: Seat lock in 3d enroll 9.522", self.log.read_text())
        self.assertIsNone(push.send("x", to="alvin@example.com", runner=run, log_path=self.log))
        self.assertEqual(run.calls[1][-2], "alvin@example.com")

    def test_failure_is_returned_and_logged(self):
        push.set_handle("+13392221646")
        err = push.send("hi", runner=FakeRun(rc=1, out="", err="Messages got an error: not signed in"), log_path=self.log)
        self.assertIn("not signed in", err)
        self.assertIn("FAILED (Messages got an error", self.log.read_text())
        run = FakeRun()
        push.send("y" * 2000, runner=run, log_path=self.log)
        self.assertLessEqual(len(run.calls[0][-1]), push.MAX_CHARS)

    def test_identical_note_within_six_hours_is_not_sent_twice(self):
        """Oct 12 2026: one "run stopped" alert reached the phone twice, 66 minutes apart."""
        push.set_handle("+13392221646")
        run = FakeRun()
        self.assertIsNone(push.send("X4 run stopped (line 17)", runner=run, log_path=self.log))
        self.assertIsNone(push.send("X4 run  stopped (line 17)\n", runner=run, log_path=self.log))   # same note
        self.assertEqual(len(run.calls), 1)
        self.assertIn("SKIPPED (same note sent", self.log.read_text())
        self.assertIsNotNone(push.recent_duplicate("X4 run stopped (line 17)", log_path=self.log))
        push.send("X4 run stopped (line 17)", to="alvin@example.com", runner=run, log_path=self.log)   # other recipient
        push.send("X4 run stopped (line 17)", runner=run, log_path=self.log, dedupe_hours=0)          # forced
        self.assertEqual(len(run.calls), 3)
        later = __import__("time").time() + 7 * 3600
        self.assertIsNone(push.recent_duplicate("X4 run stopped (line 17)", log_path=self.log, now=later))

    def test_email_copy_goes_through_mail_app_with_argv(self):
        run = FakeRun()
        self.assertIn("--set-email", push.send_email("hi", runner=run, log_path=self.log))
        push.set_email("alvine@mail.instinct.com", sender="ALVIN.EKELUND@gmail.com")
        self.assertIsNone(push.send_email("Seat lock in 3d. Enroll 9.522.", runner=run, log_path=self.log))
        args = run.calls[0]
        self.assertEqual(args[0], "osascript")
        self.assertIn('tell application "Mail"', " ".join(args))
        self.assertEqual(args[-4:], ["alvine@mail.instinct.com", "brain: Seat lock in 3d. Enroll 9.522.",
                                     "Seat lock in 3d. Enroll 9.522.", "ALVIN.EKELUND@gmail.com"])
        self.assertIn("sent → email alvine@mail.instinct.com: Seat lock in 3d", self.log.read_text())
        err = push.send_email("x", runner=FakeRun(rc=1, out="", err="Mail got an error: no account"), log_path=self.log)
        self.assertIn("no account", err)
        self.assertIn("FAILED (Mail got an error", self.log.read_text())
        push.clear_email()
        self.assertEqual(push.email_handle(), "")
        with self.assertRaises(ValueError):
            push.set_email("not-an-address")


    def test_quiet_hours_are_23_to_07(self):
        from datetime import datetime
        q = lambda h, m=0: push.quiet_now(datetime(2026, 10, 19, h, m))
        self.assertTrue(q(23) and q(0) and q(3) and q(6, 59))
        self.assertFalse(q(7) or q(12) or q(22, 59))

    def test_held_pushes_go_out_as_one_message_and_stay_when_the_send_fails(self):
        push.set_handle("+13392221646")
        held = self.tmp / "held.jsonl"
        self.assertIsNone(push.hold("Event added: Kempner talk", held_path=held, log_path=self.log))
        self.assertIsNone(push.hold("Event added:  Kempner talk", held_path=held, log_path=self.log))   # same note
        self.assertIsNone(push.hold("Helm applied to Figma", email=False, held_path=held, log_path=self.log))
        self.assertEqual(len(push.held(held)), 3)
        self.assertIn("HELD (quiet 23-07) → +13392221646: Helm applied to Figma", self.log.read_text())
        self.assertEqual(push.flush_held(runner=FakeRun(rc=1, out="", err="not signed in"), held_path=held,
                                         log_path=self.log), ["not signed in"])
        self.assertEqual(len(push.held(held)), 3)                     # kept for the next try
        run = FakeRun()
        self.assertEqual(push.flush_held(runner=run, held_path=held, log_path=self.log), [])
        self.assertEqual(len(run.calls), 1)                           # no email copy configured
        self.assertEqual(run.calls[0][-1], "Overnight: Event added: Kempner talk | Helm applied to Figma")
        self.assertFalse(held.exists())
        self.assertEqual(push.flush_held(runner=run, held_path=held, log_path=self.log), [])
        push.hold("Only one", held_path=held, log_path=self.log)
        push.set_email("alvine@mail.instinct.com")
        push.flush_held(runner=run, held_path=held, log_path=self.log)
        self.assertEqual(run.calls[1][-1], "Only one")
        self.assertIn('tell application "Mail"', " ".join(run.calls[2]))


if __name__ == "__main__":
    unittest.main()
