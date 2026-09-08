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


if __name__ == "__main__":
    unittest.main()
