"""The vault's post-commit hook re-indexes in the background after every commit (Oct 14 2026)."""
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from brain import index


class PostCommitHookTests(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.git = ["git", "-C", str(self.root)]
        try:
            subprocess.run(self.git + ["init", "-q"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("git not available")
        for k, v in (("user.email", "t@t"), ("user.name", "t"), ("commit.gpgsign", "false")):
            subprocess.run(self.git + ["config", k, v], check=True)
        self.mark = self.root / "calls.txt"
        self.brain = self.root / "fake-brain"
        self.brain.write_text(f'#!/bin/sh\necho "$@" >> "{self.mark}"\n')
        self.brain.chmod(0o755)
        (self.root / ".gitignore").write_text("calls.txt\nfake-brain\n")

    def commit(self, name):
        (self.root / name).write_text(name)
        subprocess.run(self.git + ["add", "-A"], check=True, capture_output=True)
        return subprocess.run(self.git + ["commit", "-q", "-m", name], capture_output=True, text=True, timeout=10)

    def calls(self, want, wait=5.0):
        end = time.time() + wait
        while time.time() < end:
            if self.mark.exists() and self.mark.read_text().count("index") >= want:
                break
            time.sleep(0.05)
        return self.mark.read_text().split() if self.mark.exists() else []

    def test_commit_runs_index_in_the_background(self):
        self.assertTrue(index.install_post_commit(self.root, brain=str(self.brain)))
        hook = self.root / ".git" / "hooks" / "post-commit"
        self.assertTrue(hook.stat().st_mode & stat.S_IXUSR)
        r = self.commit("a.md")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls(1), ["index"])
        lock = self.root / ".git" / "brain-index.lock"
        end = time.time() + 3
        while lock.exists() and time.time() < end:
            time.sleep(0.05)
        self.assertFalse(lock.exists())                      # the indexer cleans up after itself

    def test_a_commit_during_an_index_run_is_indexed_too(self):
        self.brain.write_text(f'#!/bin/sh\nsleep 1\necho "$@" >> "{self.mark}"\n')
        index.install_post_commit(self.root, brain=str(self.brain))
        self.commit("c.md")
        time.sleep(0.3)
        self.commit("d.md")                                  # lands while the first index runs: its hook skips
        self.assertEqual(self.calls(2, wait=6), ["index", "index"])   # the running indexer goes again

    def test_a_captured_commit_does_not_wait_for_the_index(self):
        # the brain CLI commits with capture_output: the hook must not hold its pipes open
        self.brain.write_text(f'#!/bin/sh\nsleep 3\necho "$@" >> "{self.mark}"\n')
        index.install_post_commit(self.root, brain=str(self.brain))
        t0 = time.time()
        self.commit("b.md")
        self.assertLess(time.time() - t0, 2.0)

    def test_a_foreign_hook_is_left_alone_and_install_is_idempotent(self):
        hook = self.root / ".git" / "hooks" / "post-commit"
        hook.write_text("#!/bin/sh\necho mine\n")
        self.assertFalse(index.install_post_commit(self.root, brain=str(self.brain)))
        self.assertEqual(hook.read_text(), "#!/bin/sh\necho mine\n")
        hook.unlink()
        self.assertTrue(index.install_post_commit(self.root, brain=str(self.brain)))
        self.assertTrue(index.install_post_commit(self.root, brain=str(self.brain)))

    def test_one_index_at_a_time(self):
        import os
        release = index.acquire_lock(self.root, wait=0)
        self.assertIsNotNone(release)
        self.assertIsNone(index.acquire_lock(self.root, wait=0.3))          # a second index waits, then gives up
        os.environ["BRAIN_INDEX_LOCKED"] = "1"                              # the hook's own run: it holds the lock
        try:
            self.assertIsNotNone(index.acquire_lock(self.root, wait=0))
        finally:
            del os.environ["BRAIN_INDEX_LOCKED"]
        release()
        again = index.acquire_lock(self.root, wait=0)
        self.assertIsNotNone(again)
        lock = self.root / ".git" / "brain-index.lock"
        os.utime(lock, (time.time() - 1000, time.time() - 1000))            # a crashed run's lock goes stale
        self.assertIsNotNone(index.acquire_lock(self.root, wait=0))

    def test_not_a_git_repo(self):
        self.assertFalse(index.install_post_commit(Path(tempfile.mkdtemp()), brain=str(self.brain)))


if __name__ == "__main__":
    unittest.main()
