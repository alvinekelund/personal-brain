"""Tests for the Claude Code SessionEnd capture hook — stdlib only, no network."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.config as config
import brain.db as db
import brain.llm as llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integrations"))
import claude_code_capture as capture


def transcript(tmp, entries):
    p = Path(tmp) / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return p


def user_msg(text):
    return {"type": "user", "message": {"role": "user",
                                        "content": [{"type": "text", "text": text}]}}


class CaptureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp, "brain.db")
        self._orig_config_path = config.CONFIG_PATH
        config.CONFIG_PATH = Path(self._tmp) / "config.json"
        self._orig_log = capture.LOG_PATH
        capture.LOG_PATH = Path(self._tmp) / "capture.log"
        self._orig_have_key = llm.have_key
        self._orig_generate = llm.generate
        self._orig_embed = llm.embed
        llm.have_key = lambda: False

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        config.CONFIG_PATH = self._orig_config_path
        capture.LOG_PATH = self._orig_log
        llm.have_key = self._orig_have_key
        llm.generate = self._orig_generate
        llm.embed = self._orig_embed

    def run_hook(self, transcript_path, session_id="testsess"):
        sys.stdin = io.StringIO(json.dumps(
            {"transcript_path": str(transcript_path), "session_id": session_id}))
        try:
            capture.main()
        finally:
            sys.stdin = sys.__stdin__

    def log_text(self):
        return capture.LOG_PATH.read_text() if capture.LOG_PATH.exists() else ""


class TranscriptParsingTests(CaptureTestCase):
    def test_collects_only_human_typed_text(self):
        t = transcript(self._tmp, [
            user_msg("I prefer functional style"),
            {"type": "user", "message": {"role": "user", "content": "plain string message"}},
            {"type": "assistant", "message": {"role": "assistant",
                                              "content": [{"type": "text", "text": "noted"}]}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "SECRET_TOKEN=abc123"}]}},
            {"type": "user", "isMeta": True,
             "message": {"role": "user", "content": "meta noise"}},
            user_msg("<system-reminder>injected</system-reminder>"),
            user_msg("<command-message>loop</command-message>"),
        ])
        text = capture.user_text_from_transcript(t)
        self.assertIn("functional style", text)
        self.assertIn("plain string message", text)
        self.assertNotIn("SECRET_TOKEN", text)   # tool results never reach the distiller
        self.assertNotIn("noted", text)          # assistant text excluded
        self.assertNotIn("meta noise", text)
        self.assertNotIn("injected", text)
        self.assertNotIn("command-message", text)

    def test_tolerates_malformed_lines(self):
        p = Path(self._tmp) / "broken.jsonl"
        p.write_text("not json\n" + json.dumps(user_msg("still works")))
        self.assertIn("still works", capture.user_text_from_transcript(p))

    def test_long_session_keeps_the_tail(self):
        msgs = [user_msg(f"message number {i} " + "x" * 500) for i in range(60)]
        text = capture.user_text_from_transcript(transcript(self._tmp, msgs))
        self.assertLessEqual(len(text), capture.MAX_USER_CHARS)
        self.assertIn("message number 59", text)


class HookBehaviourTests(CaptureTestCase):
    def test_ingests_distilled_facts(self):
        llm.have_key = lambda: True
        responses = iter([
            "Alvin prefers functional programming style.",  # distill
            json.dumps({"nodes": [{"name": "Functional programming", "type": "concept",
                                   "content": "Preferred style.", "confidence": 0.9,
                                   "importance": 0.5}], "edges": []}),  # extraction
            "{}",  # entity linking
        ])
        llm.generate = lambda *a, **k: next(responses, "{}")
        llm.embed = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        t = transcript(self._tmp, [user_msg(
            "For the record, I strongly prefer functional programming style "
            "and want all my projects written that way going forward, "
            "it is a durable preference of mine worth remembering."), user_msg(
            "Also please refactor the parser module today and remember that "
            "functional style applies to it as well as everything else we build.")])
        self.run_hook(t)
        conn = db.connect()
        self.assertIsNotNone(db.get_node_by_name(conn, "Functional programming"))
        conn.close()
        self.assertIn("ingested", self.log_text())

    def test_none_response_skips_ingestion(self):
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: "NONE"
        t = transcript(self._tmp, [user_msg("x" * 300)])
        self.run_hook(t)
        conn = db.connect()
        self.assertEqual(db.stats(conn)["total"], 0)
        conn.close()
        self.assertIn("nothing durable", self.log_text())

    def test_tiny_session_skipped_without_llm_call(self):
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: self.fail("must not call the LLM for a tiny session")
        t = transcript(self._tmp, [user_msg("ok thanks")])
        self.run_hook(t)
        self.assertIn("skipped", self.log_text())

    def test_no_key_skips_silently(self):
        t = transcript(self._tmp, [user_msg("x" * 300)])
        self.run_hook(t)
        self.assertIn("no GEMINI_API_KEY", self.log_text())

    def test_missing_transcript_is_a_noop(self):
        self.run_hook(Path(self._tmp) / "does-not-exist.jsonl")
        self.assertEqual(self.log_text(), "")

    def test_llm_failure_never_raises(self):
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quota gone"))
        t = transcript(self._tmp, [user_msg("x" * 300)])
        self.run_hook(t)  # must swallow the RuntimeError
        self.assertIn("error: RuntimeError: quota gone", self.log_text())


if __name__ == "__main__":
    unittest.main()
