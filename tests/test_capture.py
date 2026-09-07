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


def user_msg(text, timestamp=None):
    e = {"type": "user", "message": {"role": "user",
                                     "content": [{"type": "text", "text": text}]}}
    if timestamp:
        e["timestamp"] = timestamp
    return e


class CaptureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp, "brain.db")
        self._orig_config_path = config.CONFIG_PATH
        config.CONFIG_PATH = Path(self._tmp) / "config.json"
        (Path(self._tmp) / "vault").mkdir()
        config.save({"vault_dir": str(Path(self._tmp) / "vault")})   # never the real vault
        self._orig_seen = capture.SEEN_PATH
        capture.SEEN_PATH = Path(self._tmp) / "capture-seen.jsonl"
        self._orig_log = capture.LOG_PATH
        capture.LOG_PATH = Path(self._tmp) / "capture.log"
        self._orig_state = capture.STATE_PATH
        capture.STATE_PATH = Path(self._tmp) / "capture-state.json"
        self._orig_have_key = llm.have_key
        self._orig_generate = llm.generate
        self._orig_embed = llm.embed
        llm.have_key = lambda: False

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        config.CONFIG_PATH = self._orig_config_path
        capture.SEEN_PATH = self._orig_seen
        capture.LOG_PATH = self._orig_log
        capture.STATE_PATH = self._orig_state
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


class DistillPromptTests(unittest.TestCase):
    def test_prompt_demands_specific_names_and_attributes(self):
        """A distilled fact saying "a new project" became the graph node
        "New Project (Harvard)" on Sep 6 2026; the extractor now refuses such
        names, but the distiller must produce nameable facts in the first place."""
        p = capture.DISTILL_PROMPT
        for phrase in ("Name people, organisations, courses and projects specifically",
                       "a new project", "cannot be named is not worth keeping",
                       "new attribute of something already known", "NEVER include secrets",
                       "never one fact per name"):                    # 17 HackMIT sponsors became 17 org nodes on Sep 2
            self.assertIn(phrase, " ".join(p.split()))
        self.assertIn("{user}", p)
        self.assertIn("{messages}", p)


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

    def test_full_text_returned_in_order(self):
        """The extractor returns everything (uncapped) so the caller's watermark —
        a character offset — stays valid; the distill cap is applied at the call."""
        msgs = [user_msg(f"message number {i} " + "x" * 500) for i in range(60)]
        text = capture.user_text_from_transcript(transcript(self._tmp, msgs))
        self.assertIn("message number 0", text)
        self.assertIn("message number 59", text)
        self.assertLess(text.index("message number 0"), text.index("message number 59"))


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


class WatermarkTests(CaptureTestCase):
    """A session that ends repeatedly must only mine the turns typed since its
    last capture — re-distilling the whole transcript is what pushed the same
    Aug 31 action item into the loop inbox three times on Sep 1 2026."""

    def fake_llm(self, distill_reply="Alvin lives in Boston."):
        """generate() stub: records every distill prompt, answers the rest of
        the pipeline (extraction, entity linking) with harmless empty JSON."""
        calls = []

        def gen(prompt, *a, **k):
            if prompt.startswith("Below are the messages"):
                calls.append(prompt)
                return distill_reply
            return "{}"

        llm.have_key = lambda: True
        llm.generate = gen
        llm.embed = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        return calls

    def test_repeat_capture_skips_already_mined_turns(self):
        t = transcript(self._tmp, [user_msg("Deciding on MIT cross-registration. " + "x" * 300)])
        calls = self.fake_llm()
        self.run_hook(t, session_id="5f2be1cf-0000-0000")
        self.assertEqual(len(calls), 1)
        self.run_hook(t, session_id="5f2be1cf-0000-0000")   # ends again, no new turns
        self.assertEqual(len(calls), 1)                      # nothing re-mined
        self.assertIn("already captured", self.log_text())

    def test_recapture_distills_only_new_turns(self):
        entries = [user_msg("OLD TURN about course planning. " + "x" * 300)]
        t = transcript(self._tmp, entries)
        calls = self.fake_llm()
        self.run_hook(t, session_id="sess-incr")
        entries.append(user_msg("NEW TURN about the marathon. " + "y" * 300))
        transcript(self._tmp, entries)
        self.run_hook(t, session_id="sess-incr")
        self.assertEqual(len(calls), 2)
        self.assertIn("NEW TURN", calls[1])
        self.assertNotIn("OLD TURN", calls[1])

    def test_distill_input_capped_to_the_tail(self):
        msgs = [user_msg(f"message number {i} " + "x" * 500) for i in range(60)]
        t = transcript(self._tmp, msgs)
        calls = self.fake_llm()
        self.run_hook(t)
        self.assertLessEqual(len(calls[0]), capture.MAX_USER_CHARS + len(capture.DISTILL_PROMPT) + 100)
        self.assertIn("message number 59", calls[0])

    def test_failed_distill_is_retried_on_the_next_capture(self):
        t = transcript(self._tmp, [user_msg("z" * 300)])
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down"))
        self.run_hook(t, session_id="sess-retry")            # watermark must not advance
        calls = self.fake_llm()
        self.run_hook(t, session_id="sess-retry")
        self.assertEqual(len(calls), 1)                      # same turns offered again

    def test_nothing_durable_still_advances_watermark(self):
        t = transcript(self._tmp, [user_msg("w" * 300)])
        calls = self.fake_llm(distill_reply="NONE")
        self.run_hook(t, session_id="sess-none")
        self.run_hook(t, session_id="sess-none")
        self.assertEqual(len(calls), 1)                      # not re-asked about the same turns

    def test_watermarks_are_per_session(self):
        t = transcript(self._tmp, [user_msg("q" * 300)])
        calls = self.fake_llm()
        self.run_hook(t, session_id="sess-a")
        self.run_hook(t, session_id="sess-b")                # same transcript, other session
        self.assertEqual(len(calls), 2)

    def test_stale_sessions_pruned_from_state(self):
        import time as _time
        capture.STATE_PATH.write_text(json.dumps(
            {"ancient": {"chars": 5, "ts": _time.time() - 90 * 86400}}))
        t = transcript(self._tmp, [user_msg("p" * 300)])
        self.fake_llm()
        self.run_hook(t, session_id="sess-new")
        state = json.loads(capture.STATE_PATH.read_text())
        self.assertNotIn("ancient", state)
        self.assertIn("sess-new", state)


class StalenessAndDedupTests(CaptureTestCase):
    LONG = ("I have decided to enroll in the robotics course this fall and keep the statistics "
            "course as well, this is my settled plan going forward for the semester. ")

    def test_turns_older_than_the_window_are_ignored(self):
        import datetime
        old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).isoformat().replace("+00:00", "Z")
        new = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        t = transcript(self._tmp, [user_msg("Ancient plan: pursue 6.7960 " + self.LONG, old),
                                   user_msg("Fresh: " + self.LONG, new),
                                   user_msg("No timestamp at all " + self.LONG)])
        text = capture.user_text_from_transcript(t)
        self.assertNotIn("Ancient plan", text)
        self.assertIn("Fresh:", text)
        self.assertIn("No timestamp", text)     # untimestamped turns are kept (can't judge them)

    def test_automation_sessions_are_skipped_without_llm(self):
        llm.have_key = lambda: True
        llm.generate = lambda *a, **k: self.fail("automation prompts must never reach the distiller")
        t = transcript(self._tmp, [user_msg("---\nname: nightly-brain-sync\n---\nNightly sync of Alvin's brain " + self.LONG * 3)])
        self.run_hook(t)
        self.assertIn("automation session", self.log_text())
        self.assertTrue(capture.is_automation("Send it with the PushNotification tool"))
        self.assertFalse(capture.is_automation("I want to push my notification settings"))

    def test_repeated_fact_is_not_ingested_twice(self):
        llm.have_key = lambda: True
        calls = {"ingest": 0}
        def gen(prompt, *a, **k):
            if "DURABLE facts" in prompt:
                return "Alvin keeps STAT 211 this fall. Alvin's boss is Heli."
            calls["ingest"] += 1
            return json.dumps({"nodes": [{"name": "STAT 211", "type": "concept", "content": "kept"}], "edges": []}) \
                if calls["ingest"] % 2 else "{}"
        llm.generate = gen
        llm.embed = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        self.run_hook(transcript(self._tmp, [user_msg("First session " + self.LONG * 2)]), session_id="s1")
        self.assertIn("ingested", self.log_text())
        self.assertEqual(len(capture.seen_facts()), 2)
        # a second session restating the same two facts → nothing reaches the extractor
        before = calls["ingest"]
        self.run_hook(transcript(self._tmp, [user_msg("Second session " + self.LONG * 2)]), session_id="s2")
        self.assertEqual(calls["ingest"], before)
        self.assertIn("already captured", self.log_text())
        fresh, dropped = capture.new_facts_only("Alvin keeps STAT 211 this fall. Alvin moved to Boston.")
        self.assertEqual((fresh, dropped), (["Alvin moved to Boston."], 1))


if __name__ == "__main__":
    unittest.main()
