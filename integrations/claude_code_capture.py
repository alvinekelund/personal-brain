#!/usr/bin/env python3
"""Claude Code SessionEnd hook — ambient memory capture.

Wired into ~/.claude/settings.json as a SessionEnd hook, this runs when a
Claude Code session ends: it reads what the *user* said during the session,
asks Gemini to distill any durable personal facts (preferences, decisions,
life facts, project commitments — not debugging chatter, not secrets), and
ingests the distillate into the brain through the normal extraction pipeline.

Design constraints:
- Never block or break session exit: every failure path exits 0, silently.
- Capture is summary-level and auditable, not transcript logging — each run
  appends what it ingested (or why it skipped) to ~/.personal-brain/capture.log.
- The brain's dedup means re-stating a known fact reinforces it, so repeated
  sessions converge instead of accumulating duplicates.
- Each session carries an ingest watermark (~/.personal-brain/capture-state.json):
  a long-lived session that ends repeatedly (resume → exit → resume) only ever
  distills the turns typed since its last capture, instead of re-mining the whole
  transcript — which is what fed the same Aug 31 action item into the loop inbox
  three times on Sep 1 2026.

Stdin: hook JSON ({"transcript_path": ..., "session_id": ...}).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import DATA_DIR, config, db, extract, llm

MIN_USER_CHARS = 200       # below this the new turns can't contain much worth keeping
MAX_USER_CHARS = 15000     # cap what we send to the distiller
PER_MESSAGE_CAP = 2000
LOG_PATH = DATA_DIR / "capture.log"
STATE_PATH = DATA_DIR / "capture-state.json"
STATE_MAX_AGE_DAYS = 45    # forget watermarks for sessions idle this long

DISTILL_PROMPT = """Below are the messages a user typed to their coding assistant \
during one session. Extract any DURABLE facts about the user worth keeping in \
long-term personal memory: life facts, relationships, preferences, decisions made, \
goals, project commitments.

Rules:
- Only what the user actually stated or clearly decided; never infer or embellish.
- Ignore debugging chatter, one-off instructions, code details, and anything transient.
- NEVER include secrets, API keys, tokens, or passwords.
- Write 1-6 short plain-prose sentences, each a standalone fact, using the user's name.
- If the session contains nothing durably worth remembering, reply with exactly: NONE

The user's name is {user}.

User messages:
{messages}"""


def log(line: str):
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {line}\n")
    except OSError:
        pass


def load_state() -> dict:
    """Per-session watermarks: {session_id: {"chars": <user text already mined>, "ts": ...}}.
    A missing or corrupt file is an empty state, never an error."""
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_state(state: dict):
    """Persist watermarks, pruning sessions idle past STATE_MAX_AGE_DAYS."""
    cutoff = time.time() - STATE_MAX_AGE_DAYS * 86400
    state = {sid: rec for sid, rec in state.items()
             if isinstance(rec, dict) and rec.get("ts", 0) >= cutoff}
    try:
        STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError:
        pass


def user_text_from_transcript(path: Path) -> str:
    """Collect ALL human-typed text from a session transcript (JSONL), in order.

    Tool results, slash-command expansions, and harness-injected reminders all
    arrive as "user" entries too — skip them; ambient capture must only ever
    see what the person themselves typed.

    Returns the full text uncapped: the transcript is append-only, so the
    result only ever grows by appending, and the caller's per-session watermark
    (a character offset into this text) stays valid across captures.
    """
    pieces = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user" or entry.get("isMeta"):
            continue
        content = (entry.get("message") or {}).get("content")
        blocks = [content] if isinstance(content, str) else (
            [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if isinstance(content, list) else []
        )
        for text in blocks:
            text = (text or "").strip()
            if not text or text.startswith(("<command-", "<local-command", "<system-reminder",
                                            "<task-notification")):
                continue
            pieces.append(text[:PER_MESSAGE_CAP])
    return "\n---\n".join(pieces)


def main():
    """Run the capture; never raises — a hook must not disturb session exit."""
    try:
        _capture()
    except Exception as e:
        log(f"error: {type(e).__name__}: {e}")


def _capture():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return
    transcript = Path(hook_input.get("transcript_path") or "")
    session_id = hook_input.get("session_id") or "unknown"
    session = session_id[:8]
    if not transcript.is_file():
        return
    if not llm.have_key():
        log(f"session {session}: skipped (no GEMINI_API_KEY)")
        return

    text = user_text_from_transcript(transcript)
    state = load_state()
    rec = state.get(session_id)
    mark = rec.get("chars", 0) if isinstance(rec, dict) else 0
    if not isinstance(mark, int) or not 0 <= mark <= len(text):
        mark = 0   # transcript replaced or state damaged — re-mine from the start
    new = text[mark:]
    if len(new) < MIN_USER_CHARS:
        log(f"session {session}: skipped ({len(new)} new chars of user text, "
            f"{mark} already captured)")
        return

    user = config.get_user() or "the user"
    facts = llm.generate(DISTILL_PROMPT.format(user=user, messages=new[-MAX_USER_CHARS:])).strip()

    def advance():
        """Mark these turns as mined. Called only once they were actually handled —
        an LLM/ingest failure leaves the watermark alone so the next end retries."""
        state[session_id] = {"chars": len(text), "ts": time.time()}
        save_state(state)

    if not facts or facts.upper().startswith("NONE"):
        advance()
        log(f"session {session}: nothing durable")
        return

    conn = db.connect()
    try:
        node_ids, edge_ids = extract.ingest(
            conn, facts, source=f"claude-code session {session}", user=config.get_user()
        )
    finally:
        conn.close()
    advance()
    log(f"session {session}: ingested {len(node_ids)} node(s), {len(edge_ids)} edge(s) "
        f"from: {facts[:300]}")


if __name__ == "__main__":
    main()
    sys.exit(0)
