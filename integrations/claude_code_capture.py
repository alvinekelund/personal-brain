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
SEEN_PATH = DATA_DIR / "capture-seen.jsonl"
STATE_MAX_AGE_DAYS = 45    # forget watermarks for sessions idle this long
MAX_MESSAGE_AGE_H = 36     # older turns are a resumed stale thread, not today's truth — never re-mine them
# Scheduled tasks and other automation run as sessions too; their "user" text is a prompt,
# not the person talking. Any of these in a message marks the whole session as automation.
AUTOMATION_MARKERS = ("---\nname:", "PushNotification", "You are running Alvin's", "Nightly sync of Alvin",
                      "Morning phone brief", "<<autonomous-loop")

DISTILL_PROMPT = """Below are the messages a user typed to their coding assistant \
during one session. Extract any DURABLE facts about the user worth keeping in \
long-term personal memory: life facts, relationships, preferences, decisions made, \
goals, project commitments.

Rules:
- Only what the user actually stated or clearly decided; never infer or embellish.
- Ignore debugging chatter, one-off instructions, code details, and anything transient.
- NEVER include secrets, API keys, tokens, or passwords.
- Write 1-6 short plain-prose sentences, each a standalone fact, using the user's name.
- Name people, organisations, courses and projects specifically, as the messages do \
("the ac215 GCP project", "Heli Helskyaho", "AC 215"), never as "a new project", \
"my boss" or "the course"; a fact whose subject cannot be named is not worth keeping. \
Where the messages make the canonical name obvious, use it ("the triathlon club at \
aalto" → Aalto Triathlon Club, "the miracle thing" → Miracle Consulting Group); keep \
the user's own wording, in quotes, only when the referent is genuinely unclear.
- When a message states a new attribute of something already known (a role, a status, \
a date), state it as that thing's attribute ("Alvin is the Treasurer of the Aalto \
Triathlon Club"), not as a separate event.
- The messages are in order. If a later message changes or supersedes an earlier plan, \
keep ONLY the later state; never record a plan the user has since moved past.
- If the messages are instructions to an automation (a scheduled task prompt, a workflow), \
not a person talking about their own life, reply with exactly: NONE
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


def _ts(entry: dict) -> float | None:
    """Epoch seconds of a transcript entry's ISO timestamp, if it has one."""
    raw = entry.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def is_automation(text: str) -> bool:
    return any(m in text for m in AUTOMATION_MARKERS)


def _fact_key(fact: str) -> str:
    import hashlib
    norm = " ".join(fact.lower().split()).rstrip(".!")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def seen_facts() -> set[str]:
    try:
        return {json.loads(l)["k"] for l in SEEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def remember_facts(facts: list[str], session: str):
    try:
        with open(SEEN_PATH, "a", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps({"k": _fact_key(fact), "t": int(time.time()), "s": session, "f": fact[:120]}) + "\n")
    except OSError:
        pass


def new_facts_only(facts_text: str) -> tuple[list[str], int]:
    """Split the distiller's prose into sentences and drop any already captured
    (normalized hash). Returns (fresh, dropped). Re-stating a fact in a new
    session must not create a second node or a second inbox item."""
    import re
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", facts_text) if x.strip()]
    seen = seen_facts()
    fresh = [x for x in sentences if _fact_key(x) not in seen]
    return fresh, len(sentences) - len(fresh)


def user_text_from_transcript(path: Path, now: float | None = None,
                              max_age_h: float = MAX_MESSAGE_AGE_H) -> str:
    """Collect ALL human-typed text from a session transcript (JSONL), in order.

    Tool results, slash-command expansions, and harness-injected reminders all
    arrive as "user" entries too — skip them; ambient capture must only ever
    see what the person themselves typed.

    Returns the full text uncapped: the transcript is append-only, so the
    result only ever grows by appending, and the caller's per-session watermark
    (a character offset into this text) stays valid across captures.
    """
    pieces = []
    now = now or time.time()
    cutoff = now - max_age_h * 3600
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user" or entry.get("isMeta"):
            continue
        ts = _ts(entry)
        if ts is not None and ts < cutoff:
            continue   # a resumed old session: those turns were true days ago, not now
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
    if is_automation(text):
        log(f"session {session}: skipped (automation session — scheduled task / workflow prompt)")
        return
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
    fresh, dropped = new_facts_only(facts)
    if not fresh:
        advance()
        log(f"session {session}: all {dropped} distilled fact(s) already captured earlier")
        return

    conn = db.connect()
    try:
        node_ids, edge_ids = extract.ingest(
            conn, " ".join(fresh), source=f"claude-code session {session}", user=config.get_user()
        )
    finally:
        conn.close()
    remember_facts(fresh, session)
    advance()
    log(f"session {session}: ingested {len(node_ids)} node(s), {len(edge_ids)} edge(s) "
        f"({dropped} repeated fact(s) dropped) from: {' '.join(fresh)[:300]}")


if __name__ == "__main__":
    main()
    sys.exit(0)
