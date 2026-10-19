"""iMessage to yourself — the brain's phone channel (Sep 8 2026).

The Claude app's push notification reaches the phone only while Remote
Control is connected and leaves no trace; an iMessage sent from this Mac's
Messages.app to Alvin's own Apple ID handle lands on the iPhone like any text
and is logged here. The script names the iMessage account explicitly and the
handle is an email, so nothing can fall back to SMS (no carrier charges; the
US number was "Not Delivered" anyway). Nothing here talks to the network:
Messages.app does.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from brain import DATA_DIR, config

LOG_PATH = DATA_DIR / "push.log"
HELD_PATH = DATA_DIR / "push-held.jsonl"   # pushes held in quiet hours, one JSON line each, sent at QUIET_END
QUIET_START, QUIET_END = 23, 7   # local hours (Alvin, Oct 19 2026: routines run all night now, so a push
                                 # between 23:00 and 07:00 waits for the morning unless it is --urgent,
                                 # i.e. a deadline due before 07:00)
CONFIG_KEY = "push_to"          # the iMessage handle (phone number or Apple ID email)
EMAIL_KEY = "push_email"        # optional: an email copy of every push (Alvin's Instinct address), via Mail.app
EMAIL_FROM_KEY = "push_email_from"   # the Mail.app account address to send from (default: the first account)
SUBJECT_MAX = 60
MAX_CHARS = 1000                # iMessage carries more, but a push should stay a note
TIMEOUT = 90.0           # Messages.app can take ~45 s to hand a message to iMessage when it has
                         # been idle (Oct 18 2026: two nightly-sync pushes timed out at 25 s, then
                         # the same osascript sent in ~45 s when run by hand)
DEDUPE_HOURS = 6.0       # an identical note to the same recipient within this window is not sent again
                         # (Oct 12 2026: one "run stopped" alert reached the phone twice, 66 min apart)
RUNNER = subprocess.run   # looked up at call time, so tests can replace it and never reach Messages.app

_SCRIPT = [
    "on run argv",
    "set h to item 1 of argv",
    "set t to item 2 of argv",
    'tell application "Messages"',
    "set acct to 1st account whose service type = iMessage",
    "set p to participant h of acct",
    "send t to p",
    "end tell",
    'return "sent"',
    "end run",
]


_MAIL_SCRIPT = [
    "on run argv",
    "set a to item 1 of argv",
    "set s to item 2 of argv",
    "set t to item 3 of argv",
    "set f to item 4 of argv",
    'tell application "Mail"',
    "set m to make new outgoing message with properties {subject:s, content:t, visible:false}",
    "if f is not \"\" then set sender of m to f",
    "tell m to make new to recipient with properties {address:a}",
    "send m",
    "end tell",
    'return "sent"',
    "end run",
]


def email_handle() -> str:
    """The optional email copy's recipient — set with `brain push --set-email`."""
    return (config.load().get(EMAIL_KEY) or "").strip()


def set_email(value: str, sender: str = "") -> str:
    value = (value or "").strip()
    if not value or "@" not in value:
        raise ValueError("an email address is needed (or 'off' to stop the email copy)")
    cfg = config.load()
    cfg[EMAIL_KEY] = value
    if sender.strip():
        cfg[EMAIL_FROM_KEY] = sender.strip()
    config.save(cfg)
    return value


def clear_email():
    cfg = config.load()
    cfg.pop(EMAIL_KEY, None)
    config.save(cfg)


def send_email(text: str, to: str = "", runner=None, log_path: Path | None = None) -> str | None:
    """Email `text` to `to` (default: the configured email copy) through this
    Mac's Mail.app, from the configured account (or Mail's default). Free, no
    credentials in the brain. Returns None on success, else the reason; logged."""
    runner = runner or RUNNER
    log_path = log_path or LOG_PATH
    text = (text or "").strip()
    to = (to or email_handle()).strip()
    if not text:
        return "nothing to send"
    if not to:
        return "no email recipient: `brain push --set-email <address>`"
    subject = "brain: " + " ".join(text.split())[:SUBJECT_MAX]
    sender = (config.load().get(EMAIL_FROM_KEY) or "").strip()
    args = ["osascript"]
    for line in _MAIL_SCRIPT:
        args += ["-e", line]
    args += ["--", to, subject, text, sender]
    try:
        r = runner(args, capture_output=True, text=True, timeout=TIMEOUT)
        err = None if r.returncode == 0 and "sent" in (r.stdout or "") else (r.stderr or r.stdout or "osascript failed").strip()
    except (OSError, subprocess.SubprocessError) as e:
        err = str(e)
    _log(log_path, ("sent" if err is None else f"FAILED ({err[:120]})") + f" → email {to}: {' '.join(text.split())}")
    return err


def handle() -> str:
    """The configured recipient — Alvin's own number, set with `brain push --set-to`."""
    return (config.load().get(CONFIG_KEY) or "").strip()


def set_handle(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("a phone number (+13392221646) or Apple ID email is needed")
    cfg = config.load()
    cfg[CONFIG_KEY] = value
    config.save(cfg)
    return value


def recent_duplicate(text: str, to: str = "", hours: float = DEDUPE_HOURS, log_path: Path | None = None,
                     now: float | None = None) -> str | None:
    """When the same note went to the same recipient within `hours`, the time it was sent
    ('YYYY-MM-DD HH:MM:SS'); else None. Reads the push log, newest first."""
    log_path = log_path or LOG_PATH
    text = " ".join((text or "").split())
    to = (to or handle()).strip()
    if not text or not to or hours <= 0:
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS - 1] + "…"
    now = now or time.time()
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()[-300:]
    except OSError:
        return None
    for ln in reversed(lines):
        try:
            t = time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if now - t > hours * 3600:
            break                                 # the log is in time order: everything older is older still
        if ln[20:] == f"sent → {to}: {text}":
            return ln[:19]
    return None


def send(text: str, to: str = "", runner=None, log_path: Path | None = None,
         dedupe_hours: float | None = None) -> str | None:
    """Send `text` as an iMessage to `to` (default: the configured handle).
    Returns None on success (or when the identical note already went out within
    `dedupe_hours`, default DEDUPE_HOURS; 0 always sends), else the reason. Every attempt is logged."""
    runner = runner or RUNNER
    log_path = log_path or LOG_PATH           # resolved now, so a test's LOG_PATH patch is honoured
    text = " ".join((text or "").split())
    to = (to or handle()).strip()
    if not text:
        return "nothing to send"
    if not to:
        return "no recipient: run `brain push --set-to <your number>` once"
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS - 1] + "…"
    dup = recent_duplicate(text, to, DEDUPE_HOURS if dedupe_hours is None else dedupe_hours, log_path)
    if dup:
        _log(log_path, f"SKIPPED (same note sent {dup}) → {to}: {text}")
        return None
    args = ["osascript"]
    for line in _SCRIPT:
        args += ["-e", line]
    args += ["--", to, text]
    try:
        r = runner(args, capture_output=True, text=True, timeout=TIMEOUT)
        err = None if r.returncode == 0 and "sent" in (r.stdout or "") else (r.stderr or r.stdout or "osascript failed").strip()
    except (OSError, subprocess.SubprocessError) as e:
        err = str(e)
    _log(log_path, ("sent" if err is None else f"FAILED ({err[:120]})") + f" → {to}: {text}")
    return err


def quiet_now(now: datetime | None = None) -> bool:
    """True between QUIET_START and QUIET_END local time."""
    h = (now or datetime.now()).hour
    return h >= QUIET_START or h < QUIET_END


def hold(text: str, to: str = "", email: bool = True, held_path: Path | None = None,
         log_path: Path | None = None) -> str | None:
    """Keep `text` for the morning instead of sending it now. Returns None, else the reason."""
    held_path = held_path or HELD_PATH
    log_path = log_path or LOG_PATH
    text = " ".join((text or "").split())
    to = (to or handle()).strip()
    if not text:
        return "nothing to send"
    if not to:
        return "no recipient: run `brain push --set-to <your number>` once"
    try:
        held_path.parent.mkdir(parents=True, exist_ok=True)
        with open(held_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "to": to, "text": text,
                                "email": bool(email)}) + "\n")
    except OSError as e:
        return str(e)
    _log(log_path, f"HELD (quiet {QUIET_START:02d}-{QUIET_END:02d}) → {to}: {text}")
    return None


def held(held_path: Path | None = None) -> list[dict]:
    try:
        lines = (held_path or HELD_PATH).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def flush_held(runner=None, held_path: Path | None = None, log_path: Path | None = None) -> list[str]:
    """Send what quiet hours held: one iMessage per recipient ("Overnight: a | b"), plus the email
    copy when any of them asked for one. Kept for the next try if the send fails. Returns the errors."""
    held_path = held_path or HELD_PATH
    items = held(held_path)
    if not items:
        return []
    by_to: dict[str, list[dict]] = {}
    for it in items:
        by_to.setdefault(it.get("to") or "", []).append(it)
    errors, keep = [], []
    for to, its in by_to.items():
        texts = list(dict.fromkeys(it["text"] for it in its))   # the same note twice goes once
        text = texts[0] if len(texts) == 1 else "Overnight: " + " | ".join(texts)
        err = send(text, to=to, runner=runner, log_path=log_path)
        if err:
            errors.append(err)
            keep += its
        elif any(it.get("email") for it in its) and email_handle():
            send_email(text, runner=runner, log_path=log_path)
    try:
        if keep:
            held_path.write_text("".join(json.dumps(it) + "\n" for it in keep), encoding="utf-8")
        else:
            held_path.unlink()
    except OSError:
        pass
    return errors


def _log(log_path: Path, line: str):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    except OSError:
        pass
