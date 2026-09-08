"""iMessage to yourself — the brain's phone channel (Sep 8 2026).

The Claude app's push notification reaches the phone only while Remote
Control is connected and leaves no trace; an iMessage sent from this Mac's
Messages.app to Alvin's own number lands on the iPhone like any text and is
logged here. Nothing in this module talks to the network: Messages.app does.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from brain import DATA_DIR, config

LOG_PATH = DATA_DIR / "push.log"
CONFIG_KEY = "push_to"          # the iMessage handle (phone number or Apple ID email)
MAX_CHARS = 1000                # iMessage carries more, but a push should stay a note
TIMEOUT = 25.0
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


def send(text: str, to: str = "", runner=None, log_path: Path | None = None) -> str | None:
    """Send `text` as an iMessage to `to` (default: the configured handle).
    Returns None on success, else the reason. Every attempt is logged."""
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


def _log(log_path: Path, line: str):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    except OSError:
        pass
