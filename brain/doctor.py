"""`brain doctor` — is the brain actually wired up, and is anything stale?

On Aug 31 2026 the graph layer died silently for a whole day: the interpreter
holding the `brain` entrypoint was deleted, and every hook swallowed the error.
This module exists so that can never be silent again. It checks the things that
break in practice — the binary the hooks point at, the database, the API key,
the vault files and their freshness, the loop/decision ledgers, and every
Claude Code wiring location (hooks, MCP registration, scheduled tasks) — and
reports ok / warn / fail per check. `brain today` prints the one-line summary
at the top of every session, so a broken brain announces itself.

All path arguments are injectable so the checks run hermetically in tests.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from brain import DATA_DIR, DB_PATH, decisions, llm, loops

STALE_HOURS = 48
EXPECTED_BIN = DATA_DIR / "venv" / "bin" / "brain"
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
CLAUDE_JSON = Path.home() / ".claude.json"
SCHEDULED_TASKS = Path.home() / ".claude" / "scheduled-tasks"
# Only count a path as absolute when nothing word-like precedes it, so a relative
# mention such as "areas/brain.md" is not read as the path "/brain.md".
ABS_PATH_RE = re.compile(r"(?<![\w.~-])(~?/[\w./+-]+)")


@dataclass
class Check:
    name: str
    status: str      # ok | warn | fail
    detail: str

    @property
    def icon(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.status]


def _age_h(ts: float, now: float) -> float:
    return max(0.0, (now - ts) / 3600.0)


def check_binary(expected: Path = EXPECTED_BIN) -> Check:
    if expected.is_file() and os.access(expected, os.X_OK):
        running = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
        note = "" if not running or running == expected.resolve() else f" (this run: {running})"
        return Check("binary", "ok", f"{expected}{note}")
    return Check("binary", "fail", f"{expected} missing — reinstall: "
                 f"python3.12 -m venv {DATA_DIR}/venv && {DATA_DIR}/venv/bin/pip install -e ~/Projects/brain/personal-brain")


def check_db(db_path: Path = DB_PATH, now: float | None = None) -> Check:
    now = now or time.time()
    if not Path(db_path).is_file():
        return Check("graph", "fail", f"{db_path} missing")
    try:
        conn = sqlite3.connect(str(db_path))
        active = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        last = conn.execute("SELECT MAX(ingested_at) FROM ingestion_log").fetchone()[0]
        conn.close()
    except sqlite3.Error as e:
        return Check("graph", "fail", f"cannot read {db_path}: {e}")
    if last is None:
        return Check("graph", "warn", f"{active} nodes, never ingested")
    age = _age_h(last, now)
    status = "warn" if age > STALE_HOURS else "ok"
    return Check("graph", status, f"{active} nodes · last ingest {age:.0f}h ago")


API_PROBE_URL = "https://generativelanguage.googleapis.com/"


def _default_probe():
    import urllib.request
    urllib.request.urlopen(API_PROBE_URL, timeout=6, context=llm.ssl_context())


def check_api(probe=None) -> Check:
    """Can we actually complete a TLS handshake with the Gemini host? A missing
    CA bundle (python.org builds) fails here long before any key is checked —
    exactly what silently broke every ingest after the Sep 2026 venv move."""
    import urllib.error
    try:
        (probe or _default_probe)()
    except urllib.error.HTTPError:
        return Check("gemini-api", "ok", "TLS handshake ok")   # 404 on / is fine: we reached it
    except Exception as e:
        msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate" in msg.lower():
            return Check("gemini-api", "fail", f"TLS trust broken ({type(e).__name__}) — "
                         f"run: {DATA_DIR}/venv/bin/pip install certifi")
        return Check("gemini-api", "warn", f"unreachable: {msg[:80]} (offline?)")
    return Check("gemini-api", "ok", "reachable")


def check_graph_integrity(db_path: Path = DB_PATH, user: str = "") -> Check:
    """Is the hierarchy still a tree? Orphans, multi-parent nodes, unrooted categories,
    cycles, legacy task nodes, near-duplicate names, missing embeddings."""
    import sqlite3
    from brain import config, integrity
    if not Path(db_path).is_file():
        return Check("graph-tree", "fail", f"{db_path} missing")
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rep = integrity.check(conn, user or config.get_user())
        conn.close()
    except sqlite3.Error as e:
        return Check("graph-tree", "fail", f"cannot read graph: {e}")
    if rep.structural or rep.legacy_tasks:
        return Check("graph-tree", "fail", rep.summary() + " — `brain doctor --repair`")
    if rep.duplicates or rep.missing_embeddings or rep.oversized or rep.flat_lists or rep.thin_areas:
        detail = rep.summary()
        if rep.duplicates:
            detail += ": " + "; ".join(f"{a} ~ {b}" for a, b in rep.duplicates[:3]) + " (brain merge)"
        return Check("graph-tree", "warn", detail)
    return Check("graph-tree", "ok", rep.summary())


def check_capture(log_path: Path = DATA_DIR / "capture.log", now: float | None = None) -> Check:
    """Ambient capture: did the last runs succeed, and when did one last ingest?"""
    import re
    now = now or time.time()
    if not Path(log_path).is_file():
        return Check("capture", "warn", "no capture.log yet — the SessionEnd hook has never run")
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    recent_err = [l for l in lines if " error:" in l]
    last = lines[-1] if lines else ""
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", last)
    age_h = None
    if m:
        try:
            age_h = _age_h(time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), now)
        except ValueError:
            age_h = None
    if recent_err and recent_err[-1] == last:
        return Check("capture", "fail", f"last run failed: {last[20:140]}")
    tail = last[20:110]
    if age_h is not None and age_h > 72:
        return Check("capture", "warn", f"last run {age_h:.0f}h ago: {tail}")
    return Check("capture", "ok", f"last run {age_h:.0f}h ago: {tail}" if age_h is not None else tail)


BRIEF_MAX_AGE_H = 30   # the morning brief runs daily; older than this and it did not run or did not record


def check_brief(log_path: Path = DATA_DIR / "brief.log", now: float | None = None) -> Check:
    """The phone brief: `brain today --brief` records every line it produces in
    brief.log, so a review can see what was pushed and when. No file, or a
    line older than a day and a bit, means the morning brief did not run or
    its prompt did not call `brain today --brief`."""
    import re
    now = now or time.time()
    if not Path(log_path).is_file():
        return Check("brief", "warn", "no brief.log yet — no `brain today --brief` has run (the morning brief should)")
    lines = [l for l in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)", last)
    if not m:
        return Check("brief", "warn", "brief.log is empty or unreadable")
    try:
        age_h = _age_h(time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), now)
    except ValueError:
        return Check("brief", "warn", "brief.log has an unreadable timestamp")
    text = m.group(2)[:90]
    if age_h > BRIEF_MAX_AGE_H:
        return Check("brief", "warn", f"last brief {age_h:.0f}h ago — the morning brief did not run or did not record: {text}")
    return Check("brief", "ok", f"last brief {age_h:.0f}h ago: {text}")


def check_index(db_path: Path = DB_PATH, root: Path | None = None) -> Check:
    """Is the vault index current? (D-014: the directory is the brain; the graph
    routes questions to its files, so a stale index sends `brain ask` to old text.)"""
    from brain import config, index as vindex
    root = Path(root) if root is not None else config.vault_dir()
    if not Path(db_path).is_file() or not root.is_dir():
        return Check("vault-index", "warn", "no graph or vault to index yet")
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        s = vindex.status(conn, root)
        conn.close()
    except sqlite3.Error as e:
        return Check("vault-index", "fail", f"cannot read the index: {e}")
    if not s["indexed"]:
        return Check("vault-index", "warn", f"{s['on_disk']} vault file(s), none indexed — run `brain index`")
    changed = len(s["new"]) + len(s["stale"]) + len(s["removed"])
    if changed:
        return Check("vault-index", "warn", f"{changed} vault file(s) changed since the last index — run `brain index`")
    ledger_stale = s.get("ledger_stale") or []
    if ledger_stale:
        return Check("vault-index", "warn",
                     f"{len(ledger_stale)} ledger line(s) changed since the last index ({', '.join(ledger_stale[:4])}"
                     f"{'…' if len(ledger_stale) > 4 else ''}) — run `brain index`")
    return Check("vault-index", "ok", f"{s['indexed']} files · {s['node_links']} node links · {s['embedded']} embedded"
                 f" · {s.get('ledger_embedded', 0)}/{s.get('ledger_total', 0)} ledger lines")


def check_key() -> Check:
    return (Check("gemini-key", "ok", "present") if llm.have_key()
            else Check("gemini-key", "warn", "no GEMINI_API_KEY — brain add / capture / ask are disabled"))


def check_vault(root: Path, today: date | None = None, now: float | None = None) -> list[Check]:
    root = Path(root)
    now = now or time.time()
    out: list[Check] = []
    if not root.is_dir():
        return [Check("vault", "fail", f"{root} missing")]
    now_md = root / loops.NOW_FILE
    if not now_md.is_file():
        out.append(Check("vault", "fail", "NOW.md missing"))
        return out
    age = _age_h(now_md.stat().st_mtime, now)
    out.append(Check("vault-activity", "warn" if age > STALE_HOURS else "ok", f"NOW.md last written {age:.0f}h ago"))

    if not loops.loops_path(root).is_file():
        out.append(Check("loops", "warn", "LOOPS.md missing — `brain loop add` creates it"))
    else:
        errors, warnings = loops.lint(root, today)
        n_open = len(loops.load(root).open)
        if errors:
            out.append(Check("loops", "fail", f"{n_open} open · {len(errors)} lint error(s): " + "; ".join(errors[:3])))
        elif warnings:
            out.append(Check("loops", "warn", f"{n_open} open · " + "; ".join(warnings[:3])))
        else:
            out.append(Check("loops", "ok", f"{n_open} open · lint clean"))

    if not decisions.path(root).is_file():
        out.append(Check("decisions", "warn", "DECISIONS.md missing — `brain decide` creates it"))
    else:
        errs = decisions.lint(root)
        n = len(decisions.load(root)[0])
        out.append(Check("decisions", "fail" if errs else "ok",
                         f"{n} entries" + (" · " + "; ".join(errs[:3]) if errs else "")))

    from brain import now
    if now.is_generated(root):
        n_err, n_warn = now.lint(root, today)
        n_err = [e for e in n_err if "LOOPS" not in e]   # loop problems are reported above
        if n_err:
            out.append(Check("now.md", "fail", f"{len(n_err)} error(s): " + "; ".join(n_err[:3])))
        elif n_warn:
            out.append(Check("now.md", "warn", "; ".join(n_warn[:3])))
        else:
            out.append(Check("now.md", "ok", "generated, current; areas fresh"))
    else:
        out.append(Check("now.md", "warn", "hand-written — `brain now render` makes it generated from IDENTITY/LOOPS/areas/people/apps"))

    git_dir = root / ".git"
    if git_dir.is_dir():
        hook = git_dir / "hooks" / "pre-commit"
        if not (hook.is_file() and "DECISIONS.md is append-only" in hook.read_text(encoding="utf-8", errors="replace")):
            out.append(Check("vault-git", "warn", "append-only pre-commit hook not installed — `brain doctor --install-hooks`"))
        else:
            try:
                r = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=20)
                dirty = [l for l in r.stdout.splitlines() if l.strip()]
                out.append(Check("vault-git", "warn" if dirty else "ok",
                                 f"{len(dirty)} uncommitted change(s)" if dirty else "clean, hook installed"))
            except (subprocess.SubprocessError, OSError) as e:
                out.append(Check("vault-git", "warn", f"git status failed: {e}"))
    else:
        out.append(Check("vault-git", "warn", "vault is not a git repo"))
    return out


def _paths_in(text: str) -> list[str]:
    """Absolute paths mentioned in a hook/prompt that belong to the brain wiring."""
    found = []
    for p in ABS_PATH_RE.findall(text):
        p = str(Path(p.rstrip(".,;:/")).expanduser())
        if ("brain" in p or "python" in p) and p not in found:
            found.append(p)
    return found


def check_wiring(settings: Path | None = CLAUDE_SETTINGS, claude_json: Path | None = CLAUDE_JSON,
                 tasks_dir: Path | None = SCHEDULED_TASKS) -> list[Check]:
    out: list[Check] = []
    # hooks in ~/.claude/settings.json
    if settings is not None:
        if not settings.is_file():
            out.append(Check("hooks", "warn", f"{settings} missing"))
        else:
            try:
                cfg = json.loads(settings.read_text(encoding="utf-8"))
                cmds = [h.get("command", "") for ev in (cfg.get("hooks") or {}).values()
                        for grp in ev for h in grp.get("hooks", [])]
            except (json.JSONDecodeError, AttributeError) as e:
                cmds, cfg = [], {}
                out.append(Check("hooks", "fail", f"settings.json unreadable: {e}"))
            missing = [p for c in cmds for p in _paths_in(c) if not Path(p).exists()]
            events = sorted((cfg.get("hooks") or {}).keys()) if isinstance(cfg, dict) else []
            if missing:
                out.append(Check("hooks", "fail", "hook references missing path(s): " + ", ".join(missing)))
            elif not any("brain" in c for c in cmds):
                out.append(Check("hooks", "warn", "no hook mentions the brain"))
            else:
                out.append(Check("hooks", "ok", "brain paths resolve · events: " + ", ".join(events)))
    # MCP registration in ~/.claude.json
    if claude_json is not None:
        if not claude_json.is_file():
            out.append(Check("mcp", "warn", f"{claude_json} missing"))
        else:
            try:
                srv = (json.loads(claude_json.read_text(encoding="utf-8")).get("mcpServers") or {}).get("brain")
            except json.JSONDecodeError as e:
                srv = None
                out.append(Check("mcp", "fail", f"~/.claude.json unreadable: {e}"))
            if srv is None:
                out.append(Check("mcp", "warn", "no `brain` MCP server registered (claude mcp add --scope user brain -- <bin> mcp)"))
            else:
                cmd = srv.get("command", "")
                ok = Path(cmd).is_file() and os.access(cmd, os.X_OK)
                out.append(Check("mcp", "ok" if ok else "fail", f"brain → {cmd}" + ("" if ok else " (missing)")))
    # scheduled task prompts
    if tasks_dir is not None and tasks_dir.is_dir():
        bad = []
        for skill in sorted(tasks_dir.glob("*/SKILL.md")):
            for p in _paths_in(skill.read_text(encoding="utf-8", errors="replace")):
                if not Path(p).exists():
                    bad.append(f"{skill.parent.name}: {p}")
        out.append(Check("scheduled-tasks", "fail" if bad else "ok",
                         "; ".join(bad) if bad else f"{len(list(tasks_dir.glob('*/SKILL.md')))} prompt(s), paths resolve"))
    return out


def run(root: Path, today: date | None = None, now: float | None = None,
        db_path: Path = DB_PATH, expected_bin: Path = EXPECTED_BIN,
        settings: Path | None = CLAUDE_SETTINGS, claude_json: Path | None = CLAUDE_JSON,
        tasks_dir: Path | None = SCHEDULED_TASKS, api_probe=None,
        capture_log: Path | None = DATA_DIR / "capture.log",
        brief_log: Path | None = DATA_DIR / "brief.log") -> list[Check]:
    checks = [check_binary(expected_bin), check_db(db_path, now), check_graph_integrity(db_path),
              check_key(), check_api(api_probe)]
    if capture_log is not None:
        checks.append(check_capture(capture_log, now))
    if brief_log is not None:
        checks.append(check_brief(brief_log, now))
    checks += check_vault(root, today, now)
    checks.append(check_index(db_path, root))
    checks += check_wiring(settings, claude_json, tasks_dir)
    return checks


def worst(checks: list[Check]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    return "warn" if any(c.status == "warn" for c in checks) else "ok"


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks) if checks else 0
    return "\n".join(f"{c.icon} {c.name:<{width}}  {c.detail}" for c in checks)


def brief(checks: list[Check]) -> str:
    """One line for the top of `brain today` / a phone push."""
    fails = [c for c in checks if c.status == "fail"]
    if fails:
        return "✗ BRAIN BROKEN: " + " | ".join(f"{c.name}: {c.detail}" for c in fails)
    warns = [c for c in checks if c.status == "warn"]
    graph = next((c.detail for c in checks if c.name == "graph"), "")
    lp = next((c.detail for c in checks if c.name == "loops"), "")
    head = f"brain ✓ {graph} · loops: {lp}" if not warns else f"brain ⚠ {graph} · " + " | ".join(
        f"{c.name}: {c.detail}" for c in warns)
    return head
