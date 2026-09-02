"""DECISIONS.md — the append-only decision ledger.

Every settled decision gets one entry: what was decided, why, what was
rejected, and what would reopen it. Entries are never edited — a change of
mind is a NEW entry that names the one it supersedes. This is what stops
sessions from re-litigating a choice (two course-planning threads diverged in
Aug 2026 exactly because nothing recorded what had already been settled).

    ## D-003 · 2026-09-01 · Fourth-seat plan of record
    - **Decision:** ...
    - **Why:** ...
    - **Rejected:** ...
    - **Revisit if:** ...
    - **Source:** ...

Append-only is enforced twice: `brain decisions --lint` checks ids are
sequential and entries complete, and the vault's git pre-commit hook rejects
any commit that removes a line from this file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DECISIONS_FILE = "DECISIONS.md"
HEAD_RE = re.compile(r"^## (D-\d{3,}) · (\d{4}-\d{2}-\d{2}) · (.+)$")
FIELDS = ("Decision", "Why", "Rejected", "Revisit if", "Source")
FIELD_RE = re.compile(r"^- \*\*(Decision|Why|Rejected|Revisit if|Source):\*\* (.*)$")

HEADER = """# DECISIONS — append-only ledger
<!-- Managed by `brain decide`. NEVER edit or delete an entry: to change your mind, add a new
     entry whose Decision names the one it supersedes ("supersedes D-004"). Sessions cite ids
     (D-003) instead of re-arguing settled questions. The vault pre-commit hook rejects removals. -->
"""


class DecisionError(ValueError):
    pass


@dataclass
class Decision:
    id: str
    date: date
    title: str
    decision: str
    why: str
    rejected: str = "—"
    revisit: str = "—"
    source: str = "—"

    def to_md(self) -> str:
        return "\n".join([
            f"## {self.id} · {self.date.isoformat()} · {self.title}",
            f"- **Decision:** {self.decision}",
            f"- **Why:** {self.why}",
            f"- **Rejected:** {self.rejected}",
            f"- **Revisit if:** {self.revisit}",
            f"- **Source:** {self.source}",
        ]) + "\n"


def path(root: Path) -> Path:
    return Path(root) / DECISIONS_FILE


def parse(text: str) -> tuple[list[Decision], list[str]]:
    """(decisions, errors). Tolerant of prose between entries; strict inside them."""
    decisions: list[Decision] = []
    errors: list[str] = []
    cur: dict | None = None

    def flush():
        if cur is None:
            return
        missing = [f for f in ("Decision", "Why") if not cur.get(f)]
        if missing:
            errors.append(f"{cur['id']}: missing {', '.join(missing)}")
        decisions.append(Decision(
            id=cur["id"], date=cur["date"], title=cur["title"],
            decision=cur.get("Decision", ""), why=cur.get("Why", ""),
            rejected=cur.get("Rejected", "—"), revisit=cur.get("Revisit if", "—"),
            source=cur.get("Source", "—")))

    for no, line in enumerate(text.splitlines(), 1):
        m = HEAD_RE.match(line)
        if m:
            flush()
            try:
                d = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"line {no}: bad date {m.group(2)}")
                d = date.min
            cur = {"id": m.group(1), "date": d, "title": m.group(3).strip()}
            continue
        if line.startswith("## "):
            flush()
            cur = None
            errors.append(f"line {no}: malformed entry heading {line[:60]!r}")
            continue
        f = FIELD_RE.match(line)
        if f and cur is not None:
            cur[f.group(1)] = f.group(2).strip()
    flush()
    return decisions, errors


def load(root: Path) -> tuple[list[Decision], list[str]]:
    p = path(root)
    return parse(p.read_text(encoding="utf-8")) if p.is_file() else ([], [])


def next_id(decisions: list[Decision]) -> str:
    n = max((int(d.id.split("-")[1]) for d in decisions), default=0)
    return f"D-{n + 1:03d}"


def append(root: Path, title: str, decision: str, why: str, rejected: str = "—",
           revisit: str = "—", source: str = "—", when: date | None = None,
           commit: bool = True) -> Decision:
    """Append one entry. Never rewrites existing text — the file is only ever extended."""
    from brain import loops  # git helper lives there; avoid a circular import at module load
    when = when or date.today()
    existing, errors = load(root)
    if errors:
        raise DecisionError("DECISIONS.md has errors — run `brain decisions --lint`:\n  "
                            + "\n  ".join(errors))
    for name, text in (("title", title), ("decision", decision), ("why", why)):
        if not text.strip():
            raise DecisionError(f"{name} is required")
        if "\n" in text:
            raise DecisionError(f"{name} must be a single line")
    d = Decision(id=next_id(existing), date=when, title=title.strip(),
                 decision=decision.strip(), why=why.strip(),
                 rejected=(rejected or "—").strip(), revisit=(revisit or "—").strip(),
                 source=(source or "—").strip())
    p = path(root)
    if not p.is_file():
        p.write_text(HEADER + "\n", encoding="utf-8")
    body = p.read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"
    p.write_text(body + "\n" + d.to_md(), encoding="utf-8")
    if commit:
        loops.git_commit(root, f"decision {d.id}: {d.title}")
    return d


def lint(root: Path) -> list[str]:
    decisions, errors = load(root)
    if not path(root).is_file():
        return [f"{DECISIONS_FILE} missing"]
    nums = [int(d.id.split("-")[1]) for d in decisions]
    if nums != list(range(1, len(nums) + 1)):
        errors.append(f"ids are not sequential from D-001: {[d.id for d in decisions]}")
    for a, b in zip(decisions, decisions[1:]):
        if b.date < a.date:
            errors.append(f"{b.id} ({b.date}) is dated before {a.id} ({a.date}) — entries must be appended in order")
    return errors


def recent(root: Path, today: date | None = None, days: int = 7) -> list[str]:
    today = today or date.today()
    decisions, _ = load(root)
    return [f"{d.id} ({d.date.strftime('%b %d')}): {d.title}"
            for d in decisions if (today - d.date).days <= days]


PRE_COMMIT_HOOK = """#!/bin/sh
# Vault pre-commit hook (installed by `brain doctor --install-hooks`):
# DECISIONS.md is append-only — refuse any commit that removes or rewrites a line in it.
if git diff --cached -U0 -- DECISIONS.md | grep -E '^-' | grep -vqE '^--- '; then
  echo "pre-commit: DECISIONS.md is append-only — add a superseding entry instead of editing." >&2
  exit 1
fi
exit 0
"""


def install_pre_commit(root: Path) -> bool:
    hooks = Path(root) / ".git" / "hooks"
    if not hooks.is_dir():
        return False
    target = hooks / "pre-commit"
    if target.is_file() and "DECISIONS.md is append-only" in target.read_text(encoding="utf-8"):
        return True
    target.write_text(PRE_COMMIT_HOOK, encoding="utf-8")
    target.chmod(0o755)
    return True


_WORD = re.compile(r"[a-z0-9][a-z0-9.+-]{2,}")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


_STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "what", "when",
         "where", "which", "does", "did", "have", "has", "are", "was", "were", "not", "you", "your",
         "alvin", "alvins", "his", "her", "how", "why", "who", "will", "should", "could", "would"}


def search(root: Path, query: str, limit: int = 5) -> list[Decision]:
    """Decisions whose title/decision/why share keywords with the query, best first."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for d in load(root)[0]:
        hay = _tokens(" ".join([d.title, d.decision, d.why, d.rejected, d.revisit]))
        hit = len(q & hay)
        if hit:
            scored.append((hit, d))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [d for _, d in scored[:limit]]
