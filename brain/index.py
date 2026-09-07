"""The vault index — the graph as a retrieval layer over the directory (D-014).

The vault (`~/.personal-brain/vault/`, a directory of markdown files with
`ALVIN.md` as the hub) is the brain. This module keeps a small index of it in
`brain.db` so a question can be routed to the right FILES fast, and stays
strictly secondary: it stores paths, titles, aliases, search tokens, a content
hash and an optional embedding per file, plus two kinds of links —
file → file (markdown references such as `people/heli.md`) and
file → graph node (a node whose name matches the file's title or an alias).
Matching nodes also get `nodes.path` stamped, so every graph hit points at the
file that is its source of truth. Nothing here ever writes a vault file, and the
index never holds facts of its own: answers read the files from disk.

    brain index            # (re)index the vault: incremental by content hash
    brain ask "..."        # ledgers → files → graph nodes, files cited by path
    index.search(...)      # ranked file paths for any query (used by ask/search)

Generated views (NOW.md, DIGEST.md, graph/) and the CLI-owned ledgers (LOOPS.md,
LOOPS-INBOX.md, DECISIONS.md — served by graph.ledger_context) are skipped.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from brain import db, llm
from brain.now import NowError, parse_frontmatter

SKIP_DIRS = {".git", "graph", "__pycache__"}
SKIP_ROOT_FILES = {"NOW.md", "DIGEST.md", "LOOPS.md", "LOOPS-INBOX.md", "DECISIONS.md", "README.md"}
SHELF_KIND = {
    "profile": "profile", "courses": "course", "applications": "application", "orgs": "org",
    "projects": "project", "people": "person", "apps": "app", "topics": "topic", "docs": "doc",
    "areas": "area", "log": "log",
}
ROOT_KIND = {"ALVIN.md": "hub", "IDENTITY.md": "identity"}
# lower = preferred when several files match the same node / tie in a search
KIND_PRIORITY = {"person": 0, "org": 1, "project": 2, "course": 3, "application": 4, "profile": 5,
                 "topic": 6, "app": 7, "area": 8, "hub": 9, "identity": 10, "doc": 11,
                 "readme": 12, "log": 13}
ALIAS_KEYS = ("person", "app", "area", "code", "project", "org")
LINK_RE = re.compile(r"(?<![\w/])((?:profile|courses|applications|orgs|projects|people|apps|topics|docs|areas|log)/[\w.\-]+\.md|ALVIN\.md|IDENTITY\.md)")
MAX_TOKENS = 3000
SEMANTIC_MIN = 0.4          # cosine below this adds nothing
SEMANTIC_ONLY_MIN = 0.5     # a file with no keyword hit needs at least this to appear
EMBED_CHARS = 1500

SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_files (
    path        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    summary     TEXT NOT NULL DEFAULT '',
    tokens      TEXT NOT NULL DEFAULT '',
    updated     TEXT NOT NULL DEFAULT '',
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    sha         TEXT NOT NULL,
    embedding   TEXT,
    indexed_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS vault_file_links (
    path    TEXT NOT NULL,
    target  TEXT NOT NULL,
    PRIMARY KEY (path, target)
);
CREATE TABLE IF NOT EXISTS vault_file_nodes (
    path     TEXT NOT NULL,
    node_id  TEXT NOT NULL,
    how      TEXT NOT NULL,
    PRIMARY KEY (path, node_id)
);
CREATE TABLE IF NOT EXISTS ledger_embeddings (
    key         TEXT PRIMARY KEY,   -- L-nnn / D-nnn
    sha         TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT,
    indexed_at  REAL NOT NULL
);
"""


def ensure_schema(conn):
    conn.executescript(SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}
    if "path" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN path TEXT")
    conn.commit()


# ── walking and parsing ───────────────────────────────────────────────────────

def vault_files(root: Path) -> list[Path]:
    """Every indexable markdown file under root, sorted, vault-relative rules applied."""
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if len(rel.parts) == 1 and rel.name in SKIP_ROOT_FILES:
            continue
        out.append(p)
    return out


def kind_of(rel: str, fm: dict) -> str:
    t = str(fm.get("type", "")).strip().lower()
    parts = rel.split("/")
    if len(parts) == 1:
        return ROOT_KIND.get(parts[0], t or "note")
    if parts[-1] == "README.md":
        return "readme"
    # the shelf decides: an app file typed `artifact` (its node type) is still an
    # "app" for ranking and linking — as "artifact" it had no priority at all, so
    # an area whose aliases mention the app won the node's path
    return SHELF_KIND.get(parts[0]) or t or "note"


def tokens_of(text: str) -> set[str]:
    return {t for t in db._TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in db._STOPWORDS}


def _norm(name: str) -> str:
    return " ".join(db._TOKEN_RE.findall(name.lower()))


def _title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _summary(body: str, limit: int = 400) -> str:
    lines = [l.strip() for l in body.splitlines() if l.strip() and not l.startswith("#")]
    return " ".join(lines)[:limit]


def parse_file(root: Path, path: Path) -> dict:
    """One file → its index record (front-matter tolerant: a bad header is not fatal)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        fm, body = parse_frontmatter(text)
    except NowError:
        fm, body = {}, text
    rel = path.relative_to(root).as_posix()
    title = str(fm.get("name") or "").strip() or _title(body, path.stem.replace("-", " "))
    aliases = list(fm.get("aliases") or []) if isinstance(fm.get("aliases"), list) else (
        [fm["aliases"]] if fm.get("aliases") else [])
    for key in ALIAS_KEYS:
        v = fm.get(key)
        if isinstance(v, str) and v.strip() and key != "project" or (key == "project" and isinstance(v, str) and v.strip()):
            aliases.append(v.strip())
    aliases = [a for a in dict.fromkeys(a for a in aliases if a and a.lower() != title.lower())]
    toks = tokens_of(f"{title} {' '.join(aliases)} {body}")
    links = [t for t in dict.fromkeys(LINK_RE.findall(body)) if t != rel]
    st = path.stat()
    return {
        "path": rel, "kind": kind_of(rel, fm), "title": title, "aliases": aliases,
        "summary": _summary(body), "tokens": " ".join(sorted(toks)[:MAX_TOKENS]),
        "updated": str(fm.get("updated", "")), "mtime": st.st_mtime, "size": st.st_size,
        "sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "links": links, "embed_text": f"{title}. {' '.join(aliases)}. {body[:EMBED_CHARS]}",
        "has_frontmatter": bool(fm),
    }


# ── building the index ────────────────────────────────────────────────────────

_SCHOOL_PREFIX = re.compile(r"^(?:mit|harvard|aalto|stanford)\s+")
_QUALIFIER = re.compile(r"\s*\([^()]*\)\s*$")   # "Walkthrough (Junction 2025)" → "Walkthrough"
# a who-question wants people: boost person files that match at all, and keep
# a few of them in the result even when bigger files out-score them
_WHO_RE = re.compile(r"\b(?:who|whom|whose|people|persons?|friends?|colleagues?|classmates?|contacts?)\b")
WHO_SLOTS = 3
# question filler that would otherwise count as body hits in every large file
_QUERY_STOP = {
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "been", "was", "were", "will", "would", "can", "could", "should", "shall",
    "there", "their", "they", "them", "he", "she", "his", "her", "him", "we", "our", "you", "your",
    "at", "as", "by", "from", "into", "than", "then", "up", "out", "all", "any", "some", "so", "far",
    "too", "very", "just", "also", "now", "still", "yet", "ever", "never", "each", "every", "much",
    "many", "more", "most", "such", "not", "no", "only", "same", "other", "own", "here",
}


def _node_name_map(conn) -> dict[str, list]:
    """normalized node name → [node rows] (active nodes only). A node named
    with a school in front ("MIT 9.522", "Harvard AM 207") is also keyed by
    the bare code, so a file whose alias is "9.522" still links to it; one
    with a trailing qualifier ("Walkthrough (Junction 2025)") is also keyed
    by the bare name, so a file aliased "Walkthrough" links to it."""
    out: dict[str, list] = {}
    for n in db.all_nodes(conn):
        key = _norm(n["name"])
        out.setdefault(key, []).append(n)
        for bare in (_SCHOOL_PREFIX.sub("", key), _norm(_QUALIFIER.sub("", n["name"]))):
            if bare != key and len(bare) >= 3 and n not in out.setdefault(bare, []):
                out[bare].append(n)
    return out


def link_nodes(conn, records: list[dict]) -> int:
    """Recompute file → node links from titles and aliases; stamp nodes.path with
    the best (highest-priority) file. Returns the number of links."""
    names = _node_name_map(conn)
    conn.execute("DELETE FROM vault_file_nodes")
    best: dict[str, tuple[int, str]] = {}   # node_id → (priority, path)
    count = 0
    for rec in sorted(records, key=lambda r: (KIND_PRIORITY.get(r["kind"], 20), r["path"])):
        cands = [("name", rec["title"])] + [("alias", a) for a in rec["aliases"]]
        seen: set[str] = set()
        for how, cand in cands:
            key = _norm(cand)
            if not key or len(key) < 3:
                continue
            for n in names.get(key, []):
                if n["id"] in seen:
                    continue
                seen.add(n["id"])
                conn.execute("INSERT OR IGNORE INTO vault_file_nodes (path, node_id, how) VALUES (?,?,?)",
                             (rec["path"], n["id"], how))
                count += 1
                prio = KIND_PRIORITY.get(rec["kind"], 20)
                if n["id"] not in best or prio < best[n["id"]][0]:
                    best[n["id"]] = (prio, rec["path"])
    conn.execute("UPDATE nodes SET path = NULL")
    for nid, (_, path) in best.items():
        conn.execute("UPDATE nodes SET path = ? WHERE id = ?", (path, nid))
    return count


def build(conn, root: Path, embed: bool = True) -> dict:
    """Index the vault: add new files, refresh changed ones (by content hash),
    drop deleted ones, relink nodes, embed what changed (best-effort, only with
    a key). Returns counts."""
    root = Path(root)
    ensure_schema(conn)
    existing = {r["path"]: r for r in conn.execute("SELECT path, sha, embedding FROM vault_files")}
    now = time.time()
    stats = {"files": 0, "added": 0, "updated": 0, "removed": 0, "unchanged": 0,
             "links": 0, "node_links": 0, "embedded": 0, "no_frontmatter": []}
    records = []
    seen: set[str] = set()
    to_embed = []
    for p in vault_files(root):
        rec = parse_file(root, p)
        records.append(rec)
        seen.add(rec["path"])
        if not rec["has_frontmatter"] and rec["kind"] not in ("log", "readme", "identity"):
            stats["no_frontmatter"].append(rec["path"])
        old = existing.get(rec["path"])
        if old and old["sha"] == rec["sha"]:
            stats["unchanged"] += 1
            if embed and not old["embedding"]:
                to_embed.append(rec)
            continue
        conn.execute(
            "INSERT INTO vault_files (path, kind, title, aliases, summary, tokens, updated, mtime, size, sha, embedding, indexed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?)"
            " ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, title=excluded.title, aliases=excluded.aliases,"
            " summary=excluded.summary, tokens=excluded.tokens, updated=excluded.updated, mtime=excluded.mtime,"
            " size=excluded.size, sha=excluded.sha, embedding=NULL, indexed_at=excluded.indexed_at",
            (rec["path"], rec["kind"], rec["title"], json.dumps(rec["aliases"], ensure_ascii=False),
             rec["summary"], rec["tokens"], rec["updated"], rec["mtime"], rec["size"], rec["sha"], now))
        stats["added" if old is None else "updated"] += 1
        if embed:
            to_embed.append(rec)
    for path in set(existing) - seen:
        conn.execute("DELETE FROM vault_files WHERE path = ?", (path,))
        stats["removed"] += 1
    stats["files"] = len(records)

    conn.execute("DELETE FROM vault_file_links")
    known = {r["path"] for r in records}
    for rec in records:
        for target in rec["links"]:
            if target in known:
                conn.execute("INSERT OR IGNORE INTO vault_file_links (path, target) VALUES (?,?)",
                             (rec["path"], target))
                stats["links"] += 1
    stats["node_links"] = link_nodes(conn, records)
    conn.commit()

    if embed and to_embed and llm.have_key():
        for rec in to_embed:
            try:
                vec = llm.embed(rec["embed_text"])
            except Exception:
                continue
            conn.execute("UPDATE vault_files SET embedding = ? WHERE path = ?",
                         (json.dumps(vec), rec["path"]))
            stats["embedded"] += 1
        conn.commit()
    stats["ledger_embedded"] = embed_ledgers(conn, root, embed=embed)
    return stats


def status(conn, root: Path) -> dict:
    """How current the index is: files on disk vs indexed, and which changed."""
    root = Path(root)
    ensure_schema(conn)
    rows = {r["path"]: r for r in conn.execute(
        "SELECT path, sha, mtime, size, embedding FROM vault_files")}
    on_disk = vault_files(root)
    stale, new = [], []
    for p in on_disk:
        rel = p.relative_to(root).as_posix()
        r = rows.get(rel)
        if r is None:
            new.append(rel)
            continue
        st = p.stat()
        if st.st_mtime == r["mtime"] and st.st_size == r["size"]:
            continue
        sha = hashlib.sha256(p.read_text(encoding="utf-8", errors="replace").encode("utf-8")).hexdigest()[:16]
        if sha != r["sha"]:
            stale.append(rel)
    disk_rel = {p.relative_to(root).as_posix() for p in on_disk}
    removed = sorted(set(rows) - disk_rel)
    last = conn.execute("SELECT MAX(indexed_at) FROM vault_files").fetchone()[0]
    node_links = conn.execute("SELECT COUNT(*) FROM vault_file_nodes").fetchone()[0]
    embedded = sum(1 for r in rows.values() if r["embedding"])
    # the ledgers: a loop edited since the last index is matched by its old text
    stored = {r["key"]: r for r in conn.execute("SELECT key, sha, embedding FROM ledger_embeddings")}
    records = ledger_records(root)
    ledger_stale = sorted([r["key"] for r in records if stored.get(r["key"]) is None
                           or stored[r["key"]]["sha"] != r["sha"]]
                          + [k for k in stored if k not in {r["key"] for r in records}])
    ledger_embedded = sum(1 for r in stored.values() if r["embedding"])
    return {"indexed": len(rows), "on_disk": len(on_disk), "new": new, "stale": stale,
            "removed": removed, "node_links": node_links, "embedded": embedded, "last_indexed": last,
            "ledger_total": len(records), "ledger_stale": ledger_stale, "ledger_embedded": ledger_embedded}


# ── the ledgers as retrieval targets ──────────────────────────────────────────
# Loops and decisions are matched to a question by keyword overlap, which misses
# a loop worded differently from the question ("what am I owed?" never reached
# "collect the remaining 9 repayments"). Embed each loop and decision once
# (incremental by content hash, like files) so ledger_context can add cosine hits.

def ledger_records(root: Path) -> list[dict]:
    """One record per loop (open and closed) and per decision: {key, text, sha}."""
    from brain import decisions, loops
    out: list[dict] = []
    try:
        ledger = loops.load(Path(root))
        for l in ledger.open + ledger.closed:
            out.append({"key": l.id, "text": f"{l.id} {l.title}. Next: {l.next}. {l.note}".strip()})
    except Exception:
        pass
    try:
        for d in decisions.load(Path(root))[0]:
            out.append({"key": d.id, "text": f"{d.id} {d.title}. {d.decision} Why: {d.why}".strip()})
    except Exception:
        pass
    for r in out:
        r["sha"] = hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:16]
    return out


def embed_ledgers(conn, root: Path, embed: bool = True) -> int:
    """Upsert the ledger records and embed the new or changed ones (parallel,
    best-effort, only with a key). Returns the number embedded."""
    from concurrent.futures import ThreadPoolExecutor
    ensure_schema(conn)
    records = ledger_records(root)
    existing = {r["key"]: r for r in conn.execute("SELECT key, sha, embedding FROM ledger_embeddings")}
    now = time.time()
    todo = []
    for rec in records:
        old = existing.get(rec["key"])
        if old and old["sha"] == rec["sha"]:
            if embed and not old["embedding"]:
                todo.append(rec)
            continue
        conn.execute("INSERT INTO ledger_embeddings (key, sha, text, embedding, indexed_at) VALUES (?,?,?,NULL,?)"
                     " ON CONFLICT(key) DO UPDATE SET sha=excluded.sha, text=excluded.text, embedding=NULL,"
                     " indexed_at=excluded.indexed_at", (rec["key"], rec["sha"], rec["text"], now))
        if embed:
            todo.append(rec)
    for key in set(existing) - {r["key"] for r in records}:
        conn.execute("DELETE FROM ledger_embeddings WHERE key = ?", (key,))
    conn.commit()
    done = 0
    if todo and llm.have_key():
        def fetch(rec):
            try:
                return rec["key"], llm.embed(rec["text"])
            except Exception:
                return rec["key"], None
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            for key, vec in ex.map(fetch, todo):
                if vec:
                    conn.execute("UPDATE ledger_embeddings SET embedding = ? WHERE key = ?", (json.dumps(vec), key))
                    done += 1
        conn.commit()
    return done


def ledger_semantic(conn, query_vector, limit: int = 6, min_cos: float = SEMANTIC_ONLY_MIN) -> list[tuple[str, float]]:
    """Ledger keys closest to the query vector, [(key, cosine)] best first."""
    ensure_schema(conn)
    scored = []
    for r in conn.execute("SELECT key, embedding FROM ledger_embeddings WHERE embedding IS NOT NULL"):
        try:
            cos = _cosine(query_vector, json.loads(r["embedding"]))
        except (TypeError, ValueError):
            continue
        if cos >= min_cos:
            scored.append((r["key"], round(cos, 3)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:limit]


# ── retrieval ─────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def search(conn, query: str, k: int = 6, seed_node_ids: list[str] | None = None,
           query_vector: list | None = None) -> list[dict]:
    """Rank vault files for a query. Signals, all deterministic:
    keyword hits (title/alias hits count triple, exact phrase bonus), graph hop
    (files linked to the given seed nodes), embeddings (cosine, when a query
    vector is supplied), and one hop of file links from the top hits.
    Returns [{path, title, kind, score, why}] best first."""
    ensure_schema(conn)
    q = query.lower().strip()
    q_tokens = tokens_of(q) or ({q} if q else set())
    q_tokens = (q_tokens - _QUERY_STOP) or q_tokens  # never strip a query down to nothing
    rows = conn.execute("SELECT path, kind, title, aliases, summary, tokens, updated, embedding FROM vault_files").fetchall()
    if not rows:
        return []
    seeded: dict[str, list[str]] = {}
    if seed_node_ids:
        marks = ",".join("?" * len(seed_node_ids))
        for r in conn.execute(f"SELECT f.path, n.name FROM vault_file_nodes f JOIN nodes n ON n.id = f.node_id"
                              f" WHERE f.node_id IN ({marks})", list(seed_node_ids)):
            seeded.setdefault(r["path"], []).append(r["name"])

    who = bool(_WHO_RE.search(q))
    hits: dict[str, dict] = {}
    for r in rows:
        aliases = json.loads(r["aliases"] or "[]")
        head = f"{r['title']} {' '.join(aliases)}".lower()
        if who and r["kind"] == "person":
            # a person file's first lines are the role ("CS 2881R classmate at
            # Harvard SEAS"): for a who-question that is head text, not body text
            head += " " + (r["summary"] or "").lower()
        head_tokens = tokens_of(head)
        body_tokens = set((r["tokens"] or "").split())
        t_hits = sum(1 for t in q_tokens if any(db._stem_eq(t, h) for h in head_tokens))
        b_hits = sum(1 for t in q_tokens if any(db._stem_eq(t, h) for h in body_tokens))
        score = 3 * t_hits + b_hits
        why = []
        if t_hits:
            why.append("title match" if t_hits == len(q_tokens) else f"title match {t_hits}/{len(q_tokens)}")
        elif b_hits:
            why.append(f"body match {b_hits}/{len(q_tokens)}")
        if who and r["kind"] == "person" and (t_hits or b_hits):
            score += 3  # worth a title match: a who-question is answered from people files
            why.append("who-question: person file")
        if q and q in head:
            score += 2 * len(q_tokens)
            why.append("exact phrase")
        if r["path"] in seeded:
            score += 3
            why.append("node: " + ", ".join(seeded[r["path"]][:2]))
        if query_vector is not None and r["embedding"]:
            try:
                cos = _cosine(query_vector, json.loads(r["embedding"]))
            except (TypeError, ValueError):
                cos = 0.0
            if cos >= SEMANTIC_MIN and (score > 0 or cos >= SEMANTIC_ONLY_MIN):
                score += 5 * cos
                why.append(f"semantic {cos:.2f}")
        if score > 0:
            hits[r["path"]] = {"path": r["path"], "title": r["title"], "kind": r["kind"],
                               "score": round(score, 3), "why": why, "updated": r["updated"] or ""}

    def order(h):
        return (-h["score"], KIND_PRIORITY.get(h["kind"], 20), h["updated"] and -int(h["updated"].replace("-", "")[:8] or 0), h["path"])

    top = sorted(hits.values(), key=order)
    # one hop of file links from the best hits: related files the reader would follow
    meta = {r["path"]: r for r in rows}
    for h in top[:3]:
        for lr in conn.execute("SELECT target FROM vault_file_links WHERE path = ?", (h["path"],)):
            t = lr["target"]
            if t in hits:
                if h["path"] != t:
                    hits[t]["score"] = round(hits[t]["score"] + 1, 3)
                    hits[t]["why"].append(f"linked from {h['path']}")
            elif t in meta:
                m = meta[t]
                hits[t] = {"path": t, "title": m["title"], "kind": m["kind"], "score": 1.0,
                           "why": [f"linked from {h['path']}"], "updated": m["updated"] or ""}
    out = sorted(hits.values(), key=order)[:k]
    if who:  # reserve a few slots for the best-matching people files
        have = {h["path"] for h in out}
        extra = [h for h in sorted(hits.values(), key=order)
                 if h["kind"] == "person" and h["path"] not in have][:WHO_SLOTS]
        if extra:
            keep = [h for h in out if h["kind"] == "person"] + \
                   [h for h in out if h["kind"] != "person"][:max(0, k - len(extra) - sum(1 for h in out if h["kind"] == "person"))]
            out = sorted(keep + extra, key=order)[:k]
    for h in out:
        h.pop("updated", None)
    return out


def excerpt(root: Path, rel_path: str, query: str = "", max_chars: int = 1800,
            matches_first: bool = False) -> str:
    """The part of a file worth showing an answerer: its opening, then the lines
    that mention query tokens. Read from disk — the vault is the source of truth.
    `matches_first` (used for log files, where the opening is the oldest entry)
    leads with the matching lines, newest first, and adds the opening only if
    room remains."""
    p = Path(root) / rel_path
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    try:
        _, body = parse_frontmatter(text)
    except NowError:
        body = text
    body = body.strip()
    q_tokens = tokens_of(query)
    if not q_tokens or len(body) <= max_chars:
        return _cut(body, max_chars)

    def matching(lines):
        for line in lines:
            lt = tokens_of(line)
            if lt and any(any(db._stem_eq(t, h) for h in lt) for t in q_tokens):
                yield line

    if matches_first:
        picked, used = [], 0
        for line in matching(reversed(body.splitlines())):
            room = max_chars - used - 1
            if room < 80:
                break
            if len(line) > room:
                line = _window(line, q_tokens, room)
            picked.append(line)
            used += len(line) + 1
        rest = max_chars - used - 2
        head = _cut(body, rest) if rest >= 120 else ""
        return "\n".join(picked) + (("\n…\n" + head) if head else "") if picked else _cut(body, max_chars)

    head = _cut(body, max_chars // 2)   # the body is longer than the limit here, so the cut is marked
    picked, used = [], len(head)
    for line in matching(body[max_chars // 2:].splitlines()):
        room = max_chars - used - 1
        if room < 80:
            break
        if len(line) > room:
            # a long matching line (a day's log entry) used to be dropped whole:
            # show a window around its first hit instead
            line = _window(line, q_tokens, room)
        picked.append(line)
        used += len(line) + 1
    return head + ("\n…\n" + "\n".join(picked) if picked else "")


def _cut(text: str, limit: int) -> str:
    """Trim to `limit` at the last line or sentence boundary (never mid-word,
    never mid-quote if avoidable) and mark the cut, so an answerer does not
    mistake a truncated clause for a whole claim."""
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 2, 1)]  # room for the marker, so the result never exceeds `limit`
    for sep in ("\n", ". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        if i >= int(limit * 0.6):
            cut = cut[: i + (1 if sep == ". " else 0)]
            break
    return cut.rstrip() + " …"


def _window(line: str, q_tokens: set, room: int) -> str:
    """The slice of `line` (≤ room chars) around the first query hit, marked."""
    low = line.lower()
    pos = min((low.find(t) for t in q_tokens if low.find(t) >= 0), default=0)
    start = max(0, pos - room // 3)
    end = min(len(line), start + room - 4)
    start = max(0, end - (room - 4))
    return ("…" if start > 0 else "") + line[start:end].strip() + ("…" if end < len(line) else "")


def nodes_for_path(conn, rel_path: str) -> list:
    return [db.get_node(conn, r["node_id"]) for r in
            conn.execute("SELECT node_id FROM vault_file_nodes WHERE path = ?", (rel_path,))]
