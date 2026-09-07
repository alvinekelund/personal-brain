import math
import re
import time
from datetime import date as _date

EDGE_BASE_HALF_LIFE = 90.0   # days for a once-seen edge
EDGE_MIN_WEIGHT = 0.05       # below this, delete the edge — except the part_of spine, which is clamped
STRUCTURAL_RELATIONS = ("part_of",)   # hierarchy edges: may fade, never vanish
ARCHIVE_THRESHOLD = 0.10     # weight below this → archived
IMPORTANCE_GAIN = 4.0        # important nodes decay up to (1+gain)x slower
IMPORTANCE_FLOOR = 0.15      # weight never drops below importance * this
# People and organizations never decay - but only while they matter. Below this
# importance a one-off sponsor or a passing acquaintance is remembered like a
# concept (DEMOTED_HALF_LIFE, ~14 months untouched at importance 0.3) instead of
# forever. Categories are the spine and stay immortal whatever their importance.
IMMORTAL_MIN_IMPORTANCE = 0.4
DEMOTED_HALF_LIFE = 60.0


_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_DATE_RE = re.compile(r"\b([A-Za-z]{3,9})\.? (\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?(?:st|nd|rd|th)?(?:,? ((?:19|20)\d\d))?\b")


def upcoming(content: str, today=None) -> bool:
    """Does the content name a date that is still ahead? HackMIT (Sep 19-20)
    and the Glasswing hackathon (Sep 26-27) were fading at weight 0.89 on
    Sep 6 2026, weeks before they happened: an event decays from its creation,
    not from its date. A missing year means this year; a range counts its end."""
    today = today or _date.today()
    for m in _DATE_RE.finditer(content or ""):
        month = _MONTHS.get(m.group(1).lower())
        if not month:
            continue
        day = int(m.group(3) or m.group(2))
        year = int(m.group(4)) if m.group(4) else today.year
        try:
            when = _date(year, month, day)
        except ValueError:
            continue
        if when >= today:
            return True
    return False


def node_half_life(half_life_days: float, node_type: str, importance: float,
                   content: str | None = None, today=None) -> float:
    """The half-life decay actually uses for a node: the stored one, except that
    an immortal (inf) person/organization below IMMORTAL_MIN_IMPORTANCE gets
    DEMOTED_HALF_LIFE. Categories are never demoted."""
    if (math.isinf(half_life_days) and node_type != "category"
            and (importance or 0.0) < IMMORTAL_MIN_IMPORTANCE):
        return DEMOTED_HALF_LIFE
    if node_type == "event" and content and upcoming(content, today):
        return float("inf")      # not before it has happened
    return half_life_days


def days_until_archive(weight: float, half_life_days: float,
                       threshold: float = ARCHIVE_THRESHOLD, importance: float = 0.0) -> float:
    """Days until an untouched node decays to the archive threshold.

    Importance-aware: it stretches the effective half-life and floors the weight
    exactly like current_weight, so a floored (important) node returns inf —
    matching what decay will actually do. inf for never-decaying types.
    """
    if math.isinf(half_life_days):
        return float("inf")
    if importance * IMPORTANCE_FLOOR >= threshold:
        return float("inf")  # floored above threshold → never archives
    if weight <= threshold:
        return 0.0
    effective_hl = half_life_days * (1 + IMPORTANCE_GAIN * importance)
    return effective_hl * math.log2(weight / threshold)


def at_risk_nodes(conn, limit: int = 5, threshold: float = ARCHIVE_THRESHOLD) -> list:
    """Active, decaying nodes closest to being archived (soonest first).

    Returns dicts with name/type/weight and an importance-aware days_left, so
    `brain status` shows what's about to be forgotten. Never-decaying types and
    importance-floored nodes (which won't archive) are excluded.
    """
    rows = conn.execute(
        "SELECT * FROM nodes WHERE archived=0 AND weight >= ?", (threshold,),
    ).fetchall()
    cands = []
    for r in rows:
        imp = r["importance"] if "importance" in r.keys() else 0.0
        hl = node_half_life(r["half_life_days"], r["type"], imp, r["content"] if "content" in r.keys() else None)
        if math.isinf(hl):
            continue  # immortal: person/org that matters, or a category
        days = days_until_archive(r["weight"], hl, threshold, imp)
        if math.isinf(days):
            continue  # floored/important → not at risk
        cands.append({"name": r["name"], "type": r["type"], "weight": r["weight"],
                      "days_left": days})
    cands.sort(key=lambda x: x["days_left"])
    return cands[:limit]


def current_weight(weight: float, last_accessed: float, half_life_days: float,
                   importance: float = 0.0) -> float:
    """Importance-weighted half-life decay: w(t) = w0 * 0.5**(t / H_eff), clamped
    to an importance floor.

    Base-1/2 keeps half_life_days a *true* half-life. importance (0-1) stretches
    the effective half-life (important nodes fade much slower) and sets a weight
    floor, so a central fact never decays into the archive while a one-off detail
    still does. importance=0 reproduces plain half-life decay.
    """
    if math.isinf(half_life_days):
        return weight
    effective_hl = half_life_days * (1 + IMPORTANCE_GAIN * importance)
    days_elapsed = (time.time() - last_accessed) / 86400.0
    decayed = weight * 0.5 ** (days_elapsed / effective_hl)
    return max(decayed, importance * IMPORTANCE_FLOOR)


def edge_half_life(reinforcement_count: int) -> float:
    """
    Hebbian scaling: the more co-occurrences, the slower an edge decays.
    half_life = base * ln(1 + count)
    count=1 → ~62 days, count=5 → ~161 days, count=20 → ~271 days
    """
    return EDGE_BASE_HALF_LIFE * math.log1p(reinforcement_count)


def run_decay(conn, now: float | None = None) -> dict:
    """
    Update weights for all non-archived nodes and all edges.
    Archive nodes below 0.10, delete nodes archived 7+ days.
    Delete edges below EDGE_MIN_WEIGHT.
    Returns counts of updated / archived / deleted nodes + edges pruned.
    """
    nodes = conn.execute(
        "SELECT id, type, weight, last_accessed, last_decayed, half_life_days, archived, importance, content FROM nodes"
    ).fetchall()

    updated = archived = deleted = edges_pruned = 0
    now = now or time.time()
    today = _date.fromtimestamp(now)

    for n in nodes:
        if n["archived"]:
            days_archived = (now - n["last_accessed"]) / 86400.0
            if days_archived > 7:
                # the schema has no ON DELETE CASCADE: take the edges too, or
                # they dangle (three such ghosts were found on Sep 6 2026)
                conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (n["id"], n["id"]))
                conn.execute("DELETE FROM nodes WHERE id = ?", (n["id"],))
                deleted += 1
            continue

        # The stored weight is current as of `last_decayed` (or the later access
        # that reset it), so decay only the interval since then. Calling decay
        # twice in a row is then a no-op instead of a second full decay.
        since = max(n["last_decayed"], n["last_accessed"])
        hl = node_half_life(n["half_life_days"], n["type"], n["importance"], n["content"], today)
        new_w = current_weight(n["weight"], since, hl, n["importance"])
        new_w = max(0.0, min(1.0, new_w))

        if new_w < 0.10:
            conn.execute(
                "UPDATE nodes SET weight = ?, archived = 1, last_decayed = ? WHERE id = ?",
                (new_w, now, n["id"]),
            )
            archived += 1
        else:
            conn.execute(
                "UPDATE nodes SET weight = ?, last_decayed = ? WHERE id = ?",
                (new_w, now, n["id"]),
            )
            updated += 1

    # edge decay — same clock discipline; the part_of spine is clamped, never deleted
    edges = conn.execute(
        "SELECT id, weight, last_reinforced, last_decayed, reinforcement_count, relation FROM edges"
    ).fetchall()

    for e in edges:
        hl = edge_half_life(e["reinforcement_count"])
        since = max(e["last_decayed"], e["last_reinforced"])
        days_elapsed = max(0.0, (now - since) / 86400.0)
        new_w = e["weight"] * 0.5 ** (days_elapsed / hl)

        if new_w < EDGE_MIN_WEIGHT and e["relation"] not in STRUCTURAL_RELATIONS:
            conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
            edges_pruned += 1
        else:
            conn.execute(
                "UPDATE edges SET weight = ?, last_decayed = ? WHERE id = ?",
                (max(new_w, EDGE_MIN_WEIGHT) if e["relation"] in STRUCTURAL_RELATIONS else new_w,
                 now, e["id"]),
            )

    conn.commit()
    return {
        "updated": updated,
        "archived": archived,
        "deleted": deleted,
        "edges_pruned": edges_pruned,
    }
