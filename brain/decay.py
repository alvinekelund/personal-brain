import math
import time

EDGE_BASE_HALF_LIFE = 90.0   # days for a once-seen edge
EDGE_MIN_WEIGHT = 0.05       # below this, delete the edge
ARCHIVE_THRESHOLD = 0.10     # weight below this → archived


def days_until_archive(weight: float, half_life_days: float,
                       threshold: float = ARCHIVE_THRESHOLD) -> float:
    """Days until an untouched node decays from `weight` to the archive threshold.

    Solves w * 0.5**(d/H) = threshold for d. inf for never-decaying types.
    """
    if math.isinf(half_life_days):
        return float("inf")
    if weight <= threshold:
        return 0.0
    return half_life_days * math.log2(weight / threshold)


def at_risk_nodes(conn, limit: int = 5, threshold: float = ARCHIVE_THRESHOLD) -> list:
    """Active, decaying nodes closest to being archived (lowest weight first).

    Returns dicts with name/type/weight and an estimated days_left until archive,
    so `brain status` can show what's about to be forgotten. Never-decaying types
    (person/organization) are excluded — they're not at risk.
    """
    rows = conn.execute(
        "SELECT * FROM nodes WHERE archived=0 AND weight >= ? AND half_life_days < 1e12 "
        "ORDER BY weight ASC LIMIT ?",
        (threshold, limit),
    ).fetchall()
    return [
        {
            "name": r["name"],
            "type": r["type"],
            "weight": r["weight"],
            "days_left": days_until_archive(r["weight"], r["half_life_days"], threshold),
        }
        for r in rows
    ]


def current_weight(weight: float, last_accessed: float, half_life_days: float) -> float:
    """Half-life decay: w(t) = w0 * 0.5**(t / half_life).

    Uses base-1/2 so half_life_days is a *true* half-life — at t == half_life the
    weight is exactly halved, matching the documented table. (The previous
    exp(-t/half_life) form made half_life a mean-lifetime: 0.368 at t=H, not 0.5.)
    """
    if math.isinf(half_life_days):
        return weight
    days_elapsed = (time.time() - last_accessed) / 86400.0
    return weight * 0.5 ** (days_elapsed / half_life_days)


def edge_half_life(reinforcement_count: int) -> float:
    """
    Hebbian scaling: the more co-occurrences, the slower an edge decays.
    half_life = base * ln(1 + count)
    count=1 → ~62 days, count=5 → ~161 days, count=20 → ~271 days
    """
    return EDGE_BASE_HALF_LIFE * math.log1p(reinforcement_count)


def run_decay(conn) -> dict:
    """
    Update weights for all non-archived nodes and all edges.
    Archive nodes below 0.10, delete nodes archived 7+ days.
    Delete edges below EDGE_MIN_WEIGHT.
    Returns counts of updated / archived / deleted nodes + edges pruned.
    """
    nodes = conn.execute(
        "SELECT id, weight, last_accessed, half_life_days, archived FROM nodes"
    ).fetchall()

    updated = archived = deleted = edges_pruned = 0
    now = time.time()

    for n in nodes:
        if n["archived"]:
            days_archived = (now - n["last_accessed"]) / 86400.0
            if days_archived > 7:
                conn.execute("DELETE FROM nodes WHERE id = ?", (n["id"],))
                deleted += 1
            continue

        new_w = current_weight(n["weight"], n["last_accessed"], n["half_life_days"])
        new_w = max(0.0, min(1.0, new_w))

        if new_w < 0.10:
            conn.execute(
                "UPDATE nodes SET weight = ?, archived = 1 WHERE id = ?",
                (new_w, n["id"]),
            )
            archived += 1
        else:
            conn.execute(
                "UPDATE nodes SET weight = ? WHERE id = ?",
                (new_w, n["id"]),
            )
            updated += 1

    # edge decay
    edges = conn.execute(
        "SELECT id, weight, last_reinforced, reinforcement_count FROM edges"
    ).fetchall()

    for e in edges:
        hl = edge_half_life(e["reinforcement_count"])
        days_elapsed = (now - e["last_reinforced"]) / 86400.0
        new_w = e["weight"] * 0.5 ** (days_elapsed / hl)

        if new_w < EDGE_MIN_WEIGHT:
            conn.execute("DELETE FROM edges WHERE id = ?", (e["id"],))
            edges_pruned += 1
        else:
            conn.execute(
                "UPDATE edges SET weight = ? WHERE id = ?",
                (new_w, e["id"]),
            )

    conn.commit()
    return {
        "updated": updated,
        "archived": archived,
        "deleted": deleted,
        "edges_pruned": edges_pruned,
    }
