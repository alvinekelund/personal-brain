import math
import time

EDGE_BASE_HALF_LIFE = 90.0   # days for a once-seen edge
EDGE_MIN_WEIGHT = 0.05       # below this, delete the edge
ARCHIVE_THRESHOLD = 0.10     # weight below this → archived
IMPORTANCE_GAIN = 4.0        # important nodes decay up to (1+gain)x slower
IMPORTANCE_FLOOR = 0.15      # weight never drops below importance * this


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
        "SELECT * FROM nodes WHERE archived=0 AND weight >= ? AND half_life_days < 1e12",
        (threshold,),
    ).fetchall()
    cands = []
    for r in rows:
        imp = r["importance"] if "importance" in r.keys() else 0.0
        days = days_until_archive(r["weight"], r["half_life_days"], threshold, imp)
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


def run_decay(conn) -> dict:
    """
    Update weights for all non-archived nodes and all edges.
    Archive nodes below 0.10, delete nodes archived 7+ days.
    Delete edges below EDGE_MIN_WEIGHT.
    Returns counts of updated / archived / deleted nodes + edges pruned.
    """
    nodes = conn.execute(
        "SELECT id, weight, last_accessed, half_life_days, archived, importance FROM nodes"
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

        new_w = current_weight(
            n["weight"], n["last_accessed"], n["half_life_days"], n["importance"]
        )
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
