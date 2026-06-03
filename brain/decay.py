import math
import time


def current_weight(weight: float, last_accessed: float, half_life_days: float) -> float:
    """Ebbinghaus forgetting curve: w(t) = w0 * exp(-t / half_life)."""
    if math.isinf(half_life_days):
        return weight
    days_elapsed = (time.time() - last_accessed) / 86400.0
    return weight * math.exp(-days_elapsed / half_life_days)


def run_decay(conn) -> dict:
    """
    Update weights for all non-archived nodes.
    Archive nodes that fall below 0.10.
    Delete nodes that have been archived for 7+ days.
    Returns counts of updated / archived / deleted.
    """
    nodes = conn.execute(
        "SELECT id, weight, last_accessed, half_life_days, archived FROM nodes"
    ).fetchall()

    updated = archived = deleted = 0
    now = time.time()

    for n in nodes:
        if n["archived"]:
            # check if it's been archived long enough to delete
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

    conn.commit()
    return {"updated": updated, "archived": archived, "deleted": deleted}
