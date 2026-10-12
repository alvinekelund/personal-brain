import json
from pathlib import Path
from brain import DATA_DIR

CONFIG_PATH = DATA_DIR / "config.json"


def load() -> dict:
    """Load config, tolerating a missing, unreadable, or corrupt file.

    A broken ~/.personal-brain/config.json must never crash every CLI command —
    fall back to empty config (it gets rewritten on the next save).
    """
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_user() -> str:
    return load().get("user", "")


def set_user(name: str):
    cfg = load()
    cfg["user"] = name
    save(cfg)


def distinct_pairs() -> set:
    """Name pairs marked as two different things (`brain distinct A B`); the duplicate checks skip them."""
    out = set()
    for p in load().get("distinct_pairs", []):
        if isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p):
            out.add(frozenset(x.strip().lower() for x in p))
    return out


def add_distinct(a: str, b: str) -> bool:
    """Record that A and B are different things; False when that was already recorded."""
    if frozenset((a.strip().lower(), b.strip().lower())) in distinct_pairs():
        return False
    cfg = load()
    cfg.setdefault("distinct_pairs", []).append([a, b])
    save(cfg)
    return True


def vault_dir() -> Path:
    """The vault location: config `vault_dir` if set, else <data>/vault.
    Lives here (not in vault.py) so graph/loops can resolve it without importing the renderer."""
    configured = load().get("vault_dir", "")
    return Path(configured).expanduser() if configured else DATA_DIR / "vault"
