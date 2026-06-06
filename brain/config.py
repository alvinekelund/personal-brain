import json
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
