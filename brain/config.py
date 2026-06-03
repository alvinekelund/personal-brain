import json
from brain import DATA_DIR

CONFIG_PATH = DATA_DIR / "config.json"


def load() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_user() -> str:
    return load().get("user", "")


def set_user(name: str):
    cfg = load()
    cfg["user"] = name
    save(cfg)
