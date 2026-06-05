import os
from pathlib import Path

DATA_DIR = Path.home() / ".personal-brain"
DB_PATH = DATA_DIR / "brain.db"

DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv():
    """Populate os.environ from a .env file (data dir, then cwd) without
    overriding variables already set in the real environment."""
    for env_path in (DATA_DIR / ".env", Path.cwd() / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()
