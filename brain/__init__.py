import os
from pathlib import Path

DATA_DIR = Path.home() / ".personal-brain"
DB_PATH = DATA_DIR / "brain.db"

DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv():
    """Populate os.environ from a .env file (data dir, then cwd).

    A non-empty value in the file fills in any variable that is unset OR set to
    an empty/whitespace string in the real environment — so a stray
    `export GEMINI_API_KEY=` in the shell can't silently shadow a real key.
    A genuinely-set (non-empty) environment variable still wins.
    """
    for env_path in (DATA_DIR / ".env", Path.cwd() / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if val and not os.environ.get(key, "").strip():
                os.environ[key] = val


_load_dotenv()
