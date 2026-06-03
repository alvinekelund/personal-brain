from pathlib import Path

DATA_DIR = Path.home() / ".personal-brain"
DB_PATH = DATA_DIR / "brain.db"

DATA_DIR.mkdir(exist_ok=True)
