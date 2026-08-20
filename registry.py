"""Index registry: collection name <-> project root, with file/last-indexed metadata."""
import json
from datetime import datetime
from pathlib import Path

# Dual-mode import (plugin package vs plugin-dir-on-sys.path).
try:
    from . import paths as _paths
except ImportError:
    import paths as _paths

DATA_DIR = _paths.ensure_data_dir()
REGISTRY_PATH = DATA_DIR / "registry.json"


def load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save(reg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def record(collection: str, root: str, file_count: int) -> None:
    reg = load()
    reg[collection] = {"root": str(Path(root).resolve()),
                       "last_indexed": datetime.now().isoformat(timespec="seconds"),
                       "file_count": file_count}
    save(reg)


def collection_for_root(root: str) -> str | None:
    root = str(Path(root).resolve())
    for name, entry in load().items():
        if entry.get("root") == root:
            return name
    return None
