"""Index registry: collection name <-> project root metadata."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    from . import paths as _paths
    from . import index_lock as _index_lock
except ImportError:
    import paths as _paths
    import index_lock as _index_lock

DATA_DIR = _paths.ensure_data_dir()
REGISTRY_PATH = DATA_DIR / "registry.json"
INDEX_SCHEMA_VERSION = 2  # named dense + lexical vectors, rel_path point ids


def load() -> dict:
    """Read a complete old-or-new registry snapshot."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        value = json.loads(REGISTRY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def save(reg: dict) -> None:
    """Atomically replace the registry (administrative restore only)."""
    with _index_lock.metadata_lock("registry"):
        _index_lock.atomic_write_json(REGISTRY_PATH, reg)


def mutate(change: Callable[[dict], None]) -> dict:
    """Apply one read-modify-write transaction without losing other writers."""
    with _index_lock.metadata_lock("registry"):
        reg = load()
        change(reg)
        _index_lock.atomic_write_json(REGISTRY_PATH, reg)
        return reg


def record(collection: str, root: str, file_count: int) -> None:
    resolved = str(Path(root).resolve())

    def _record(reg: dict) -> None:
        reg[collection] = {
            "root": resolved,
            "last_indexed": datetime.now().isoformat(timespec="seconds"),
            "file_count": file_count,
            "schema_version": INDEX_SCHEMA_VERSION,
        }

    mutate(_record)


def remove(collection: str) -> bool:
    removed = False

    def _remove(reg: dict) -> None:
        nonlocal removed
        removed = reg.pop(collection, None) is not None

    mutate(_remove)
    return removed


def collection_for_root(root: str) -> str | None:
    root = str(Path(root).resolve())
    for name, entry in load().items():
        if entry.get("root") == root:
            return name
    return None


def root_for_collection(collection: str) -> str | None:
    entry = load().get(collection)
    return str(entry.get("root")) if isinstance(entry, dict) and entry.get("root") else None
