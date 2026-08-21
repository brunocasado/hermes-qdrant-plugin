"""Cross-process coordination for index mutations and shared JSON metadata."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

try:
    from .paths import ensure_data_dir
except ImportError:
    from paths import ensure_data_dir

LOCKS_DIR = ensure_data_dir() / "index-locks"


class IndexAlreadyRunning(RuntimeError):
    """Raised when a root or collection already has an active writer."""

    def __init__(self, root: str, collection: str, owner: dict | None = None):
        self.root = root
        self.collection = collection
        self.owner = owner or {}
        super().__init__(f"index already running for {root} ({collection})")


def canonical_root(root: str) -> str:
    return str(Path(root).expanduser().resolve())


def _paths(kind: str, value: str) -> tuple[Path, Path]:
    key = hashlib.sha256(value.encode()).hexdigest()[:24]
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    return LOCKS_DIR / f"{kind}-{key}.lock", LOCKS_DIR / f"{kind}-{key}.json"


def _read_owner(meta_path: Path) -> dict:
    try:
        value = json.loads(meta_path.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


@contextmanager
def index_operation_lock(root: str, collection: str) -> Iterator[None]:
    """Hold non-blocking locks for both canonical root and collection."""
    root = canonical_root(root)
    specs = sorted(
        [_paths("root", root), _paths("collection", collection)],
        key=lambda pair: str(pair[0]),
    )
    held: list[tuple[FileLock, Path]] = []
    try:
        for lock_path, meta_path in specs:
            lock = FileLock(str(lock_path), timeout=0, thread_local=False)
            try:
                lock.acquire()
            except Timeout:
                raise IndexAlreadyRunning(root, collection, _read_owner(meta_path)) from None
            held.append((lock, meta_path))
            meta_path.write_text(json.dumps({
                "pid": os.getpid(),
                "root": root,
                "collection": collection,
                "started_at": time.time(),
            }))
        yield
    finally:
        for lock, meta_path in reversed(held):
            meta_path.unlink(missing_ok=True)
            lock.release()


@contextmanager
def metadata_lock(name: str, timeout: float = 10.0) -> Iterator[None]:
    """Short-lived lock for a shared read-modify-write JSON transaction."""
    lock_path, _ = _paths("metadata", name)
    lock = FileLock(str(lock_path), timeout=timeout, thread_local=False)
    with lock:
        yield


def atomic_write_json(path: Path, data: dict) -> None:
    """Durably replace a JSON file without exposing partial contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
