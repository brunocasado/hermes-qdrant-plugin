import multiprocessing as mp
import threading
from pathlib import Path

import pytest


def _hold_lock(root, collection, ready, release, data_dir):
    import index_lock
    index_lock.LOCKS_DIR = Path(data_dir)
    with index_lock.index_operation_lock(root, collection):
        ready.set()
        release.wait(5)


def _isolated_locks(monkeypatch, tmp_path):
    import index_lock
    locks = tmp_path / "locks"
    monkeypatch.setattr(index_lock, "LOCKS_DIR", locks)
    return index_lock


def test_same_project_second_writer_fails_fast(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    entered = threading.Event()
    release = threading.Event()

    def first():
        with index_lock.index_operation_lock(str(root), "project"):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(2)
    try:
        with pytest.raises(index_lock.IndexAlreadyRunning) as exc:
            with index_lock.index_operation_lock(str(root), "other-collection"):
                pass
        assert exc.value.root == str(root.resolve())
    finally:
        release.set()
        thread.join(2)


def test_same_project_is_blocked_across_processes(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    ready = mp.Event()
    release = mp.Event()
    proc = mp.Process(target=_hold_lock, args=(str(root), "project", ready, release, str(index_lock.LOCKS_DIR)))
    proc.start()
    assert ready.wait(3)
    try:
        with pytest.raises(index_lock.IndexAlreadyRunning):
            with index_lock.index_operation_lock(str(root), "project"):
                pass
    finally:
        release.set()
        proc.join(3)
        if proc.is_alive():
            proc.kill()


def test_different_roots_same_collection_cannot_overlap(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    first = tmp_path / "a" / "service"
    second = tmp_path / "b" / "service"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    with index_lock.index_operation_lock(str(first), "service"):
        with pytest.raises(index_lock.IndexAlreadyRunning):
            with index_lock.index_operation_lock(str(second), "service"):
                pass


def test_different_projects_and_collections_can_overlap(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    with index_lock.index_operation_lock(str(first), "first"):
        with index_lock.index_operation_lock(str(second), "second"):
            pass


def test_lock_released_after_exception(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(RuntimeError):
        with index_lock.index_operation_lock(str(root), "project"):
            raise RuntimeError("boom")

    with index_lock.index_operation_lock(str(root), "project"):
        pass


def test_symlink_alias_uses_same_project_lock(monkeypatch, tmp_path):
    index_lock = _isolated_locks(monkeypatch, tmp_path)
    root = tmp_path / "project"
    alias = tmp_path / "alias"
    root.mkdir()
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink unavailable")

    with index_lock.index_operation_lock(str(root), "one"):
        with pytest.raises(index_lock.IndexAlreadyRunning):
            with index_lock.index_operation_lock(str(alias), "two"):
                pass


def test_core_index_directory_is_the_admission_boundary(monkeypatch, tmp_path):
    import asyncio
    import core
    import index_lock

    monkeypatch.setattr(index_lock, "LOCKS_DIR", tmp_path / "locks")
    monkeypatch.setattr(core._registry, "collection_for_root", lambda _root: None)
    monkeypatch.setattr(core._registry, "root_for_collection", lambda _collection: None)
    root = tmp_path / "project"
    root.mkdir()
    entered = threading.Event()
    release = threading.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.to_thread(release.wait, 5)
        return "done"

    monkeypatch.setattr(core, "_index_directory_locked", blocked, raising=False)

    first = threading.Thread(
        target=lambda: asyncio.run(core.index_directory(str(root), collection_name="project"))
    )
    first.start()
    assert entered.wait(2)
    try:
        with pytest.raises(index_lock.IndexAlreadyRunning):
            asyncio.run(core.index_directory(str(root), collection_name="project"))
    finally:
        release.set()
        first.join(2)


def test_collection_name_is_disambiguated_for_different_roots(monkeypatch, tmp_path):
    import core

    first = tmp_path / "a" / "service"
    second = tmp_path / "b" / "service"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    monkeypatch.setattr(core._registry, "collection_for_root", lambda root: None)
    monkeypatch.setattr(
        core._registry,
        "root_for_collection",
        lambda collection: str(first.resolve()) if collection == "service" else None,
    )

    resolved = core.resolve_collection_name(str(second))

    assert resolved.startswith("service-")
    assert resolved == core.resolve_collection_name(str(second))


def test_explicit_collection_owned_by_other_root_is_rejected(monkeypatch, tmp_path):
    import core

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(core._registry, "collection_for_root", lambda root: None)
    monkeypatch.setattr(core._registry, "root_for_collection", lambda collection: str(first.resolve()))

    with pytest.raises(core.CollectionNameConflict):
        core.resolve_collection_name(str(second), requested="shared")
