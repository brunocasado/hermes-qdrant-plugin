import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import core
import index_lock
import registry


class FakeClient:
    def __init__(self, schema_current=False):
        self.events = []
        self.points = 10
        self.last_points = []
        self.schema_current = schema_current

    def count(self, collection_name):
        return SimpleNamespace(count=self.points)

    def create_collection(self, **kwargs):
        self.events.append(("create_collection", kwargs["collection_name"]))
        self.points = 0
        self.schema_current = True

    def delete_collection(self, collection_name):
        self.events.append(("delete_collection", collection_name))
        self.points = 0

    def delete(self, collection_name, points_selector, **kwargs):
        condition = points_selector.filter.must[0]
        self.events.append(("delete_file", condition.match.value))

    def upsert(self, collection_name, points):
        self.last_points.extend(points)
        self.events.append(("upsert", [p.payload["rel_path"] for p in points]))
        self.points += len(points)

    def get_collection(self, collection_name):
        vectors = {"dense": SimpleNamespace(size=768)} if self.schema_current else SimpleNamespace(size=768)
        sparse = {"lexical": SimpleNamespace()} if self.schema_current else None
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=vectors, sparse_vectors=sparse,
        )))


def _isolate(monkeypatch, tmp_path, fake):
    data = tmp_path / "data"
    monkeypatch.setattr(core, "HASH_CACHE_PATH", data / "hash-cache.json")
    monkeypatch.setattr(core, "LEGACY_CACHE_PATH", data / "legacy.json")
    monkeypatch.setattr(registry, "DATA_DIR", data)
    monkeypatch.setattr(registry, "REGISTRY_PATH", data / "registry.json")
    monkeypatch.setattr(index_lock, "LOCKS_DIR", data / "locks")
    monkeypatch.setattr(core, "get_client", lambda: fake)
    monkeypatch.setattr(core, "live_collection_state", lambda *a, **k: ("ok", fake.points))

    async def fake_embed(texts, emb_cfg, semaphore):
        return [[0.1] * 768 for _ in texts]

    monkeypatch.setattr(core, "_embed_batch", fake_embed)


def _deleted_rel_paths(fake):
    return [value for event, value in fake.events if event == "delete_file"]


def test_discovery_excludes_tool_worktrees_but_keeps_project_sources(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.ts").write_text("export const main = 1\n")
    (root / ".worktrees" / "branch").mkdir(parents=True)
    (root / ".worktrees" / "branch" / "copy.ts").write_text("copy\n")
    (root / "package" / ".kilo" / "worktrees" / "task").mkdir(parents=True)
    (root / "package" / ".kilo" / "worktrees" / "task" / "copy.ts").write_text("copy\n")
    (root / "service" / "backups" / "snapshot").mkdir(parents=True)
    (root / "service" / "backups" / "snapshot" / "slots.json").write_text("{}\n")

    discovered = {
        str(Path(path).relative_to(root)) for path in core.discover_files(str(root))
    }

    assert discovered == {"src/main.ts"}


def test_changed_file_deletes_old_points_before_upsert(monkeypatch, tmp_path):
    fake = FakeClient()
    _isolate(monkeypatch, tmp_path, fake)
    root = tmp_path / "project"
    root.mkdir()
    source = root / "a.py"
    source.write_text("print('new')\n")
    core.save_hash_cache("project", {"a.py": {"hash": "old"}})

    asyncio.run(core.index_directory(str(root), collection_name="project"))

    assert _deleted_rel_paths(fake) == ["a.py"]
    assert fake.events.index(("delete_file", "a.py")) < next(
        i for i, event in enumerate(fake.events) if event[0] == "upsert"
    )


def test_deleted_file_is_removed_from_qdrant_and_cache(monkeypatch, tmp_path):
    fake = FakeClient()
    _isolate(monkeypatch, tmp_path, fake)
    root = tmp_path / "project"
    root.mkdir()
    core.save_hash_cache("project", {"deleted.py": {"hash": "old"}})

    asyncio.run(core.index_directory(str(root), collection_name="project"))

    assert _deleted_rel_paths(fake) == ["deleted.py"]
    cache = json.loads(core.HASH_CACHE_PATH.read_text())
    assert cache["project"] == {}


def test_registered_legacy_schema_is_rebuilt_automatically(monkeypatch, tmp_path):
    fake = FakeClient()
    _isolate(monkeypatch, tmp_path, fake)
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("print('x')\n")
    registry.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.REGISTRY_PATH.write_text(json.dumps({
        "project": {"root": str(root.resolve()), "file_count": 1}
    }))

    asyncio.run(core.index_directory(str(root), collection_name="project"))

    names = [event[0] for event in fake.events]
    assert names[:2] == ["delete_collection", "create_collection"]
    assert json.loads(registry.REGISTRY_PATH.read_text())["project"]["schema_version"] == core.INDEX_SCHEMA_VERSION


def test_interrupted_schema_upgrade_resumes_without_recreating_collection(monkeypatch, tmp_path):
    """A new-schema collection plus a partial cache must resume, never restart."""
    fake = FakeClient(schema_current=True)
    _isolate(monkeypatch, tmp_path, fake)
    root = tmp_path / "project"
    root.mkdir()
    completed = root / "completed.py"
    pending = root / "pending.py"
    completed.write_text("print('done')\n")
    pending.write_text("print('pending')\n")
    core.save_hash_cache("project", {
        "completed.py": {"hash": core.file_hash(str(completed))},
    })
    registry.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.REGISTRY_PATH.write_text(json.dumps({
        "project": {"root": str(root.resolve()), "file_count": 2}
    }))

    asyncio.run(core.index_directory(str(root), collection_name="project"))

    names = [event[0] for event in fake.events]
    assert "delete_collection" not in names
    assert "create_collection" not in names
    assert [p.payload["rel_path"] for p in fake.last_points] == ["pending.py"]
    assert json.loads(registry.REGISTRY_PATH.read_text())["project"]["schema_version"] == core.INDEX_SCHEMA_VERSION


def test_reindex_recreates_collection_before_writing(monkeypatch, tmp_path):
    fake = FakeClient()
    _isolate(monkeypatch, tmp_path, fake)
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("print('x')\n")

    asyncio.run(core.index_directory(str(root), collection_name="project", reindex=True))

    names = [event[0] for event in fake.events]
    assert names[:2] == ["delete_collection", "create_collection"]
    assert "upsert" in names
    assert set(fake.last_points[0].vector) == {"dense", "lexical"}


def test_point_payload_uses_relative_path_and_metadata(tmp_path):
    root = tmp_path / "project"
    source = root / "internal" / "scheduler.py"
    source.parent.mkdir(parents=True)
    source.write_text("def schedule():\n    pass\n")

    payload = core.build_point_payload(
        root=str(root),
        filepath=str(source),
        chunk="def schedule(): pass",
        line_start=1,
        line_end=2,
        chunk_index=0,
        file_hash="abc",
        language="python",
        symbols=["schedule"],
        chunk_type="function",
    )

    assert payload["rel_path"] == "internal/scheduler.py"
    assert payload["basename"] == "scheduler.py"
    assert payload["language"] == "python"
    assert payload["symbols"] == ["schedule"]
    assert payload["chunk_type"] == "function"
    assert core._file_point_id(payload["rel_path"], 0) == core._file_point_id(payload["rel_path"], 0)


def test_status_marks_deleted_cached_files_as_stale(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    registry_data = {
        "project": {
            "root": str(root.resolve()),
            "last_indexed": "now",
            "file_count": 1,
        }
    }

    status = core.compute_status(
        str(root),
        cache={"deleted.py": {"hash": "old"}},
        registry=registry_data,
        collection="project",
    )

    assert status["deleted"] == 1
    assert status["stale"] is True
