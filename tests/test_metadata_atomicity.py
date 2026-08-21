import json
from concurrent.futures import ThreadPoolExecutor

import core
import registry


def test_parallel_hash_cache_writers_preserve_every_collection(monkeypatch, tmp_path):
    cache_path = tmp_path / "hash-cache.json"
    monkeypatch.setattr(core, "HASH_CACHE_PATH", cache_path)
    monkeypatch.setattr(core, "LEGACY_CACHE_PATH", tmp_path / "no-legacy.json")
    count = 40

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(core.save_hash_cache, f"collection-{i}", {f"file-{i}.py": {"hash": str(i)}})
            for i in range(count)
        ]
        for future in futures:
            future.result()

    raw = json.loads(cache_path.read_text())
    assert len(raw) == count
    for i in range(count):
        assert raw[f"collection-{i}"][f"file-{i}.py"]["hash"] == str(i)


def test_parallel_registry_records_preserve_every_collection(monkeypatch, tmp_path):
    path = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "DATA_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_PATH", path)
    count = 40

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(registry.record, f"collection-{i}", str(tmp_path / f"root-{i}"), i)
            for i in range(count)
        ]
        for future in futures:
            future.result()

    raw = json.loads(path.read_text())
    assert len(raw) == count
    for i in range(count):
        assert raw[f"collection-{i}"]["root"] == str((tmp_path / f"root-{i}").resolve())
