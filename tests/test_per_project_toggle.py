import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import qconfig


def _isolated_config(monkeypatch, tmp_path):
    data = tmp_path / "data"
    monkeypatch.setattr(qconfig, "DATA_DIR", data)
    monkeypatch.setattr(qconfig, "CONFIG_PATH", data / "config.json")
    return data


def test_unknown_project_is_disabled_by_default(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)

    assert qconfig.is_enabled(str(tmp_path / "project")) is False


def test_enabling_one_project_does_not_enable_another(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    qconfig.set_enabled(str(first), True)

    assert qconfig.is_enabled(str(first)) is True
    assert qconfig.is_enabled(str(second)) is False
    raw = json.loads(qconfig.CONFIG_PATH.read_text())
    assert raw["projects"] == {str(first.resolve()): True}


def test_disabling_project_is_persisted(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    root = tmp_path / "project"
    root.mkdir()

    qconfig.set_enabled(str(root), True)
    qconfig.set_enabled(str(root), False)

    assert qconfig.is_enabled(str(root)) is False
    assert json.loads(qconfig.CONFIG_PATH.read_text())["projects"][str(root.resolve())] is False


def test_project_key_is_canonicalized(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    qconfig.set_enabled(str(alias), True)

    assert qconfig.is_enabled(str(root)) is True
    assert list(json.loads(qconfig.CONFIG_PATH.read_text())["projects"]) == [str(root.resolve())]


def test_legacy_true_seeds_registered_projects_once(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    qconfig.DATA_DIR.mkdir(parents=True)
    qconfig.CONFIG_PATH.write_text(json.dumps({"enabled": True, "qdrant": {"host": "q"}}))
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(qconfig, "_registry_roots", lambda: [str(r) for r in roots])

    qconfig.migrate_legacy_enabled()

    raw = json.loads(qconfig.CONFIG_PATH.read_text())
    assert "enabled" not in raw
    assert raw["qdrant"] == {"host": "q"}
    assert raw["projects"] == {str(r.resolve()): True for r in roots}
    assert qconfig.is_enabled(str(roots[0])) is True
    assert qconfig.is_enabled(str(tmp_path / "new")) is False


def test_legacy_false_migrates_to_empty_projects(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    qconfig.DATA_DIR.mkdir(parents=True)
    qconfig.CONFIG_PATH.write_text(json.dumps({"enabled": False}))
    monkeypatch.setattr(qconfig, "_registry_roots", lambda: [str(tmp_path / "known")])

    qconfig.migrate_legacy_enabled()

    assert json.loads(qconfig.CONFIG_PATH.read_text()) == {"projects": {}}


def test_existing_projects_map_is_not_overwritten_by_legacy_flag(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    root = tmp_path / "kept"
    root.mkdir()
    qconfig.DATA_DIR.mkdir(parents=True)
    qconfig.CONFIG_PATH.write_text(json.dumps({
        "enabled": True,
        "projects": {str(root.resolve()): False},
    }))
    monkeypatch.setattr(qconfig, "_registry_roots", lambda: [str(tmp_path / "other")])

    qconfig.migrate_legacy_enabled()

    raw = json.loads(qconfig.CONFIG_PATH.read_text())
    assert raw["projects"] == {str(root.resolve()): False}
    assert qconfig.is_enabled(str(root)) is False


def test_parallel_project_toggles_preserve_all_entries(monkeypatch, tmp_path):
    _isolated_config(monkeypatch, tmp_path)
    roots = [tmp_path / f"project-{i}" for i in range(30)]
    for root in roots:
        root.mkdir()

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(qconfig.set_enabled, str(root), True) for root in roots]
        for future in futures:
            future.result()

    raw = json.loads(qconfig.CONFIG_PATH.read_text())
    assert raw["projects"] == {str(root.resolve()): True for root in roots}
