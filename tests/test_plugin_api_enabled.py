from pathlib import Path

from dashboard import plugin_api


class FakeConfig:
    def __init__(self):
        self.values = {}
        self.read_roots = []

    def is_enabled(self, root):
        root = str(Path(root).resolve())
        self.read_roots.append(root)
        return self.values.get(root, False)

    def set_enabled(self, root, value):
        self.values[str(Path(root).resolve())] = bool(value)


def test_enabled_routes_are_project_scoped(monkeypatch, tmp_path):
    fake = FakeConfig()
    monkeypatch.setattr(plugin_api, "_qconfig", lambda: fake)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    assert plugin_api.get_enabled(str(first)) == {
        "root": str(first.resolve()), "enabled": False,
    }
    assert plugin_api.put_enabled(str(first), {"enabled": True}) == {
        "root": str(first.resolve()), "enabled": True,
    }
    assert plugin_api.get_enabled(str(first))["enabled"] is True
    assert plugin_api.get_enabled(str(second))["enabled"] is False


def test_status_non_indexable_still_reads_that_root(monkeypatch, tmp_path):
    fake = FakeConfig()
    monkeypatch.setattr(plugin_api, "_qconfig", lambda: fake)
    monkeypatch.setattr(plugin_api, "_is_indexable", lambda root: (False, "blocked"))
    root = tmp_path / "project"

    result = plugin_api.get_status(str(root))

    assert result["enabled"] is False
    assert fake.read_roots == [str(root.resolve())]


def test_refresh_disabled_check_uses_requested_root(monkeypatch, tmp_path):
    fake = FakeConfig()
    monkeypatch.setattr(plugin_api, "_qconfig", lambda: fake)
    monkeypatch.setattr(plugin_api, "_core", lambda: (object(), object()))
    root = tmp_path / "project"
    root.mkdir()

    result = plugin_api.post_refresh(str(root))

    assert result == {"started": False, "reason": "disabled", "enabled": False}
    assert fake.read_roots == [str(root.resolve())]


def test_reindex_does_not_overwrite_running_operation(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    class FakeRegistry:
        @staticmethod
        def collection_for_root(_root):
            return "project"

    class FakeCore:
        @staticmethod
        def derive_collection_name(_root):
            return "project"

        @staticmethod
        def resolve_collection_name(_root, requested=None):
            return requested or "project"

    monkeypatch.setattr(plugin_api, "_core", lambda: (FakeCore, FakeRegistry))
    monkeypatch.setattr(plugin_api, "_is_indexable", lambda _root: (True, ""))
    with plugin_api._OPS_LOCK:
        plugin_api._OPS[str(root.resolve())] = {
            "status": "running", "collection": "project", "at": 1,
        }
    try:
        result = plugin_api.post_reindex(str(root))
    finally:
        with plugin_api._OPS_LOCK:
            plugin_api._OPS.pop(str(root.resolve()), None)

    assert result == {
        "started": False,
        "reason": "already-running",
        "collection": "project",
    }
