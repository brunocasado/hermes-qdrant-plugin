"""hermes-qdrant-plugin — runtime-editable server & embedding config.

Lets you point the plugin at a different Qdrant server and/or a different
embedding endpoint without touching code. Three ways to set values (highest
precedence first): environment variables, config.json, then built-in defaults.

config.json shape (all keys optional):

  {
    "projects":  {"/absolute/project/root": true},
    "qdrant":    {"host": "localhost", "port": 6333},
    "embedding": {"base_url": "http://localhost:8080/v1",
                  "model": "embeddings", "api_key": "EMPTY",
                  "vector_dim": 768}
  }

Automatic indexing is disabled by default and remembered per canonical project
root. Manual indexing remains available even when automatic indexing is off.
A legacy top-level ``enabled`` flag is migrated once: registered projects keep
their previous on-state while unknown projects stay off.

This module is intentionally dependency-free (no `core`, no `qdrant_client`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from . import paths as _paths
    from . import index_lock as _index_lock
except ImportError:
    import paths as _paths
    import index_lock as _index_lock

DATA_DIR = _paths.ensure_data_dir()
CONFIG_PATH = DATA_DIR / "config.json"

DEFAULTS: dict[str, dict[str, Any]] = {
    "qdrant": {"host": "localhost", "port": 6333},
    "embedding": {
        "base_url": "http://localhost:8080/v1",
        "model": "embeddings",
        "api_key": "EMPTY",
        "vector_dim": 768,
    },
}
DEFAULT_ENABLED = False

_ENV_MAP = {
    "QDRANT_HOST": ("qdrant", "host"),
    "QDRANT_PORT": ("qdrant", "port"),
    "EMBEDDING_BASE_URL": ("embedding", "base_url"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "EMBEDDING_API_KEY": ("embedding", "api_key"),
    "EMBEDDING_VECTOR_DIM": ("embedding", "vector_dim"),
}


def _read_file() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_file(file_cfg: dict[str, Any]) -> None:
    _index_lock.atomic_write_json(CONFIG_PATH, file_cfg)


def _canonical_root(root: str) -> str:
    return str(Path(root).expanduser().resolve())


def _projects_map(file_cfg: dict[str, Any]) -> dict[str, bool]:
    projects = file_cfg.get("projects")
    if not isinstance(projects, dict):
        return {}
    return {str(k): v for k, v in projects.items() if isinstance(v, bool)}


def _registry_roots() -> list[str]:
    """Return registered roots without importing the heavy Qdrant core."""
    try:
        try:
            from . import registry
        except ImportError:
            import registry
        return [str(entry["root"]) for entry in registry.load().values()
                if isinstance(entry, dict) and entry.get("root")]
    except Exception:
        return []


def migrate_legacy_enabled() -> None:
    """Convert the old global flag to explicit per-project state once."""
    with _index_lock.metadata_lock("config"):
        file_cfg = _read_file()
        if "projects" in file_cfg:
            return
        legacy = file_cfg.pop("enabled", None)
        if not isinstance(legacy, bool):
            return
        roots = _registry_roots() if legacy else []
        file_cfg["projects"] = {_canonical_root(root): True for root in roots}
        _write_file(file_cfg)


def load_config() -> dict[str, Any]:
    """Return the fully-resolved config (defaults <- file <- env)."""
    file_cfg = _read_file()
    resolved: dict[str, Any] = {
        "enabled": DEFAULT_ENABLED,
        "projects": _projects_map(file_cfg),
        "qdrant": dict(DEFAULTS["qdrant"]),
        "embedding": dict(DEFAULTS["embedding"]),
    }
    # Keep the legacy value visible until migration is triggered by a project
    # lookup; no live indexing caller should use this summary field.
    if isinstance(file_cfg.get("enabled"), bool):
        resolved["enabled"] = file_cfg["enabled"]
    for section in ("qdrant", "embedding"):
        fsec = file_cfg.get(section)
        if isinstance(fsec, dict):
            for key in resolved[section]:
                if key in fsec and fsec[key] is not None:
                    resolved[section][key] = fsec[key]
    for env, (section, key) in _ENV_MAP.items():
        val = os.environ.get(env)
        if val is None or val == "":
            continue
        if key in ("port", "vector_dim"):
            try:
                resolved[section][key] = int(val)
            except ValueError:
                continue
        else:
            resolved[section][key] = val
    return resolved


def is_enabled(root: str | None = None) -> bool:
    """Return automatic-index state for one canonical project root."""
    migrate_legacy_enabled()
    if not root:
        return DEFAULT_ENABLED
    return _projects_map(_read_file()).get(_canonical_root(root), DEFAULT_ENABLED)


def set_enabled(root: str, value: bool) -> dict[str, Any]:
    """Persist automatic-index state for exactly one project."""
    migrate_legacy_enabled()
    with _index_lock.metadata_lock("config"):
        file_cfg = _read_file()
        projects = file_cfg.setdefault("projects", {})
        if not isinstance(projects, dict):
            projects = {}
            file_cfg["projects"] = projects
        projects[_canonical_root(root)] = bool(value)
        _write_file(file_cfg)
    return load_config()


def save_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge server/embedding overrides into config.json."""
    with _index_lock.metadata_lock("config"):
        file_cfg = _read_file()
        # Backward-compatible import only. New callers use set_enabled(root, value).
        if "enabled" in overrides and isinstance(overrides.get("enabled"), bool):
            file_cfg["enabled"] = overrides["enabled"]
        for section in ("qdrant", "embedding"):
            osec = overrides.get(section)
            if not isinstance(osec, dict):
                continue
            cur = file_cfg.setdefault(section, {})
            for key, value in osec.items():
                if value is None:
                    continue
                if key in ("port", "vector_dim"):
                    try:
                        cur[key] = int(value)
                    except (TypeError, ValueError):
                        raise ValueError(f"{section}.{key} must be an integer, got {value!r}")
                else:
                    cur[key] = value
        _write_file(file_cfg)
    return load_config()


def qdrant_signature() -> tuple[str, int]:
    c = load_config()
    return str(c["qdrant"]["host"]), int(c["qdrant"]["port"])


def embedding_signature() -> tuple[str, str, str, int]:
    c = load_config()
    e = c["embedding"]
    return str(e["base_url"]), str(e["model"]), str(e["api_key"]), int(e["vector_dim"])


def describe() -> str:
    """Human-readable, API-key-redacted summary of resolved config."""
    c = load_config()
    q, e = c["qdrant"], c["embedding"]
    key = str(e["api_key"])
    redacted = (key[:4] + "***" + key[-2:]) if len(key) > 8 else ("set" if key else "EMPTY")
    enabled_count = sum(1 for value in c.get("projects", {}).values() if value)
    return (
        f"Auto-index default: off (enabled projects: {enabled_count})\n"
        f"Qdrant:    {q['host']}:{q['port']}\n"
        f"Embedding: {e['base_url']}  (model={e['model']}, dim={e['vector_dim']}, key={redacted})"
    )
