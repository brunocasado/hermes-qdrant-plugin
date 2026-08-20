"""hermes-qdrant-plugin — runtime-editable server & embedding config.

Lets you point the plugin at a different Qdrant server and/or a different
embedding endpoint without touching code. Three ways to set values (highest
precedence first):

  1. Environment variables (per-process override):
       QDRANT_HOST, QDRANT_PORT,
       EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_VECTOR_DIM
  2. config.json (persisted; edited via `qidx config set` or the
     qdrant_set_server tool) at  data/config.json
  3. Built-in defaults (generic placeholders — override for your setup)

config.json shape (all keys optional — missing keys fall back to defaults):

  {
    "enabled":   true,
    "qdrant":    {"host": "localhost", "port": 6333},
    "embedding": {"base_url": "http://localhost:8080/v1",
                  "model": "embeddings", "api_key": "EMPTY",
                  "vector_dim": 768}
  }

``enabled`` is the master switch for automatic indexing: when false the
post-tool auto-reindex hook and the desktop pill's automatic /refresh are
both off (manual "Index now" clicks still work — the user asked for it
explicitly). The desktop settings popover toggles it.

This module is intentionally dependency-free (no `core`, no `qdrant_client`)
so it can be imported from the CLI, the tool handler, and the dashboard REST
door without pulling in the heavy backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# Dual-mode import (plugin package vs plugin-dir-on-sys.path)
try:
    from . import paths as _paths
except ImportError:
    import paths as _paths

# Data dir — the sanctioned per-plugin state root (see paths.py). Resolved
# lazily so a standalone CLI (no Hermes process) still works, and migrated
# from the old install-tree location on first access.
DATA_DIR = _paths.ensure_data_dir()
CONFIG_PATH = DATA_DIR / "config.json"

# Built-in defaults — generic placeholders; override via config.json or env.
DEFAULTS: dict[str, dict[str, Any]] = {
    "qdrant": {"host": "localhost", "port": 6333},
    "embedding": {
        "base_url": "http://localhost:8080/v1",
        "model": "embeddings",
        "api_key": "EMPTY",
        "vector_dim": 768,
    },
}
# Master switch default: the feature is on unless the user turns it off.
DEFAULT_ENABLED = True

# env var name -> (section, key)
_ENV_MAP = {
    "QDRANT_HOST": ("qdrant", "host"),
    "QDRANT_PORT": ("qdrant", "port"),
    "EMBEDDING_BASE_URL": ("embedding", "base_url"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "EMBEDDING_API_KEY": ("embedding", "api_key"),
    "EMBEDDING_VECTOR_DIM": ("embedding", "vector_dim"),
}


def _read_file() -> dict[str, dict[str, Any]]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_config() -> dict[str, dict[str, Any]]:
    """Return the fully-resolved config (defaults <- file <- env).

    Always returns the complete nested shape with every key present, so
    callers never have to guard against missing fields. Top-level keys:
    ``enabled`` (bool), ``qdrant``, ``embedding``.
    """
    file_cfg = _read_file()
    resolved: dict[str, Any] = {
        "enabled": DEFAULT_ENABLED,
        "qdrant": dict(DEFAULTS["qdrant"]),
        "embedding": dict(DEFAULTS["embedding"]),
    }
    # file overrides defaults
    if isinstance(file_cfg.get("enabled"), bool):
        resolved["enabled"] = file_cfg["enabled"]
    for section in ("qdrant", "embedding"):
        fsec = file_cfg.get(section)
        if isinstance(fsec, dict):
            for k in resolved[section]:
                if k in fsec and fsec[k] is not None:
                    resolved[section][k] = fsec[k]
    # env overrides file (and defaults)
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


def is_enabled() -> bool:
    """Master switch — False when the user disabled automatic indexing."""
    return bool(load_config().get("enabled", True))


def save_config(overrides: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge `overrides` into config.json and return the resolved config.

    `overrides` uses the same nested shape as config.json; only the keys you
    pass are written, everything else in the file is preserved. A top-level
    boolean ``"enabled"`` key sets the master switch.
    """
    file_cfg = _read_file()
    if "enabled" in overrides and isinstance(overrides.get("enabled"), bool):
        file_cfg["enabled"] = overrides["enabled"]
    for section in ("qdrant", "embedding"):
        osec = overrides.get(section)
        if not isinstance(osec, dict):
            continue
        cur = file_cfg.setdefault(section, {})
        for k, v in osec.items():
            if v is None:
                continue
            if k in ("port", "vector_dim"):
                try:
                    cur[k] = int(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{section}.{k} must be an integer, got {v!r}")
            else:
                cur[k] = v
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(file_cfg, indent=2) + "\n")
    return load_config()


def qdrant_signature() -> tuple[str, int]:
    """(host, port) tuple used as a cache key so the client is rebuilt when
    the server changes."""
    c = load_config()
    return (str(c["qdrant"]["host"]), int(c["qdrant"]["port"]))


def embedding_signature() -> tuple[str, str, str, int]:
    c = load_config()
    e = c["embedding"]
    return (str(e["base_url"]), str(e["model"]), str(e["api_key"]), int(e["vector_dim"]))


def describe() -> str:
    """Human-readable, API-key-redacted summary of the resolved config."""
    c = load_config()
    q, e = c["qdrant"], c["embedding"]
    key = str(e["api_key"])
    redacted = (key[:4] + "***" + key[-2:]) if len(key) > 8 else ("set" if key else "EMPTY")
    return (
        f"Enabled:   {'on' if c.get('enabled', True) else 'off (automatic indexing disabled)'}\n"
        f"Qdrant:    {q['host']}:{q['port']}\n"
        f"Embedding: {e['base_url']}  (model={e['model']}, dim={e['vector_dim']}, key={redacted})"
    )
