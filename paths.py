"""Single source of truth for this plugin's durable state location.

Per the Hermes plugin convention (Developer Guide → "Store durable state"),
state lives in the per-plugin data root ``<hermes home>/plugin-data/<name>/``
— NOT inside the install tree (``~/.hermes/plugins/<name>/``), which
``hermes plugins update`` git-pulls and ``hermes plugins remove`` deletes.
Parking state in the install tree means user data dies with the code.

One-time migration: older versions of this plugin wrote state into the
install tree (``<install dir>/data/``). On first access, any files found
there are COPIED into the sanctioned data root (copy, never move — the
originals are gitignored and harmless until the next update/remove wipes
the install tree).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-qdrant-plugin"

# Where older versions parked state (inside the install tree), in
# chronological order. Kept as candidates so a rename mid-life still finds it.
_LEGACY_DATA_DIRS = (
    Path.home() / ".hermes" / "plugins" / "qdrant-index" / "data",
    Path.home() / ".hermes" / "plugins" / "hermes-qdrant-plugin" / "data",
)


def data_dir() -> Path:
    """Return (and create) this plugin's durable state directory.

    Uses ``plugins.plugin_storage.plugin_data_dir`` when importable (inside a
    Hermes process — profile-aware via ``get_hermes_home``); falls back to the
    same layout under the default hermes home for standalone CLI use.
    """
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir(PLUGIN_NAME)
    except Exception:
        d = Path.home() / ".hermes" / "plugin-data" / PLUGIN_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d


def _migrate_legacy_state(target: Path) -> None:
    """Copy state files from the old install-tree location into *target*."""
    for src_dir in _LEGACY_DATA_DIRS:
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if not f.is_file():
                continue
            dest = target / f.name
            if dest.exists():
                continue
            try:
                shutil.copyfile(f, dest)
                logger.info("%s: migrated state %s -> %s", PLUGIN_NAME, f, dest)
            except OSError as e:
                logger.warning("%s: failed to migrate %s: %s", PLUGIN_NAME, f, e)


_migrated = False


def ensure_data_dir() -> Path:
    """``data_dir()`` plus a one-time legacy migration (idempotent per process)."""
    global _migrated
    d = data_dir()
    if not _migrated:
        _migrated = True
        try:
            _migrate_legacy_state(d)
        except Exception as e:  # never break the plugin over a migration hiccup
            logger.warning("%s: state migration failed: %s", PLUGIN_NAME, e)
    return d
