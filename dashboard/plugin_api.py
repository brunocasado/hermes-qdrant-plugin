"""REST door for the desktop statusbar pill. Mounted at /api/plugins/qdrant-index/.

The web server imports this module at startup (see hermes_cli.web_server.py
``_mount_plugin_api_routes``) and mounts the module-level ``router`` under
``/api/plugins/qdrant-index/``. The pill calls:

    GET  /status?root=<abs path>
    POST /reindex?root=<abs path>

Security: this module must NOT import ``core`` / ``registry`` (and therefore
must never pull in ``qdrant_client``) at import time. Those imports happen
lazily inside the route handlers via ``_core()`` so that a failed/absent
backend dependency can't break the web server's startup import of the router.

Routes are plain ``def`` (not ``async def``): FastAPI runs sync routes in a
threadpool, so the blocking index call can't stall the event loop. Same
pattern the Kanban dashboard API uses.
"""
import asyncio
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Last reindex operation per root, exposed via /status so the desktop pill
# can show the result of a click even when the index was already fresh
# (an incremental no-op otherwise looks like "nothing happened").
_OPS = {}
_OPS_LOCK = threading.Lock()


def _core():
    """Lazily import the plugin's ``core`` and ``registry`` modules.

    The plugin root (parent of this ``dashboard/`` dir) is prepended to
    ``sys.path`` so ``import core`` / ``import registry`` resolve to the
    plugin's own files rather than any same-named module on the path.
    """
    import importlib
    import sys

    base = Path(__file__).resolve().parent.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import core
    import registry

    return core, registry


def _is_indexable(root: str) -> tuple[bool, str]:
    """Return (ok, reason). ``ok`` is False when ``root`` is the home
    directory or the filesystem root — indexing those would sweep in
    every project and dotfile on the machine.

    Sessions in the "Home" sidebar bucket get a fallback cwd of ``~/``,
    which would otherwise trigger a full-home index.
    """
    p = Path(root).resolve()
    home = Path.home().resolve()
    if p == home:
        return False, "home"
    if str(p) == "/":
        return False, "root"
    return True, ""


@router.get("/status")
def get_status(root: str = ""):
    """Return the index status for ``root`` (defaults to the server cwd)."""
    root = str(Path(root).expanduser().resolve()) if root else str(Path.cwd())
    ok, reason = _is_indexable(root)
    if not ok:
        return {"indexable": False, "reason": reason, "indexed": False, "stale": False,
                "file_count": 0, "changed": 0, "new": 0, "total": 0, "reindexing": False,
                "enabled": _qconfig().is_enabled()}
    core, registry = _core()
    collection = registry.collection_for_root(root)
    cache = core.load_hash_cache(collection) if collection else {}
    # Live-check Qdrant: a collection deleted out-of-band must downgrade the
    # status (the local registry/hash cache can't know about it).
    try:
        client = core.get_client()
    except Exception:
        client = None  # server unreachable — fall back to the local view
    st = core.compute_status(
        root, cache=cache, registry=registry.load(), collection=collection,
        client=client
    )
    with _OPS_LOCK:
        op = _OPS.get(root)
    if op:
        st["last_op"] = dict(op)
        st["reindexing"] = op["status"] == "running"
    else:
        st["reindexing"] = False
    # Master switch — the desktop pill shows a "disabled" state (and skips
    # automatic /refresh) when the user has turned indexing off.
    st["enabled"] = _qconfig().is_enabled()
    return st


@router.post("/reindex")
def post_reindex(root: str = "", collection: str = ""):
    """Start a background reindex of ``root`` and return immediately.

    Fire-and-forget: the index runs in a daemon thread so the HTTP call
    returns without blocking on the (slow) embedding work.
    """
    core, registry = _core()
    root = str(Path(root).expanduser().resolve()) if root else str(Path.cwd())
    ok, reason = _is_indexable(root)
    if not ok:
        return {"started": False, "reason": reason}
    # Known project -> existing collection; unknown -> derive from the
    # folder name so the button can index a brand-new project, not just
    # reindex a registered one.
    collection = collection or registry.collection_for_root(root) or core.derive_collection_name(root)

    with _OPS_LOCK:
        _OPS[root] = {"status": "running", "collection": collection, "at": time.time()}

    def _bg():
        # Live progress into the op record: /status (polled by the pill every
        # 3s) surfaces files_done/files_total while the pipeline runs.
        def _progress(done, total):
            with _OPS_LOCK:
                op = _OPS.get(root)
                if op and op["status"] == "running":
                    op["files_done"] = done
                    op["files_total"] = total
        try:
            # reindex=True: an explicit click always forces a full reindex,
            # even when the hash cache says everything is up to date.
            message = asyncio.run(
                core.index_directory(root, collection_name=collection, reindex=True,
                                     on_progress=_progress)
            )
        except Exception as exc:
            message = f"reindex failed: {exc}"
        with _OPS_LOCK:
            _OPS[root] = {"status": "done", "collection": collection,
                          "message": message, "at": time.time()}

    threading.Thread(target=_bg, daemon=True, name="qdrant-reindex-ui").start()
    return {"started": True, "collection": collection}


@router.post("/refresh")
def post_refresh(root: str = ""):
    """Bring the focused project's index up to date (incremental, no force).

    Called by the desktop pill when the focused project changes: the index
    may be stale (edited in another session) or not exist at all, so the
    pill brings the new project up to date automatically. Unknown roots are
    indexed too (collection derived from the folder name, like /reindex) so
    a never-indexed project shows its own state — never the previous
    project's. Only changed/new files are re-embedded, so switching back
    and forth is cheap."""
    core, registry = _core()
    root = str(Path(root).expanduser().resolve()) if root else str(Path.cwd())
    # Automatic refresh is the master switch's domain: when the user turned
    # indexing off, don't index on focus-change. (Explicit /reindex still works.)
    if not _qconfig().is_enabled():
        return {"started": False, "reason": "disabled", "enabled": False}
    ok, reason = _is_indexable(root)
    if not ok:
        return {"started": False, "reason": reason}
    collection = registry.collection_for_root(root) or core.derive_collection_name(root)

    with _OPS_LOCK:
        op = _OPS.get(root)
        if op and op["status"] == "running":
            return {"started": False, "reason": "already-running", "collection": collection}
        _OPS[root] = {"status": "running", "collection": collection, "at": time.time()}

    def _bg():
        try:
            # reindex=False: incremental — only changed/new files are
            # re-embedded. First-ever index of a project is naturally full.
            message = asyncio.run(core.index_directory(root, collection_name=collection))
        except Exception as exc:
            message = f"refresh failed: {exc}"
        with _OPS_LOCK:
            _OPS[root] = {"status": "done", "collection": collection,
                          "message": message, "at": time.time()}

    threading.Thread(target=_bg, daemon=True, name="qdrant-refresh-ui").start()
    return {"started": True, "collection": collection}


# --- Config: read/edit the Qdrant server & embedding endpoint from the UI ---
# These import only ``qconfig`` (dependency-free — no qdrant_client), so they
# can be mounted without risking the web server's startup import.
def _qconfig():
    import sys

    base = Path(__file__).resolve().parent.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import qconfig
    return qconfig


_VALID_KEYS = {
    "qdrant.host": ("qdrant", "host"),
    "qdrant.port": ("qdrant", "port"),
    "embedding.base_url": ("embedding", "base_url"),
    "embedding.model": ("embedding", "model"),
    "embedding.api_key": ("embedding", "api_key"),
    "embedding.vector_dim": ("embedding", "vector_dim"),
}
_INT_KEYS = {"qdrant.port", "embedding.vector_dim"}


@router.get("/config")
def get_config():
    """Return the resolved server & embedding config (api_key redacted)."""
    qc = _qconfig()
    c = qc.load_config()
    key = str(c["embedding"]["api_key"])
    redacted = (key[:4] + "…" + key[-2:]) if len(key) > 8 else ("set" if key else "EMPTY")
    return {
        "enabled": bool(c.get("enabled", True)),
        "qdrant": dict(c["qdrant"]),
        "embedding": {**c["embedding"], "api_key": redacted},
        "api_key_redacted": len(key) > 8,
        "config_path": str(qc.CONFIG_PATH),
        "config_exists": qc.CONFIG_PATH.exists(),
    }


@router.put("/config")
def put_config(body: dict):
    """Update server & embedding config keys. ``body`` is a flat map of
    ``{"qdrant.host": "...", ...}`` or nested ``{"qdrant": {...}}``.
    A top-level boolean ``"enabled"`` toggles the master switch (works
    standalone — no other keys required)."""
    qc = _qconfig()
    # Accept both flat dotted and nested shapes.
    overrides: dict = {}
    if "enabled" in body and isinstance(body.get("enabled"), bool):
        overrides["enabled"] = body["enabled"]
    if "qdrant" in body and isinstance(body["qdrant"], dict):
        overrides["qdrant"] = dict(body["qdrant"])
    if "embedding" in body and isinstance(body["embedding"], dict):
        overrides["embedding"] = dict(body["embedding"])
    for k, v in body.items():
        if k in _VALID_KEYS:
            section, key = _VALID_KEYS[k]
            if v is not None:
                overrides.setdefault(section, {})[key] = v
    # Drop empty sections so we don't write nothing.
    # (Keep ``enabled`` even when False — a falsy value is a real change.)
    overrides = {s: v for s, v in overrides.items() if s == "enabled" or v}
    if not overrides:
        raise HTTPException(status_code=400, detail="No valid config keys provided")
    try:
        resolved = qc.save_config(overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    key = str(resolved["embedding"]["api_key"])
    redacted = (key[:4] + "…" + key[-2:]) if len(key) > 8 else ("set" if key else "EMPTY")
    return {
        "enabled": bool(resolved.get("enabled", True)),
        "qdrant": dict(resolved["qdrant"]),
        "embedding": {**resolved["embedding"], "api_key": redacted},
    }


@router.get("/config/test")
def test_config():
    """Probe the configured Qdrant server and embedding endpoint.

    Returns per-leg reachability so the UI can confirm a retargeted server
    actually connects before the user relies on it. Uses the *current*
    resolved config (env > file > defaults).
    """
    import httpx

    qc = _qconfig()
    c = qc.load_config()
    result = {"qdrant": {"ok": False, "error": None, "collections": None},
              "embedding": {"ok": False, "error": None, "dim": None}}

    # Leg 1: Qdrant — list collections.
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=c["qdrant"]["host"], port=c["qdrant"]["port"], timeout=5)
        cols = client.get_collections()
        result["qdrant"]["ok"] = True
        result["qdrant"]["collections"] = len(cols.collections)
    except Exception as exc:  # noqa: BLE001 — report any failure to the UI
        result["qdrant"]["error"] = str(exc)[:200]

    # Leg 2: embedding — tiny 1-token request.
    e = c["embedding"]
    try:
        with httpx.Client(timeout=8) as hc:
            resp = hc.post(
                e["base_url"].rstrip("/") + "/embeddings",
                json={"model": e["model"], "input": "ping", "encoding_format": "float"},
                headers={"Authorization": f"Bearer {e['api_key']}"},
            )
        if resp.status_code == 200:
            data = resp.json()
            vec = data["data"][0]["embedding"]
            result["embedding"]["ok"] = True
            result["embedding"]["dim"] = len(vec)
        else:
            result["embedding"]["error"] = f"HTTP {resp.status_code}: {resp.text[:150]}"
    except Exception as exc:  # noqa: BLE001
        result["embedding"]["error"] = str(exc)[:200]

    return result
