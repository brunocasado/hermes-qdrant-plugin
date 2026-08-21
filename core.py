"""Qdrant Semantic Search core logic.

Ported from ~/projects/qdrant-mcp-server/server.py (lines 31-329) for the
Hermes Qdrant plugin (hermes-qdrant-plugin). Uses an OpenAI-compatible embedding API and
connects to a Qdrant server — both are runtime-editable (no code change) via
`qidx config set ...` or the qdrant_set_server tool; see qconfig.py.
Defaults: Qdrant at localhost:6333, embeddings at http://localhost:8080/v1
(model: embeddings, 768-dim) — override via config or the qdrant_set_server tool.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
# The Qdrant server and the embedding endpoint are runtime-editable — no code
# change needed. Precedence: env var > data/config.json > built-in defaults.
# Edit with `qidx config set <key> <value>` or the qdrant_set_server tool.
# See qconfig.py for the full shape and the env-var names.
CHUNK_SIZE = 350  # estimated tokens per chunk (metadata must also fit 512)
CHUNK_OVERLAP = 60  # estimated token overlap
EMBEDDING_MAX_CHARS = 480  # live-probed safe ceiling for the local 512-token model

# Dual-mode import (plugin package vs plugin-dir-on-sys.path):
try:
    from . import registry as _registry
    from . import qconfig as _qconfig
    from . import index_lock as _index_lock
    from .chunking import estimate_tokens, token_chunks_file, token_chunks_from_text
    from .structural import structural_chunks
except ImportError:
    import registry as _registry
    import qconfig as _qconfig
    import index_lock as _index_lock
    from chunking import estimate_tokens, token_chunks_file, token_chunks_from_text
    from structural import structural_chunks

IndexAlreadyRunning = _index_lock.IndexAlreadyRunning
INDEX_SCHEMA_VERSION = _registry.INDEX_SCHEMA_VERSION

# --- Data dir ---
# Sanctioned per-plugin state root (see paths.py) — survives plugin
# update/remove and follows the active profile.
DATA_DIR = _qconfig.DATA_DIR
HASH_CACHE_PATH = DATA_DIR / "hash-cache.json"
LEGACY_CACHE_PATH = Path.home() / ".hermes" / "qdrant-hash-cache.json"

# --- Globals ---
_client: Optional[object] = None
_client_sig: Optional[tuple] = None  # (host, port) the client was built for
_http_client: Optional[httpx.AsyncClient] = None
_http_sig: Optional[tuple] = None  # (base_url, api_key) the client was built for
_hash_cache: dict[str, dict[str, int]] = {}  # collection -> {filepath: chunk_count}


def get_client():
    """Return a QdrantClient for the currently-configured server.

    The client is rebuilt automatically if the (host, port) changed since the
    last call — so `qidx config set qdrant.host ...` takes effect on the next
    operation without a restart.
    """
    global _client, _client_sig
    from qdrant_client import QdrantClient
    sig = _qconfig.qdrant_signature()
    if _client is None or _client_sig != sig:
        host, port = sig
        _client = QdrantClient(host=host, port=port)
        _client_sig = sig
    return _client


def get_http_client() -> httpx.AsyncClient:
    """Return the embedding httpx client, rebuilt if base_url/api_key changed."""
    global _http_client, _http_sig
    e = _qconfig.load_config()["embedding"]
    sig = (e["base_url"], e["api_key"])
    if _http_client is None or _http_client.is_closed or _http_sig != sig:
        if _http_client is not None:
            try:
                _http_client.close()
            except Exception:
                pass
        _http_client = httpx.AsyncClient(
            base_url=e["base_url"],
            timeout=60.0,
            headers={"Authorization": f"Bearer {e['api_key']}"},
            http2=False,
        )
        _http_sig = sig
    return _http_client


async def _embed_batch(batch_texts: list[str], emb: dict, semaphore: asyncio.Semaphore) -> list[list[float]]:
    """Embed one batch (≤200 chunks) with retry.

    Shared by the bulk driver (get_embeddings) and the indexing pipeline so
    both use the same endpoint handling, timeout, and retry policy.
    """
    emb_url, emb_model, emb_key = emb["base_url"], emb["model"], emb["api_key"]
    async with semaphore:
        for attempt in range(3):
            try:
                # Create fresh client each attempt to avoid stale connection pool
                async with httpx.AsyncClient(
                    base_url=emb_url,
                    timeout=60.0,
                    headers={"Authorization": f"Bearer {emb_key}"},
                ) as client:
                    resp = await client.post(
                        "/embeddings",
                        json={"model": emb_model, "input": batch_texts, "encoding_format": "float"},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
                logger.warning(f"Request failed: {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"Request error: {e}")
            await asyncio.sleep(1 * (attempt + 1))
        raise Exception(f"Failed after 3 attempts for batch of {len(batch_texts)} chunks")


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI-compatible API.

    Batches requests (200 chunks per batch) and processes batches in parallel.
    Uses 3 concurrent workers for maximum throughput.
    """
    batch_size = 200  # API supports up to 200 chunks per request
    emb = _qconfig.load_config()["embedding"]
    semaphore = asyncio.Semaphore(3)  # 3 concurrent requests to avoid rate limiting

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # Execute all batches in parallel
    results = await asyncio.gather(*[_embed_batch(b, emb, semaphore) for b in batches])

    all_embeddings = []
    for result in results:
        all_embeddings.extend(result)

    return all_embeddings


# --- File Discovery ---
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", ".worktrees", ".kilo", "dist", "build", ".next", ".nuxt", ".output", ".cache", "vendor", "target", "out", "bin", "obj", ".hermes", "benchmarks", "backups"}
SKIP_EXTS = {".lock", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map", ".pt", ".pth", ".bin", ".so", ".dylib", ".dll", ".exe", ".pyc", ".pyo", ".DS_Store"}
INDEXABLE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".scala", ".rb", ".php",
    ".cs", ".swift", ".kt", ".kts",
    ".css", ".scss", ".sass", ".less", ".styl",
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".tex", ".adoc",
    ".sql", ".graphql", ".proto", ".thrift",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".env", ".dockerfile",
    ".ipynb",
    ".lua", ".r", ".jl", ".dart",
}


MAX_DISCOVER_FILES = 20000  # hard cap: protects against walking huge roots (e.g. $HOME)


def discover_files(directory: str, max_files: int = MAX_DISCOVER_FILES) -> list[str]:
    """Discover indexable files in a directory (bounded by max_files)."""
    dir_path = Path(directory).resolve()
    # The plugin's own state (hash cache / registry / config) is never index
    # content. Without this exclusion, indexing the plugin's own root is
    # self-referential: every index run rewrites hash-cache.json and
    # registry.json (new last_indexed timestamp + new hashes), so the next
    # status check always reports them changed and the index never settles.
    state_files: set[str] = set()
    if DATA_DIR.exists():
        state_files = {str(p) for p in DATA_DIR.iterdir() if p.is_file()}
    # Backward-compatible install-tree state from pre-plugin-data versions.
    # Exclude this plugin's exact legacy directory even when indexing a parent
    # such as ~/.hermes/plugins; do not globally skip arbitrary project data/.
    legacy_data = Path(__file__).resolve().parent / "data"
    if legacy_data.exists():
        state_files.update(str(p) for p in legacy_data.iterdir() if p.is_file())
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in filenames:
            full = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in INDEXABLE_EXTS and full not in state_files:
                files.append(full)
                if len(files) >= max_files:
                    return sorted(files[:max_files])
    return sorted(files)


# --- Chunking ---
def chunk_file(filepath: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Prefer semantic units; fall back to token-budgeted overlapping chunks."""
    structured = structural_chunks(filepath, chunk_size, chunk_overlap)
    return structured or token_chunks_file(filepath, chunk_size, chunk_overlap)


# --- Collection naming ---
def derive_collection_name(directory: str) -> str:
    """Derive a meaningful collection name from the project folder name.

    Uses the last path component (the project/folder name), slugified to a
    safe identifier: lowercase, non-alphanumeric runs collapsed to a single
    hyphen. Falls back to ``ws-<md5[:16]>`` only if the folder name is empty
    (e.g. the filesystem root) or produces no usable slug.
    """
    import re

    folder = Path(directory).expanduser().resolve().name
    slug = re.sub(r"[^a-z0-9]+", "-", folder.lower()).strip("-")
    if slug:
        return slug
    return "ws-" + hashlib.md5(str(directory).encode()).hexdigest()[:16]


class CollectionNameConflict(ValueError):
    """An explicit collection belongs to a different project root."""


def resolve_collection_name(root: str, requested: str | None = None) -> str:
    """Resolve a stable collection without mixing same-basename projects."""
    root = str(Path(root).expanduser().resolve())
    existing = _registry.collection_for_root(root)
    if requested:
        owner = _registry.root_for_collection(requested)
        if owner and str(Path(owner).resolve()) != root:
            raise CollectionNameConflict(
                f"Collection '{requested}' belongs to {owner}, not {root}"
            )
        return requested
    if existing:
        return existing
    base = derive_collection_name(root)
    owner = _registry.root_for_collection(base)
    if not owner or str(Path(owner).resolve()) == root:
        return base
    suffix = hashlib.sha256(root.encode()).hexdigest()[:8]
    return f"{base}-{suffix}"


# --- Indexing ---
def load_hash_cache(collection_name: str) -> dict[str, dict[str, int]]:
    """Load the SHA-256 hash cache for a collection."""
    cache_path = HASH_CACHE_PATH
    if not cache_path.exists() and LEGACY_CACHE_PATH.exists():
        # One-time migration: copy legacy cache to the new data dir, never delete legacy.
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEGACY_CACHE_PATH, cache_path)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
            return cache.get(collection_name, {})
    return {}


def save_hash_cache(collection_name: str, cache: dict[str, dict[str, int]]):
    """Atomically merge one collection checkpoint into the shared cache."""
    cache_path = HASH_CACHE_PATH
    with _index_lock.metadata_lock("hash-cache"):
        full_cache = {}
        if cache_path.exists():
            try:
                full_cache = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                full_cache = {}
        full_cache[collection_name] = cache
        _index_lock.atomic_write_json(cache_path, full_cache)


def file_hash(path: str) -> str:
    """Short SHA-256 prefix used by the hash cache."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# --- Live collection check (with a short TTL cache) ---
# /status is polled every 30s (plus event-driven refetches on session switch /
# window focus), so an uncached live check would hit Qdrant with a points/count
# on every refetch — that's the bursty traffic visible in the Docker logs.
# Memoize the result per collection for LIVE_CHECK_TTL_SECS. index_directory()
# invalidates its collection's entry before it starts mutating points and again
# when it finishes, so the pill never shows a pre-rebuild (stale) count.
LIVE_CHECK_TTL_SECS = 30.0
_live_state_cache: dict = {}   # collection -> (monotonic_ts, state, count)
_live_state_lock = threading.Lock()


def live_collection_state(client, collection: str, *, force: bool = False) -> tuple[str, int]:
    """Ask Qdrant the truth about a collection: ('missing'|'empty'|'ok', count).

    One ``points/count`` call answers both questions at once: it returns the
    point count for an existing collection and 404s (raises) for a missing one,
    so no separate get_collection round-trip is needed.

    The local registry/hash cache is a *cache* of Qdrant's state, not the state
    itself — the user can delete a collection directly in Qdrant (or wipe the
    data dir), and no local bookkeeping reflects that. Status and re-indexing
    must verify against the server or they'll happily report FRESH for a
    collection that no longer exists.

    Results are memoized per collection for LIVE_CHECK_TTL_SECS to keep the 30s
    status poll (and its event-driven refetches) off the wire. Pass force=True
    (or call invalidate_live_state) to bypass the cache.
    """
    if not force:
        with _live_state_lock:
            hit = _live_state_cache.get(collection)
            if hit is not None and (time.monotonic() - hit[0]) < LIVE_CHECK_TTL_SECS:
                return hit[1], hit[2]
    try:
        n = client.count(collection).count
    except Exception:
        state, n = "missing", 0
    else:
        state, n = ("empty" if n == 0 else "ok"), n
    with _live_state_lock:
        _live_state_cache[collection] = (time.monotonic(), state, n)
    return state, n


def invalidate_live_state(collection: str | None = None) -> None:
    """Drop memoized live-check results (all collections, or just one)."""
    with _live_state_lock:
        if collection is None:
            _live_state_cache.clear()
        else:
            _live_state_cache.pop(collection, None)


def compute_status(root: str, *, cache: dict, registry: dict, collection: str | None,
                   client=None) -> dict:
    """Compare registry + hash cache against the filesystem for one project root.

    With ``client`` (a QdrantClient) the result is verified against the live
    server: a collection that was deleted out-of-band — or is empty —
    downgrades ``indexed`` to False with ``collection_state`` explaining why,
    so the pill and the status tool never keep claiming FRESH for a phantom
    index. Without a client the check is skipped (backward compat).
    """
    entry = registry.get(collection) if collection else None
    if entry is None or entry.get("root") != str(Path(root).resolve()):
        # No known collection for this root — cheap bounded count, never a full walk
        return {"indexed": False, "collection": None, "root": str(Path(root).resolve()),
                "total": len(discover_files(root, max_files=2000)), "unchanged": 0, "changed": 0,
                "new": 0, "deleted": 0, "stale": False, "last_indexed": None}
    files = discover_files(root)
    unchanged = changed = new = 0
    current_rel_paths = set()
    for fp in files:
        rel = str(Path(fp).relative_to(Path(root).resolve()))
        current_rel_paths.add(rel)
        if rel not in cache:
            new += 1
        elif cache[rel].get("hash") == file_hash(fp):
            unchanged += 1
        else:
            changed += 1
    deleted = len(set(cache) - current_rel_paths)
    st = {"indexed": True, "collection": collection, "root": entry["root"],
          "total": len(files), "unchanged": unchanged, "changed": changed,
          "new": new, "deleted": deleted, "stale": (changed + new + deleted) > 0,
          "last_indexed": entry.get("last_indexed"), "file_count": entry.get("file_count")}
    if client is not None:
        state, n = live_collection_state(client, collection)
        st["collection_state"] = state
        if state != "ok":
            # The local cache says indexed but the server says otherwise —
            # trust the server. The stale unchanged/changed/new counts are
            # meaningless without the points; total still tells the caller
            # what a rebuild would cover.
            st["indexed"] = False
            st["file_count"] = None
            st["stale"] = False
        else:
            st["point_count"] = n
    return st


def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".rs": "rust", ".java": "java", ".kt": "kotlin", ".cs": "csharp",
        ".rb": "ruby", ".php": "php", ".swift": "swift", ".sql": "sql",
        ".graphql": "graphql", ".yaml": "yaml", ".yml": "yaml",
        ".json": "json", ".toml": "toml", ".md": "markdown",
    }.get(ext, ext.lstrip(".") or "text")


def build_point_payload(*, root: str, filepath: str, chunk: str,
                        line_start: int, line_end: int, chunk_index: int,
                        file_hash: str, language: str | None = None,
                        symbols: list[str] | None = None,
                        chunk_type: str = "other") -> dict:
    absolute = str(Path(filepath).resolve())
    rel_path = str(Path(absolute).relative_to(Path(root).resolve()))
    return {
        "file": absolute,  # backward compatibility for current consumers
        "rel_path": rel_path,
        "basename": Path(rel_path).name,
        "language": language or detect_language(absolute),
        "symbols": symbols or [],
        "chunk_type": chunk_type,
        "chunk": chunk,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_index": chunk_index,
        "file_hash": file_hash,
    }


def build_embedding_text(*, project: str, rel_path: str, language: str,
                         symbols: list[str], code: str) -> str:
    """Augment code with compact navigation metadata.

    Paths/symbols are navigation hints, so bound only that metadata. Raw code
    is never truncated here; ``prepare_embedding_chunks`` splits it instead.
    """
    def clip(value: str, limit: int, *, keep_end: bool = False) -> str:
        if len(value) <= limit:
            return value
        if keep_end:
            return "…" + value[-(limit - 1):]
        return value[:limit - 1] + "…"

    project = clip(project, 24)
    rel_path = clip(rel_path, 96, keep_end=True)
    language = clip(language, 16)
    symbol_text = clip(", ".join(symbols), 64)
    return (
        f"project: {project}\n"
        f"path: {rel_path}\n"
        f"language: {language}\n"
        f"symbols: {symbol_text}\n"
        f"code:\n{code}"
    )


def collection_schema_is_current(client, collection_name: str, vector_dim: int) -> bool:
    """Read Qdrant's actual schema; registry metadata may lag after a crash."""
    try:
        params = client.get_collection(collection_name).config.params
        vectors = params.vectors
        sparse = params.sparse_vectors
        return (
            isinstance(vectors, dict)
            and "dense" in vectors
            and int(vectors["dense"].size) == vector_dim
            and isinstance(sparse, dict)
            and "lexical" in sparse
        )
    except Exception:
        return False


def prepare_embedding_chunks(chunks: list[dict], *, filepath: str, project: str,
                             rel_path: str, language: str) -> list[dict]:
    """Re-split enriched inputs that exceed the model-safe hard ceiling."""
    prepared = []
    for chunk in chunks:
        symbols = chunk.get("symbols", [])
        enriched = build_embedding_text(
            project=project, rel_path=rel_path, language=language,
            symbols=symbols, code=chunk["chunk"],
        )
        if len(enriched) <= EMBEDDING_MAX_CHARS:
            prepared.append(dict(chunk))
            continue
        prefix = build_embedding_text(
            project=project, rel_path=rel_path, language=language,
            symbols=symbols, code="",
        )
        available_chars = max(100, EMBEDDING_MAX_CHARS - len(prefix))
        split_tokens = max(50, available_chars // 2)
        prepared.extend(token_chunks_from_text(
            chunk["chunk"], filepath=filepath, chunk_tokens=split_tokens,
            overlap_tokens=min(40, split_tokens - 1),
            base_line=chunk.get("line_start", 1),
            chunk_type=chunk.get("chunk_type", "text"), symbols=symbols,
        ))
    for index, chunk in enumerate(prepared):
        chunk["chunk_index"] = index
    return prepared


def _file_point_id(rel_path: str, chunk_index: int) -> str:
    """Portable deterministic id based on the canonical relative path."""
    return hashlib.md5(f"{rel_path}:{chunk_index}".encode()).hexdigest()[:32]


async def index_directory(
    directory: str,
    collection_name: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reindex: bool = False,
    on_progress: Optional[callable] = None,
) -> str:
    """Single admission boundary for every index writer.

    REST, tools, hooks and CLI all call this function. The cross-process root
    and collection locks therefore cover the complete mutation transaction.
    """
    root = str(Path(directory).expanduser().resolve())
    collection = resolve_collection_name(root, requested=collection_name)
    with _index_lock.index_operation_lock(root, collection):
        return await _index_directory_locked(
            root,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reindex=reindex,
            on_progress=on_progress,
        )


async def _index_directory_locked(
    directory: str,
    collection_name: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reindex: bool = False,
    on_progress: Optional[callable] = None,
) -> str:
    """Index all files in a directory into Qdrant.

    Pipelined: each file's chunks are embedded and upserted as soon as they
    are ready (up to 3 files in flight) instead of embedding everything
    first and inserting afterwards. Every file is checkpointed in the hash
    cache the moment its points land, so an interrupted run (crash, timeout,
    embedding endpoint failure) resumes from the first incomplete file on
    the next call — completed files are never re-embedded.

    ``on_progress(files_done, files_total)`` is invoked after each file is
    checkpointed, so callers can surface live progress.
    """
    from qdrant_client import models
    from qdrant_client.models import VectorParams, Distance, PointStruct

    c = get_client()
    dir_path = Path(directory).resolve()
    vector_dim = int(_qconfig.load_config()["embedding"]["vector_dim"])

    if not dir_path.exists():
        return f"Directory not found: {directory}"

    if collection_name is None:
        collection_name = derive_collection_name(str(dir_path))

    # Collections created before named dense+sparse vectors cannot accept the
    # new point shape. Upgrade registered legacy collections once, under the
    # same root+collection writer lock as every other mutation.
    registered = _registry.load().get(collection_name)
    if (isinstance(registered, dict)
            and registered.get("root") == str(dir_path)
            and int(registered.get("schema_version", 0)) < INDEX_SCHEMA_VERSION):
        if collection_schema_is_current(c, collection_name, vector_dim):
            # A previous rebuild created the new collection but was interrupted
            # before the final registry.record(). Resume from per-file cache.
            _registry.mark_schema_current(collection_name, str(dir_path))
        else:
            reindex = True

    # Ensure collection exists, and learn whether it actually holds points.
    # The hash cache is a *cache of Qdrant's state* — if the user deleted the
    # collection (or it's empty) out-of-band, the cache still claims every
    # file is up to date. Trust the server: a missing/empty collection means
    # the whole index is gone and must be rebuilt, regardless of the cache.
    # We're about to mutate points, so force a fresh server read — a cached
    # count from the last /status would be stale the moment we upsert.
    if reindex:
        try:
            c.delete_collection(collection_name)
        except Exception:
            pass
        c.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=vector_dim, distance=Distance.COSINE)},
            sparse_vectors_config={
                "lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        _registry.mark_schema_current(collection_name, str(dir_path))
        invalidate_live_state(collection_name)
        live_state, live_count = "empty", 0
    else:
        live_state, live_count = live_collection_state(c, collection_name, force=True)
        if live_state == "missing":
            c.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=vector_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            _registry.mark_schema_current(collection_name, str(dir_path))
            invalidate_live_state(collection_name)
            live_state, live_count = "empty", 0

    # Load hash cache
    hash_cache = load_hash_cache(collection_name)

    # Discover files
    files = discover_files(str(dir_path))
    logger.info(f"Discovered {len(files)} files in {directory}")

    current_rel_paths = {
        str(Path(filepath).relative_to(dir_path)) for filepath in files
    }
    deleted_rel_paths = set(hash_cache) - current_rel_paths
    for rel_path in sorted(deleted_rel_paths):
        selector = models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(
                key="rel_path", match=models.MatchValue(value=rel_path)
            )
        ]))
        c.delete(collection_name=collection_name, points_selector=selector, wait=True)
        hash_cache.pop(rel_path, None)
    if deleted_rel_paths:
        save_hash_cache(collection_name, hash_cache)

    # Filter files that need indexing. An empty/missing live collection means
    # the hash cache describes points that no longer exist — treat it exactly
    # like a forced reindex (everything is new), otherwise the run below would
    # find "no new files" and return without uploading a single point.
    force_all = reindex or live_state != "ok"
    new_files = []
    changed_rel_paths = set()
    for filepath in files:
        rel_path = str(Path(filepath).relative_to(dir_path))
        if force_all:
            new_files.append(filepath)
        elif rel_path not in hash_cache:
            new_files.append(filepath)
        else:
            if file_hash(filepath) != hash_cache[rel_path].get("hash", ""):
                new_files.append(filepath)
                changed_rel_paths.add(rel_path)

    if not new_files:
        _registry.record(collection_name, str(dir_path), len(files))
        return f"No new or changed files to index in '{collection_name}' (total points: {c.count(collection_name).count})"

    logger.info(f"Indexing {len(new_files)} files (pipeline, {min(3, len(new_files))} in flight)..."
                + (" [rebuilding — collection was empty/missing]" if live_state != "ok" else ""))

    # Forced reindex / rebuild: the cache can't tell "already re-embedded this
    # run" from "indexed in a previous run" (hashes match either way), so start
    # the checkpoint set fresh — every file must be re-processed, and each
    # completion is recorded as it lands. A crash mid-reindex resumes from the
    # first file not yet in the (new) cache.
    if force_all:
        hash_cache = {}

    emb_cfg = _qconfig.load_config()["embedding"]
    semaphore = asyncio.Semaphore(3)  # 3 files in flight (≈3 concurrent embedding requests)
    files_total = len(new_files)
    files_done = 0
    points_written = 0
    progress_lock = asyncio.Lock()

    async def process_file(filepath: str) -> int:
        nonlocal files_done, points_written
        rel_path = str(Path(filepath).relative_to(dir_path))
        if rel_path in changed_rel_paths:
            selector = models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="rel_path", match=models.MatchValue(value=rel_path)
                )
            ]))
            c.delete(collection_name=collection_name, points_selector=selector, wait=True)
        chunks = chunk_file(str(filepath), chunk_size, chunk_overlap)
        language = detect_language(str(filepath))
        chunks = prepare_embedding_chunks(
            chunks, filepath=str(filepath), project=dir_path.name,
            rel_path=rel_path, language=language,
        )
        if not chunks:
            # Unreadable/empty file — checkpoint it so it isn't retried every run.
            async with progress_lock:
                hash_cache[rel_path] = {"hash": file_hash(filepath)}
                save_hash_cache(collection_name, hash_cache)
                files_done += 1
                if on_progress:
                    on_progress(files_done, files_total)
            return 0

        # Embed this file's chunks in batches of 200 (sequential within the
        # file). Do NOT acquire `semaphore` here: _embed_batch already holds
        # it, and asyncio.Semaphore is not re-entrant — double acquisition
        # deadlocks the 3rd in-flight file.
        texts = [build_embedding_text(
            project=dir_path.name,
            rel_path=rel_path,
            language=language,
            symbols=ch.get("symbols", []),
            code=ch["chunk"],
        ) for ch in chunks]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), 200):
            embeddings.extend(await _embed_batch(texts[i : i + 200], emb_cfg, semaphore))

        # Build points (deterministic ids — idempotent on resume).
        fh = file_hash(filepath)
        points = []
        for i, ch in enumerate(chunks):
            payload = build_point_payload(
                root=str(dir_path), filepath=str(filepath), chunk=ch["chunk"],
                line_start=ch["line_start"], line_end=ch["line_end"],
                chunk_index=ch["chunk_index"], file_hash=fh,
                language=language, symbols=ch.get("symbols", []),
                chunk_type=ch.get("chunk_type", "text"),
            )
            points.append(PointStruct(
                id=_file_point_id(payload["rel_path"], payload["chunk_index"]),
                vector={"dense": embeddings[i], "lexical": sparse_vector(texts[i])},
                payload=payload,
            ))

        # Upsert in batches of 100 — points land in Qdrant as soon as this
        # file's embeddings are ready (the whole point of the pipeline).
        for i in range(0, len(points), 100):
            c.upsert(collection_name, points=points[i : i + 100])

        # Checkpoint: only after the points are in Qdrant. If we die before
        # this line, the next run re-embeds + re-upserts this file (idempotent).
        async with progress_lock:
            hash_cache[rel_path] = {"hash": fh}
            save_hash_cache(collection_name, hash_cache)
            files_done += 1
            points_written += len(points)
            if on_progress:
                on_progress(files_done, files_total)
        return len(points)

    # Bounded concurrency: all file tasks are created, but only 3 hold the
    # semaphore at once. A file that fails all retries aborts the run via
    # gather(); every file that completed before it is already checkpointed,
    # so the next call resumes where this one left off.
    await asyncio.gather(*[process_file(fp) for fp in new_files])

    # Points just changed — drop the memoized live check so the next /status
    # reflects the new count instead of a pre-rebuild value.
    invalidate_live_state(collection_name)
    total_points = c.count(collection_name).count
    _registry.record(collection_name, str(dir_path), len(files))
    return f"Indexed {len(new_files)} files, {points_written} chunks into '{collection_name}' (total: {total_points} points)"


# --- Search ---
@dataclass
class RetrievedHit:
    id: object
    score: float
    payload: dict


async def search_qdrant(
    collection_name: str,
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
) -> list:
    """Route dense/lexical retrieval and combine mixed queries with RRF."""
    client = get_client()
    route = route_query(query)
    dense_hits = []
    lexical_hits = []

    if route != "lexical":
        query_vector = (await get_embeddings([query]))[0]
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        # min_score remains a dense cosine threshold; it is never applied to
        # incomparable sparse or RRF scores.
        dense_hits = [hit for hit in response.points if hit.score >= min_score]

    if route != "semantic":
        response = client.query_points(
            collection_name=collection_name,
            query=sparse_vector(query),
            using="lexical",
            limit=limit,
            with_payload=True,
        )
        lexical_hits = list(response.points)

    if route == "semantic":
        return dense_hits
    if route == "lexical":
        return lexical_hits

    rankings = [[hit.id for hit in dense_hits], [hit.id for hit in lexical_hits]]
    fused_ids = rrf_fuse(rankings)
    by_id = {hit.id: hit for hit in dense_hits + lexical_hits}
    rrf_scores = {}
    for ranking in rankings:
        for rank, point_id in enumerate(ranking, 1):
            rrf_scores[point_id] = rrf_scores.get(point_id, 0.0) + 1.0 / (60 + rank)
    return [RetrievedHit(
        id=point_id,
        score=rrf_scores[point_id],
        payload=dict(by_id[point_id].payload or {}),
    ) for point_id in fused_ids]


# --- Hybrid retrieval + per-file aggregation ---
def _lexical_tokens(text: str) -> list[str]:
    import re
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[A-Za-z0-9]+", expanded.lower())
    return [token for token in tokens if len(token) > 1]


def sparse_vector(text: str):
    """Deterministic hashed sparse vector for identifiers and lexical terms."""
    from collections import Counter
    from qdrant_client.models import SparseVector

    counts = Counter(_lexical_tokens(text))
    pairs = []
    for token, count in counts.items():
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") & 0x7FFFFFFF
        pairs.append((index, float(count)))
    pairs.sort()
    return SparseVector(indices=[p[0] for p in pairs], values=[p[1] for p in pairs])


def route_query(query: str) -> str:
    """Route without an LLM: exact identifier/path, semantic sentence, or mixed."""
    import re
    words = query.split()
    has_identifier = bool(re.search(r"[a-z0-9][A-Z]|[a-z]+_[a-z]|['\"]", query))
    has_path = "/" in query or bool(re.search(r"\.[A-Za-z0-9]{1,6}$", query.strip()))
    if has_path or ((has_identifier or len(words) == 1) and len(words) <= 2):
        return "lexical"
    if has_identifier:
        return "hybrid"
    if len(words) >= 4:
        return "semantic"
    return "hybrid"


def rrf_fuse(rankings: list[list], k: int = 60) -> list:
    """Reciprocal Rank Fusion (RRF) without cross-score normalization."""
    scores = {}
    first_seen = {}
    seen_counter = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            if item not in first_seen:
                first_seen[item] = seen_counter
                seen_counter += 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))


def aggregate_hits_by_file(hits, top_chunks_per_file: int = 1,
                           query: str | None = None) -> list[dict]:
    """Rank files by best hit + corroborating chunks + symbol/path evidence."""
    by_file: dict[str, list] = {}
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        key = payload.get("rel_path") or payload.get("file", "unknown")
        by_file.setdefault(key, []).append((hit, payload))

    query_tokens = set(_lexical_tokens(query or ""))
    summaries = []
    for key, entries in by_file.items():
        entries.sort(key=lambda ep: ep[0].score, reverse=True)
        best_score = float(entries[0][0].score)
        line_starts = [ep[1].get("line_start", 0) for ep in entries]
        line_ends = [ep[1].get("line_end", 0) for ep in entries]
        symbols = []
        for _, payload in entries:
            for symbol in payload.get("symbols", []):
                if symbol not in symbols:
                    symbols.append(symbol)
        symbol_tokens = set(_lexical_tokens(" ".join(symbols)))
        path = entries[0][1].get("rel_path") or key
        path_tokens = set(_lexical_tokens(path))
        multi_bonus = min(0.10, 0.05 * max(0, len(entries) - 1))
        symbol_bonus = 0.08 if query_tokens & symbol_tokens else 0.0
        path_bonus = 0.04 if query_tokens & path_tokens else 0.0
        file_score = best_score + multi_bonus + symbol_bonus + path_bonus
        top = [{
            "chunk": payload.get("chunk", ""),
            "score": float(hit.score),
            "line_start": payload.get("line_start", 0),
            "line_end": payload.get("line_end", 0),
        } for hit, payload in entries[:max(0, top_chunks_per_file)]]
        summaries.append({
            "file": entries[0][1].get("file", key),
            "rel_path": path,
            "chunk_count": len(entries),
            "best_score": best_score,
            "file_score": file_score,
            "symbols": symbols,
            "line_start": min(line_starts) if line_starts else 0,
            "line_end": max(line_ends) if line_ends else 0,
            "best_chunk": entries[0][1].get("chunk", ""),
            "chunks": top,
        })
    summaries.sort(key=lambda summary: summary["file_score"], reverse=True)
    return summaries


def format_file_results(files: list[dict], query: str) -> str:
    """Render compact navigation evidence; callers should read real files next."""
    lines = [f"Best files for: {query}", ""]
    for index, result in enumerate(files, 1):
        lines.append(f"{index}. {result.get('rel_path') or result.get('file')} (score {result['file_score']:.4f})")
        symbols = result.get("symbols") or []
        if symbols:
            lines.append("   symbols: " + ", ".join(symbols[:8]))
        lines.append(f"   best match: lines {result.get('line_start', 0)}-{result.get('line_end', 0)}")
        snippet = " ".join((result.get("best_chunk") or "").split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        if snippet:
            lines.append("   snippet: " + snippet)
        lines.append("")
    return "\n".join(lines).rstrip()
