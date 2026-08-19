"""qdrant-index plugin — agent half.

Registers the 5 Qdrant tools (index / search / status / list / delete),
the post_tool_call auto-reindex hook (60s debounce), and the
register_system_prompt_section project-selection awareness line.
"""

import asyncio
import time
import threading
from pathlib import Path

# Module-level state
_last_reindex: dict = {}  # root -> monotonic timestamp of last scheduled reindex
_DEBOUNCE_SECS = 60

# In-flight / last background index ops, keyed by resolved root. Mirrors the
# _OPS registry in dashboard/plugin_api.py so the pill and the tool see the
# same state. {root: {"status": "running"|"done"|"error", "collection": str,
#                     "message": str, "at": float}}
_index_ops: dict = {}
_index_ops_lock = threading.Lock()

# How long a qdrant_index tool call waits inline for the background thread to
# finish before returning "still running". Kept well under the gateway's 420s
# RPC timeout so small projects return their real result immediately while big
# monorepos degrade to background + poll instead of timing out.
_INDEX_INLINE_WAIT_SECS = 25
_INDEX_POLL_INTERVAL_SECS = 0.5


def register(ctx):
    try:
        from . import core
        from . import registry
        from . import qconfig
    except ImportError:
        import core
        import registry
        import qconfig

    def _resolve(directory: str) -> str:
        return str(Path(directory).expanduser().resolve()) if directory else _session_cwd()

    def _session_cwd() -> str:
        """Resolve the current project directory for a no-argument call.

        Priority (most authoritative first):
          1. The gateway session DB (sessions.cwd), looked up by the
             HERMES_SESSION_ID env var. When a project is open in the window
             the desktop pins it per session, and this is the authoritative
             "current project" — it survives the process-level cwd being home.
          2. The agent process's real cwd (os.getcwd), when it is a real
             project (not the home dir).
          3. resolve_agent_cwd() (Hermes single source of truth), when it
             resolves to a real project (not the home dir).
          4. The process cwd as a last resort.

        Why not just resolve_agent_cwd()? Its TERMINAL_CWD env bridge
        (from the generic terminal.cwd config) points at the gateway launch
        dir (home) and shadows the per-session pin in tool handlers, because
        the _SESSION_CWD contextvar set in the gateway process does not
        propagate into the tool handler's execution context. The session DB
        is the durable, process-independent record of the window's project.
        """
        home = Path.home()

        # 1. Gateway session DB — authoritative per-session project pin.
        try:
            import os as _os
            import sqlite3
            session_id = _os.environ.get("HERMES_SESSION_ID", "").strip()
            if session_id:
                from hermes_constants import get_hermes_home_override
                hermes_home = Path(get_hermes_home_override() or Path.home() / ".hermes")
                db_path = hermes_home / "state.db"
                if db_path.exists():
                    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
                    try:
                        row = con.execute(
                            "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
                        ).fetchone()
                    finally:
                        con.close()
                    if row and row[0]:
                        p = Path(row[0]).expanduser()
                        if p.is_dir() and p.resolve() != home.resolve():
                            return str(p.resolve())
        except Exception:
            pass

        # 2. Process cwd, if it is a real project (not home).
        proc_cwd = Path.cwd()
        try:
            if proc_cwd.resolve() != home.resolve():
                return str(proc_cwd.resolve())
        except Exception:
            pass

        # 3. Hermes resolver, if it yields a real project (not home).
        try:
            from agent.runtime_cwd import resolve_agent_cwd
            resolved = resolve_agent_cwd()
            if resolved.resolve() != home.resolve():
                return str(resolved)
        except Exception:
            pass

        # 4. Last resort.
        return str(proc_cwd)

    # --- qdrant_index (background-thread, like the REST pill) ---
    # The index runs in a daemon thread so the tool never blocks on the (slow)
    # embedding work. Small projects finish inside the inline wait and return
    # their real result; big roots return "started in background" well before
    # the gateway's RPC timeout, and the agent polls qdrant_status.
    def _start_index(directory, collection_name, chunk_size, chunk_overlap, reindex):
        root = _resolve(directory)
        collection = (collection_name
                      or registry.collection_for_root(root)
                      or core.derive_collection_name(root))
        with _index_ops_lock:
            op = _index_ops.get(root)
            if op and op["status"] == "running":
                return root, collection, True
            _index_ops[root] = {"status": "running", "collection": collection,
                                "at": time.time()}

        def _bg():
            # Live progress: the pipeline calls on_progress after each file
            # is checkpointed; mirror it into the op record so qdrant_status
            # can show "N/M files processed" while the run is in flight.
            def _progress(done, total):
                with _index_ops_lock:
                    op = _index_ops.get(root)
                    if op and op["status"] == "running":
                        op["files_done"] = done
                        op["files_total"] = total
            try:
                message = asyncio.run(core.index_directory(
                    root,
                    collection_name=collection,
                    chunk_size=chunk_size if chunk_size else core.CHUNK_SIZE,
                    chunk_overlap=chunk_overlap if chunk_overlap else core.CHUNK_OVERLAP,
                    reindex=reindex,
                    on_progress=_progress,
                ))
                status = "done"
            except Exception as exc:
                message = f"index failed: {exc}"
                status = "error"
            with _index_ops_lock:
                _index_ops[root] = {"status": status, "collection": collection,
                                    "message": message, "at": time.time()}

        threading.Thread(target=_bg, daemon=True,
                         name=f"qdrant-index-{collection}").start()
        return root, collection, False

    def _wait_for_index(root, timeout):
        def _poll():
            deadline = time.time() + timeout
            while time.time() < deadline:
                with _index_ops_lock:
                    op = _index_ops.get(root)
                if op and op["status"] != "running":
                    return dict(op)
                time.sleep(_INDEX_POLL_INTERVAL_SECS)
            with _index_ops_lock:
                return dict(_index_ops.get(root, {}))
        return asyncio.to_thread(_poll)

    async def qdrant_index(args, **kw):
        root, collection, already = _start_index(
            directory=args.get("directory", ""),
            collection_name=args.get("collection_name"),
            chunk_size=args.get("chunk_size"),
            chunk_overlap=args.get("chunk_overlap"),
            reindex=args.get("reindex", False),
        )
        if already:
            return (f"Index already running for '{collection}' (root: {root}). "
                    f"Call qdrant_status to poll for the result.")
        op = await _wait_for_index(root, _INDEX_INLINE_WAIT_SECS)
        if op.get("status") == "running":
            return (f"Index of '{collection}' started in the background "
                    f"(root: {root}) — it takes longer than the inline wait "
                    f"({_INDEX_INLINE_WAIT_SECS}s). Call qdrant_status to poll "
                    f"until it reports FRESH.")
        if op.get("status") == "error":
            return op.get("message", f"Index of '{collection}' failed.")
        return op.get("message", f"Indexed '{collection}' (root: {root})")

    ctx.register_tool(
        name="qdrant_index",
        toolset="qdrant",
        schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to index (default: session working dir)"},
                "collection_name": {"type": "string", "description": "Collection name (auto: the project folder name, slugified, if omitted)"},
                "chunk_size": {"type": "integer", "description": "Lines per chunk (default 10)"},
                "chunk_overlap": {"type": "integer", "description": "Overlap in lines (default 3)"},
                "reindex": {"type": "boolean", "description": "Force full re-index"},
            },
            "required": [],
        },
        handler=qdrant_index,
        is_async=True,
        description="Index a project's files into Qdrant for semantic search (incremental by SHA-256).",
        emoji="📦",
    )

    # --- qdrant_search ---
    async def _do_search(query, collection_name="", limit=10, min_score=0.0):
        root = _session_cwd()
        collection = collection_name or registry.collection_for_root(root)
        if not collection:
            reg = registry.load()
            if reg:
                return f"No collection for project '{root}'. Available collections: {', '.join(reg.keys())}"
            return f"No collection for project '{root}' and no collections indexed. Run qdrant_index first."
        # Fetch more raw chunks than the requested file count so collapsing
        # to per-file summaries doesn't lose file diversity.
        fetch_limit = max(limit * 3, 15)
        hits = await core.search_qdrant(collection, query, fetch_limit, min_score)
        if not hits:
            return f"No results for: '{query}'"
        summaries = core.aggregate_hits_by_file(hits, top_chunks_per_file=1)
        # Cap at `limit` files.
        summaries = summaries[:limit]
        out = len(summaries)
        header = (f"Search results for: '{query}' (collection: {collection})\n"
                  f"{out} file(s), {len(hits)} chunk(s) matched\n\n" + "=" * 80 + "\n\n")
        output = header
        for i, s in enumerate(summaries, 1):
            output += f"--- {i}. {s['file']} (best score: {s['best_score']:.4f}, " \
                      f"{s['chunk_count']} chunk(s), lines {s['line_start']}-{s['line_end']}) ---\n"
            chunk = s["best_chunk"]
            if len(chunk) > 800:
                chunk = chunk[:800] + "..."
            output += f"Content:\n{chunk}\n" + "-" * 40 + "\n\n"
        output += f"Total: {out} file(s) (from {len(hits)} chunks)"
        return output

    async def qdrant_search(args, **kw):
        return await _do_search(
            query=args.get("query", ""),
            collection_name=args.get("collection_name", ""),
            limit=args.get("limit", 10),
            min_score=args.get("min_score", 0.0),
        )

    ctx.register_tool(
        name="qdrant_search",
        toolset="qdrant",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "collection_name": {"type": "string", "description": "Omit to search the current project's collection"},
                "limit": {"type": "integer", "default": 10},
                "min_score": {"type": "number", "default": 0.0},
            },
            "required": ["query"],
        },
        handler=qdrant_search,
        is_async=True,
        description="Semantic search over indexed project files.",
        emoji="🔎",
    )

    # --- qdrant_status ---
    async def _do_status(directory=""):
        def _run():
            root = _resolve(directory) if directory else _session_cwd()
            collection = registry.collection_for_root(root)
            cache = core.load_hash_cache(collection) if collection else {}
            try:
                client = core.get_client()
            except Exception:
                client = None
            st = core.compute_status(root, cache=cache, registry=registry.load(), collection=collection, client=client)
            lines = [f"Collection: {st['collection'] or '(none)'}", f"Root: {st['root']}"]
            if st["indexed"]:
                lines.append(f"Indexed: {st['file_count']} files at {st['last_indexed']}")
                if st.get("point_count") is not None:
                    lines.append(f"  {st['point_count']} points in Qdrant")
                lines.append(f"  ✓ {st['unchanged']} unchanged")
                if st["changed"]:
                    lines.append(f"  ~ {st['changed']} changed")
                if st["new"]:
                    lines.append(f"  + {st['new']} new")
                lines.append("Status: STALE — run qdrant_index to refresh" if st["stale"] else "Status: FRESH")
            else:
                # Distinguish "never indexed" from "deleted out-of-band".
                state = st.get("collection_state")
                if state in ("missing", "empty"):
                    lines.append(f"Not indexed — collection '{st['collection']}' is {state} in Qdrant.")
                    lines.append(f"Run qdrant_index to rebuild it ({st['total']} indexable files found).")
                else:
                    lines.append(f"Not indexed. {st['total']} indexable files found.")
            # Surface an in-flight / recent background index op for this root.
            with _index_ops_lock:
                op = _index_ops.get(root)
            if op:
                if op["status"] == "running":
                    prog = ""
                    if op.get("files_total"):
                        prog = f" — {op.get('files_done', 0)}/{op['files_total']} files processed"
                    lines.append(f"Index in progress: '{op['collection']}'{prog} (started {time.strftime('%H:%M:%S', time.localtime(op['at']))}) — poll again in a moment.")
                elif op["status"] == "error":
                    lines.append(f"Last index FAILED: {op.get('message', '')}")
                else:
                    lines.append(f"Last index: {op.get('message', 'done')}")
            return "\n".join(lines)
        return await asyncio.to_thread(_run)

    async def qdrant_status(args, **kw):
        return await _do_status(directory=args.get("directory", ""))

    ctx.register_tool(
        name="qdrant_status",
        toolset="qdrant",
        schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Project root (default: session working dir)"},
            },
            "required": [],
        },
        handler=qdrant_status,
        is_async=True,
        description="Index health for a project: file counts, staleness, last index time.",
        emoji="📊",
    )

    # --- qdrant_list_collections (port of server.py list branch) ---
    async def _do_list_collections():
        def _run():
            c = core.get_client()
            collections = c.get_collections()
            output = "Collections:\n"
            for coll in collections.collections:
                info = c.get_collection(coll.name)
                output += f"  - {coll.name}: {info.points_count} points\n"
            return output
        return await asyncio.to_thread(_run)

    async def qdrant_list_collections(args, **kw):
        return await _do_list_collections()

    ctx.register_tool(
        name="qdrant_list_collections",
        toolset="qdrant",
        schema={"type": "object", "properties": {}},
        handler=qdrant_list_collections,
        is_async=True,
        description="List all Qdrant collections with their point counts.",
        emoji="📋",
    )

    # --- qdrant_delete_collection (port of server.py delete branch) ---
    async def _do_delete_collection(collection_name):
        def _run():
            c = core.get_client()
            c.delete_collection(collection_name)
            # The live-check TTL cache may still claim this collection exists —
            # drop it so the next /status sees the deletion immediately.
            core.invalidate_live_state(collection_name)
            # Drop the registry entry (and its cache) so the project is treated as unindexed.
            reg = registry.load()
            if collection_name in reg:
                del reg[collection_name]
                registry.save(reg)
            cache = core.load_hash_cache(collection_name)
            if cache:
                all_cache = {}
                import json
                if core.HASH_CACHE_PATH.exists():
                    all_cache = json.loads(core.HASH_CACHE_PATH.read_text())
                all_cache.pop(collection_name, None)
                core.DATA_DIR.mkdir(parents=True, exist_ok=True)
                core.HASH_CACHE_PATH.write_text(json.dumps(all_cache, indent=2))
            return f"Deleted collection '{collection_name}'"
        return await asyncio.to_thread(_run)

    async def qdrant_delete_collection(args, **kw):
        return await _do_delete_collection(args.get("collection_name", ""))

    ctx.register_tool(
        name="qdrant_delete_collection",
        toolset="qdrant",
        schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "description": "The collection to delete"},
            },
            "required": ["collection_name"],
        },
        handler=qdrant_delete_collection,
        is_async=True,
        description="Delete a Qdrant collection. This action cannot be undone.",
        emoji="🗑️",
    )

    # --- qdrant_set_server — point the plugin at a different Qdrant server
    #     and/or embedding endpoint at runtime (no code edit, no restart).
    #     Takes effect on the next operation: get_client() rebuilds the client
    #     when the (host, port) signature changes.
    def _set_server(host, port, base_url, model, api_key, vector_dim):
        overrides = {}
        if host or port:
            q = {}
            if host:
                q["host"] = host
            if port:
                q["port"] = int(port)
            overrides["qdrant"] = q
        if base_url or model or api_key or vector_dim:
            e = {}
            if base_url:
                e["base_url"] = base_url
            if model:
                e["model"] = model
            if api_key:
                e["api_key"] = api_key
            if vector_dim:
                e["vector_dim"] = int(vector_dim)
            overrides["embedding"] = e
        if not overrides:
            return "Nothing to set — pass at least one of host/port/base_url/model/api_key/vector_dim. Current config:\n" + qconfig.describe()
        resolved = qconfig.save_config(overrides)
        return "Saved. Current resolved config:\n" + qconfig.describe() + (
            "\n\nNote: the Qdrant client will reconnect to the new server on the next operation."
            if "qdrant" in overrides else ""
        )

    async def qdrant_set_server(args, **kw):
        return await asyncio.to_thread(
            _set_server,
            args.get("host"),
            args.get("port"),
            args.get("base_url"),
            args.get("model"),
            args.get("api_key"),
            args.get("vector_dim"),
        )

    ctx.register_tool(
        name="qdrant_set_server",
        toolset="qdrant",
        schema={
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Qdrant host (e.g. 'localhost')"},
                "port": {"type": "integer", "description": "Qdrant port (default 6333)"},
                "base_url": {"type": "string", "description": "Embedding API base URL (OpenAI-compatible, e.g. 'http://localhost:8080/v1')"},
                "model": {"type": "string", "description": "Embedding model name (e.g. 'embeddings')"},
                "api_key": {"type": "string", "description": "Embedding API key (use 'EMPTY' if the server needs no auth)"},
                "vector_dim": {"type": "integer", "description": "Embedding vector dimension (must match the model)"},
            },
            "required": [],
        },
        handler=qdrant_set_server,
        is_async=True,
        description="Point the plugin at a different Qdrant server and/or embedding endpoint (runtime, persisted to config.json).",
        emoji="🎛️",
    )

    # --- T6: post_tool_call hook — auto-reindex after edits (60s debounce) ---
    EDIT_TOOLS = {"write_file", "patch"}

    def _on_tool_call(**kwargs):
        name = kwargs.get("tool_name")
        if name not in EDIT_TOOLS:
            return
        # Master switch off: the user disabled automatic indexing — skip
        # silently (manual /qdrant-index index still works).
        if not qconfig.is_enabled():
            return
        args = kwargs.get("args") or {}
        path = args.get("path") or args.get("file_path") or ""
        if not path:
            return
        file_path = Path(path).expanduser().resolve()
        # Find the containing registered project root
        root = None
        collection = None
        for parent in file_path.parents:
            collection = registry.collection_for_root(str(parent))
            if collection:
                root = str(parent)
                break
        if root is None:
            return
        now = time.monotonic()
        if now - _last_reindex.get(root, 0) < _DEBOUNCE_SECS:
            return
        _last_reindex[root] = now

        def _bg():
            try:
                asyncio.run(core.index_directory(root, collection_name=collection))
            except Exception:
                pass  # fire-and-forget; failures are logged by core

        threading.Thread(target=_bg, daemon=True, name="qdrant-reindex").start()

    ctx.register_hook("post_tool_call", _on_tool_call)

    # --- Slash command: /qdrant-index (terminal-free bootstrap/maintenance) ---
    # The tools below are async and already push blocking I/O (Qdrant calls,
    # disk walks) off the event loop via asyncio.to_thread, so awaiting them
    # here never stalls the gateway's command.dispatch (30s RPC timeout).
    _VALID_KEYS = (
        "enabled",
        "qdrant.host", "qdrant.port",
        "embedding.base_url", "embedding.model", "embedding.api_key", "embedding.vector_dim",
    )
    _INT_KEYS = ("qdrant.port", "embedding.vector_dim")

    def _cmd_config(parts):
        if not parts or parts[0] in ("show", ""):
            return qconfig.describe() + f"\n\nConfig file: {qconfig.CONFIG_PATH}"
        act = parts[0]
        if act == "reset":
            if qconfig.CONFIG_PATH.exists():
                qconfig.CONFIG_PATH.unlink()
            return "Reset to built-in defaults (config.json removed).\n\n" + qconfig.describe()
        if act == "get":
            if len(parts) < 2 or parts[1] not in _VALID_KEYS:
                return f"Usage: /qdrant-index config get <key>  (keys: {', '.join(_VALID_KEYS)})"
            if parts[1] == "enabled":
                return f"enabled = {str(qconfig.is_enabled()).lower()}"
            section, key = parts[1].split(".", 1)
            return f"{parts[1]} = {qconfig.load_config()[section][key]}"
        if act == "set":
            if len(parts) < 3 or parts[1] not in _VALID_KEYS:
                return f"Usage: /qdrant-index config set <key> <value>  (keys: {', '.join(_VALID_KEYS)})"
            if parts[1] == "enabled":
                if parts[2] in ("on", "true", "1"):
                    qconfig.save_config({"enabled": True})
                elif parts[2] in ("off", "false", "0"):
                    qconfig.save_config({"enabled": False})
                else:
                    return "'enabled' must be on/off (or true/false, 1/0)"
                return ("Automatic indexing " + ("enabled." if parts[2] in ("on", "true", "1") else "DISABLED.")
                        + "\n\n" + qconfig.describe())
            section, key = parts[1].split(".", 1)
            value = parts[2]
            if parts[1] in _INT_KEYS:
                try:
                    value = str(int(value))
                except ValueError:
                    return f"'{parts[1]}' must be an integer"
            qconfig.save_config({section: {key: value}})
            return f"Set {parts[1]} = {value}\n\n" + qconfig.describe()
        return f"Unknown config action '{act}'. Try: show | get <key> | set <key> <value> | reset"

    async def _cmd_qdrant(raw_args: str):
        parts = (raw_args or "").split()
        if not parts:
            return ("Usage: /qdrant-index index [dir] | status [dir] | "
                    "search <query> [limit] | list | delete <collection> | "
                    "config [show | set <key> <value> | reset]")
        sub = parts[0]
        if sub == "config":
            return _cmd_config(parts[1:])
        if sub == "list":
            return await _do_list_collections()
        if sub == "status":
            return await _do_status(directory=" ".join(parts[1:]))
        if sub == "index":
            # Reuse the tool handler so the slash command gets the same
            # background-thread behaviour (no 30s command.dispatch stall).
            return await qdrant_index({"directory": " ".join(parts[1:])})
        if sub == "search":
            rest = list(parts[1:])
            limit = 5
            if rest and rest[-1].isdigit():
                limit = int(rest.pop())
            if not rest:
                return "Usage: /qdrant-index search <query> [limit]"
            return await _do_search(query=" ".join(rest), limit=limit)
        if sub == "delete":
            if len(parts) < 2:
                return "Usage: /qdrant-index delete <collection>"
            return await _do_delete_collection(collection_name=parts[1])
        return f"Unknown subcommand '{sub}'. Try: index | status | search | list | delete"

    ctx.register_command(
        "qdrant-index",
        _cmd_qdrant,
        description="Qdrant semantic index: index/status/search/list/delete for the current project",
        args_hint="[index|status|search|list|delete] [args]",
    )

    # --- T7: register_system_prompt_section — project-selection awareness ---
    def _prompt_section(info):
        cwd = info.get("cwd") or ""
        if not cwd:
            return ""
        collection = registry.collection_for_root(cwd)
        if not collection:
            return ""  # project not indexed — stay silent (YAGNI)
        cache = core.load_hash_cache(collection)
        st = core.compute_status(cwd, cache=cache, registry=registry.load(), collection=collection)
        if st["stale"]:
            return (f"[qdrant] {collection}: {st['changed']} changed + {st['new']} new files "
                    f"since last index — use qdrant_search for semantic lookups, "
                    f"qdrant_index to refresh.")
        return (
            f"[qdrant] {collection}: {st['file_count']} files indexed, fresh.\n"
            f"  qdrant_search is available for semantic search over THIS project's code.\n"
            f"  Use it when it fits — e.g. 'how does X work' / 'what does the code do' questions where you want behavior-based ranking and related files (service, UI, tests) surfaced together. "
            f"Use search_files (ripgrep) for exact string/symbol lookups."
        )

    ctx.register_system_prompt_section("qdrant.index-status", _prompt_section, max_chars=500)
