#!/usr/bin/env python3
"""qdrant-index CLI — thin wrapper over core.py for terminal use.

Usage:
  qidx index [dir] [--collection NAME] [--reindex]
  qidx status [dir]
  qidx search QUERY [--collection NAME] [--limit N] [--min-score F]
  qidx list
  qidx delete COLLECTION
  qidx config show | get KEY | set KEY VALUE | reset

Default dir: current working directory.
Default collection: auto-resolved from the registry for that dir
(auto ws-<md5[:16]> if never indexed — pass --collection to name it).

`config` edits where the plugin talks to Qdrant and the embedding model at
runtime (no code change). Valid keys:
  qdrant.host  qdrant.port
  embedding.base_url  embedding.model  embedding.api_key  embedding.vector_dim
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import qconfig  # noqa: E402
import registry  # noqa: E402


def _resolve_dir(d: str) -> str:
    return str(Path(d).expanduser().resolve()) if d else str(Path.cwd())


def cmd_index(a):
    root = _resolve_dir(a.directory)
    collection = a.collection or registry.collection_for_root(root)
    out = asyncio.run(core.index_directory(
        root,
        collection_name=collection,
        chunk_size=a.chunk_size or core.CHUNK_SIZE,
        chunk_overlap=a.chunk_overlap or core.CHUNK_OVERLAP,
        reindex=a.reindex,
    ))
    print(out)


def cmd_status(a):
    root = _resolve_dir(a.directory)
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
        lines.append(f"  ✓ {st['unchanged']} unchanged"
                     + (f", ~ {st['changed']} changed" if st["changed"] else "")
                     + (f", + {st['new']} new" if st["new"] else ""))
        lines.append("Status: STALE — run 'index' to refresh" if st["stale"] else "Status: FRESH")
    else:
        state = st.get("collection_state")
        if state in ("missing", "empty"):
            lines.append(f"Not indexed — collection '{st['collection']}' is {state} in Qdrant.")
            lines.append(f"Run 'index' to rebuild it ({st['total']} indexable files found).")
        else:
            lines.append(f"Not indexed. {st['total']} indexable files found.")
    print("\n".join(lines))


def cmd_search(a):
    root = _resolve_dir("")
    collection = a.collection or registry.collection_for_root(root)
    if not collection:
        reg = registry.load()
        avail = ", ".join(reg.keys()) if reg else "no collections indexed"
        sys.exit(f"No collection for '{root}'. Available: {avail}")
    query = " ".join(a.query) if isinstance(a.query, list) else a.query
    fetch_limit = max(a.limit * 3, 15)
    hits = asyncio.run(core.search_qdrant(collection, query, fetch_limit, a.min_score))
    if not hits:
        print(f"No results for: '{query}'")
        return
    summaries = core.aggregate_hits_by_file(hits, top_chunks_per_file=1)[:a.limit]
    print(f"{len(summaries)} file(s), {len(hits)} chunk(s) matched\n")
    for i, s in enumerate(summaries, 1):
        print(f"--- {i}. {s['file']} (best {s['best_score']:.4f}, "
              f"{s['chunk_count']} chunk(s), lines {s['line_start']}-{s['line_end']})")
        for ln in s["best_chunk"].splitlines()[:30]:
            print("    " + ln)
        print()


def cmd_list(_a):
    c = core.get_client()
    reg = registry.load()
    for coll in c.get_collections().collections:
        info = c.get_collection(coll.name)
        entry = reg.get(coll.name)
        root_s = entry["root"] if entry else "(not in registry)"
        print(f"  {coll.name}: {info.points_count} points — {root_s}")


def cmd_delete(a):
    c = core.get_client()
    c.delete_collection(a.collection)
    reg = registry.load()
    if a.collection in reg:
        del reg[a.collection]
        registry.save(reg)
    print(f"Deleted collection '{a.collection}' (registry entry removed if present)")


# --- config: read/edit the Qdrant server & embedding endpoint at runtime ---
# Keys are dotted, section-prefixed:
#   qdrant.host  qdrant.port
#   embedding.base_url  embedding.model  embedding.api_key  embedding.vector_dim
_CONFIG_KEYS = {
    "qdrant.host": ("qdrant", "host"),
    "qdrant.port": ("qdrant", "port"),
    "embedding.base_url": ("embedding", "base_url"),
    "embedding.model": ("embedding", "model"),
    "embedding.api_key": ("embedding", "api_key"),
    "embedding.vector_dim": ("embedding", "vector_dim"),
}
_INT_KEYS = {"qdrant.port", "embedding.vector_dim"}


def _parse_config_key(key: str):
    if key not in _CONFIG_KEYS:
        valid = ", ".join(_CONFIG_KEYS)
        raise SystemExit(f"Unknown config key '{key}'. Valid keys: {valid}")
    return _CONFIG_KEYS[key]


def _coerce(key: str, value: str) -> str:
    if key in _INT_KEYS:
        try:
            return str(int(value))
        except ValueError:
            raise SystemExit(f"'{key}' must be an integer, got '{value}'")
    return value


def cmd_config(a):
    sub = a.action
    if sub == "show":
        print(qconfig.describe())
        print(f"\nConfig file: {qconfig.CONFIG_PATH}")
        return
    if sub == "get":
        section, key = _parse_config_key(a.key)
        val = qconfig.load_config()[section][key]
        print(val)
        return
    if sub == "set":
        section, key = _parse_config_key(a.key)
        value = _coerce(a.key, a.value)
        qconfig.save_config({section: {key: value}})
        print(f"Set {a.key} = {value}")
        print(qconfig.describe())
        return
    if sub == "reset":
        # Remove the persisted config file so everything falls back to defaults.
        if qconfig.CONFIG_PATH.exists():
            qconfig.CONFIG_PATH.unlink()
        print("Reset config to built-in defaults (config.json removed).")
        print(qconfig.describe())
        return
    raise SystemExit(f"Unknown config action '{sub}'")


def main():
    ap = argparse.ArgumentParser(prog="qdrant-index", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index a directory (incremental by default)")
    p.add_argument("directory", nargs="?", default="")
    p.add_argument("--collection", default="")
    p.add_argument("--reindex", action="store_true", help="force full re-index")
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--chunk-overlap", type=int, default=0)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("status", help="index health for a project")
    p.add_argument("directory", nargs="?", default="")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("search", help="semantic search")
    p.add_argument("query", nargs="+", help="query words (joined with spaces)")
    p.add_argument("--collection", default="")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.0)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("list", help="list collections")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("delete", help="delete a collection")
    p.add_argument("collection")
    p.set_defaults(fn=cmd_delete)

    p = sub.add_parser("config", help="show/edit the Qdrant server & embedding endpoint")
    csub = p.add_subparsers(dest="action", required=True)
    csub.add_parser("show", help="print the resolved config")
    pg = csub.add_parser("get", help="print one key's value")
    pg.add_argument("key", choices=sorted(_CONFIG_KEYS))
    ps = csub.add_parser("set", help="set one key and persist to config.json")
    ps.add_argument("key", choices=sorted(_CONFIG_KEYS))
    ps.add_argument("value")
    csub.add_parser("reset", help="remove config.json and fall back to defaults")
    p.set_defaults(fn=cmd_config)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
