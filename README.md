# hermes-qdrant-plugin

Hermes Agent plugin that uses Qdrant as a **project file-discovery layer**. It retrieves a small set of likely files; Hermes then reads the real files and reasons from the actual source code.

```text
question → dense + lexical retrieval → file aggregation → top 5–8 files → Hermes reads real files
```

## What it does

- **`qdrant_index <path>`** — incrementally indexes a project; `reindex=true` performs a clean collection rebuild.
- **`qdrant_search <query>`** — returns file candidates with score, symbols, best line range and a short snippet.
- **`qdrant_status`**, **`qdrant_list_collections`**, **`qdrant_delete_collection`** — inspect/manage collections.
- **`qdrant_set_server`** — changes Qdrant or the OpenAI-compatible embedding endpoint at runtime.
- **Auto-index hook** — opt-in **per project**, disabled by default. Manual Index/Reindex always remains available.
- **Desktop pill** — shows focused-project status/progress and remembers its auto-index toggle independently from other projects.

## Retrieval architecture

1. Files are keyed by `rel_path`; stale points are deleted before changed files are upserted, and deleted files are purged.
2. Python uses stdlib AST structural chunks; JS/TS, Go, Rust and Java use official Tree-sitter bindings. Unsupported/worker-thread paths fall back safely to token-budgeted chunks.
3. Final enriched embedding inputs are hard-bounded for a 512-token-class model; content is split, never silently truncated.
4. Embedding text includes project, relative path, language, symbols and raw code.
5. Each point stores `rel_path`, basename, language, symbols, chunk type, line range and file hash.
6. Qdrant stores named `dense` and `lexical` sparse vectors.
7. Exact identifiers/path-like queries prefer lexical search; natural-language questions prefer dense; mixed queries combine both with Reciprocal Rank Fusion.
8. Internal retrieval is broad (at least 60 chunks); output is narrow (at most 8 files), ranked with multi-chunk, symbol and path evidence.

Qdrant is navigation evidence, not source of truth. Consumers should read returned files before reasoning or editing.

## Requirements

- Hermes Agent (desktop or CLI)
- Qdrant (default `localhost:6333`)
- OpenAI-compatible embeddings endpoint (default `http://localhost:8080/v1`, model `embeddings`, dimension 768)
- Python dependencies are declared in `plugin.yaml` and installed by Hermes.

## Install

```bash
hermes plugins install brunocasado/hermes-qdrant-plugin
```

Or clone into `~/.hermes/plugins/hermes-qdrant-plugin`. After install/update or route changes, fully quit Hermes and relaunch it so desktop REST routes are mounted.

## Configure

Precedence: environment variables → per-plugin `config.json` → built-in defaults.

| Env var | Config key | Meaning |
|---|---|---|
| `QDRANT_HOST` | `qdrant.host` | Qdrant host |
| `QDRANT_PORT` | `qdrant.port` | Qdrant port |
| `EMBEDDING_BASE_URL` | `embedding.base_url` | OpenAI-compatible base URL |
| `EMBEDDING_MODEL` | `embedding.model` | Embedding model |
| `EMBEDDING_API_KEY` | `embedding.api_key` | API key (`EMPTY` for unauthenticated local endpoints) |
| `EMBEDDING_VECTOR_DIM` | `embedding.vector_dim` | Vector dimension |

Auto-indexing is configured per project from the desktop pill or `/qdrant config set enabled on|off` while that project is focused. New projects default to off.

## State and concurrency

Durable state lives under `<hermes home>/plugin-data/hermes-qdrant-plugin/`, outside the install tree.

- Cross-process root + collection locks guarantee one active writer per project/collection across REST, agent tools, hooks and CLI.
- Shared config, registry and hash-cache updates use short metadata locks and atomic JSON replacement.
- Different projects/collections may still index concurrently.
- Same-basename projects receive stable disambiguated collection names; explicit conflicting names are rejected.

## Upgrade note

The current schema uses relative-path point IDs plus named dense/sparse vectors. Collections created by older versions require one clean rebuild:

```bash
python3 qidx_cli.py index /path/to/project --reindex
```

## CLI

```bash
python3 qidx_cli.py status /path/to/project
python3 qidx_cli.py index /path/to/project
python3 qidx_cli.py search where is campaign scheduling handled --collection project-name
python3 qidx_cli.py config show
python3 qidx_cli.py list
```

## Tests and benchmark

```bash
python3 -m pytest tests/ -q
node --check desktop/plugin.js
python3 benchmarks/run_benchmark.py
```

`benchmarks/queries.yaml` contains 30 manually-labelled discovery questions. The runner records Recall@1, Recall@3, Recall@5 and MRR in `benchmarks/RESULTS.md`; Recall@5 is the primary metric.

## Notes

- Point IDs are deterministic from `rel_path:chunk_index`.
- Out-of-band collection deletion is detected by a live Qdrant count with a short TTL cache.
- Legacy install-tree state is excluded from indexing and remains available for migration.
- Per-machine state and credentials are never published (see `.gitignore`).

## License

MIT — see [LICENSE](LICENSE).
