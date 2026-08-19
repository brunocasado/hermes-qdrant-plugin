# qdrant-index

Hermes Agent plugin: semantic index of your project files in [Qdrant](https://qdrant.tech), with agent tools, auto-reindex hooks, and a desktop statusbar pill.

## What it does

- **`qdrant_index <path>`** — chunks a project directory, embeds the chunks (any OpenAI-compatible embeddings endpoint), and upserts them into a Qdrant collection named after the project.
- **`qdrant_search <query>`** — semantic search across all indexed projects (or one collection).
- **`qdrant_status`** / **`qdrant_list_collections`** / **`qdrant_delete_collection`** — inspect and manage collections.
- **`qdrant_set_server`** — repoint the plugin at a different Qdrant server or embedding endpoint at runtime (persisted to `data/config.json`).
- **Auto-reindex hook** — after file-mutating tool calls, changed files are re-indexed incrementally (hash-cache based, idempotent, deterministic point IDs).
- **Desktop pill** — statusbar indicator with a popover: per-collection status, progress during indexing, settings (master switch for auto-indexing), and manual "Index now" / "Reindex".

## Requirements

- Hermes Agent (desktop or CLI)
- A Qdrant server (default `localhost:6333`)
- An OpenAI-compatible embeddings endpoint (default `http://localhost:8080/v1`, model `embeddings`, 768-dim — e.g. a local llama-swap/LiteLLM gateway; any compatible API works)

## Install

Copy this directory into `~/.hermes/plugins/qdrant-index` and restart Hermes.

## Configure

Precedence (highest first): environment variables → `data/config.json` → built-in defaults.

| Env var | Config key | Meaning |
|---|---|---|
| `QDRANT_HOST` | `qdrant.host` | Qdrant server host |
| `QDRANT_PORT` | `qdrant.port` | Qdrant server port (default 6333) |
| `EMBEDDING_BASE_URL` | `embedding.base_url` | OpenAI-compatible embeddings base URL |
| `EMBEDDING_MODEL` | `embedding.model` | Embedding model name |
| `EMBEDDING_API_KEY` | `embedding.api_key` | API key (`EMPTY` for unauthenticated local servers) |
| `EMBEDDING_VECTOR_DIM` | `embedding.vector_dim` | Vector dimension (must match the model) |

Or use the CLI / agent tool:

```bash
qidx config set embedding.base_url http://your-host:4000/v1
qidx config set qdrant.host 10.0.0.5
qidx status
```

`data/config.json` is local state — it is gitignored and created on first write.

## CLI

`qidx_cli.py` (`qidx`) — `status`, `index <path>`, `search <query>`, `config [set key value]`, `list`, `delete <collection>`.

## Tests

```bash
python3 -m pytest tests/ -q
```

## Notes

- Point IDs are deterministic (`md5(file:chunk_index)`), so re-indexing is idempotent.
- If a collection is deleted externally, the next status check detects it (live `count()` with a 30 s TTL cache) and the next index run rebuilds it from scratch.
- `data/` (registry, hash-cache, config) is per-machine state and is never published.
