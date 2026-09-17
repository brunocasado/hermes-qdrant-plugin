"""hermes-qdrant-plugin — per-turn auto-search (pre_llm_call) policy.

Pure decision + formatting logic for the automatic retrieval that runs on
every user turn: decide whether the user's message is worth a semantic
search, and render the retrieved evidence as the context block that Hermes
appends to the user message at API time (the ``pre_llm_call`` hook contract).

The block is additive starting evidence: it never removes tools from the
model's schema, and it says so explicitly, so the model can still pick
``qdrant_search`` (deeper/different query), ``search_files`` (exact
matches) or any other tool.

This module is intentionally dependency-free (no ``core``, no
``qdrant_client``) so the gating/formatting logic is unit-testable without
a Qdrant server.
"""
from __future__ import annotations

MIN_QUERY_CHARS = 12        # shorter than this: greeting/chat, not a codebase question
MIN_DENSE_FILE_SCORE = 0.30  # cosine-scale bar for semantic-route injection (tool floor is 0.25)
MAX_EVIDENCE_CHARS = 2500   # hard cap on the injected block
MAX_FILES = 5

INJECTION_MARKER = "[qdrant auto-search]"


def should_run(user_message, *, auto_search_enabled: bool, has_collection: bool) -> bool:
    """Gate a user turn: cheap checks only (no I/O, no Qdrant)."""
    if not auto_search_enabled or not has_collection:
        return False
    if not isinstance(user_message, str):
        return False
    text = user_message.strip()
    if len(text) < MIN_QUERY_CHARS:
        return False
    if text.startswith("/"):  # slash command, not a question
        return False
    return True


def _format_evidence(summaries: list[dict], query: str) -> str:
    """Compact navigation evidence (same shape as core.format_file_results)."""
    lines = [f"Best files for: {query}", ""]
    for index, result in enumerate(summaries[:MAX_FILES], 1):
        lines.append(f"{index}. {result.get('rel_path') or result.get('file')} "
                     f"(score {result.get('file_score', 0):.4f})")
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


def build_context(query: str, summaries: list[dict], route: str) -> str | None:
    """Render the auto-injected block, or None when the evidence is not worth injecting.

    ``route`` is ``core.route_query(query)``: "semantic" (dense cosine
    scores), "lexical" (sparse scores) or "hybrid" (RRF scores). The 0.30
    bar only applies to the cosine scale — sparse/RRF scores are
    incomparable (core never applies the dense threshold to them), and a
    top-ranked lexical match is evidence by construction.
    """
    if not summaries:
        return None
    top = summaries[0]
    if route == "semantic" and float(top.get("file_score", 0)) < MIN_DENSE_FILE_SCORE:
        return None
    evidence = _format_evidence(summaries, query)
    block = (
        f"{INJECTION_MARKER} Auto-retrieved from this project's index for your message — "
        f"starting evidence only. All tools remain available: use qdrant_search for a "
        f"deeper/different query, search_files for exact matches, and read the returned "
        f"files before relying on them.\n" + evidence
    )
    if len(block) > MAX_EVIDENCE_CHARS:
        block = block[:MAX_EVIDENCE_CHARS] + "\n… (truncated)"
    return block
