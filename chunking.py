"""Token-budgeted fallback chunking shared by structural parsers."""
from __future__ import annotations

from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Conservative local-code token estimate (roughly 2 chars/token)."""
    return max(1, (len(text) + 1) // 2) if text else 0


def token_chunks_from_text(
    text: str,
    *,
    filepath: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
    base_line: int = 1,
    chunk_type: str = "text",
    symbols: list[str] | None = None,
) -> list[dict]:
    if not text:
        return []
    chunk_tokens = max(1, int(chunk_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), chunk_tokens - 1))
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    while start < len(lines):
        end = start
        used = 0
        while end < len(lines):
            cost = estimate_tokens(lines[end] + "\n")
            if end > start and used + cost > chunk_tokens:
                break
            if cost > chunk_tokens and end == start:
                # Pathological generated/minified line: character windows keep
                # the embedding request bounded while preserving line evidence.
                max_chars = chunk_tokens * 2
                overlap_chars = overlap_tokens * 2
                raw = lines[end]
                offset = 0
                while offset < len(raw):
                    piece = raw[offset: offset + max_chars]
                    chunks.append({
                        "file": filepath,
                        "chunk": piece,
                        "line_start": base_line + end,
                        "line_end": base_line + end,
                        "chunk_type": chunk_type,
                        "symbols": list(symbols or []),
                    })
                    step = max(1, max_chars - overlap_chars)
                    offset += step
                end += 1
                used = 0
                break
            used += cost
            end += 1
        if end > start and used:
            chunks.append({
                "file": filepath,
                "chunk": "\n".join(lines[start:end]),
                "line_start": base_line + start,
                "line_end": base_line + end - 1,
                "chunk_type": chunk_type,
                "symbols": list(symbols or []),
            })
        if end >= len(lines):
            break
        next_start = end
        overlap = 0
        while next_start > start:
            cost = estimate_tokens(lines[next_start - 1] + "\n")
            if overlap + cost > overlap_tokens:
                break
            next_start -= 1
            overlap += cost
        start = next_start if next_start > start else end

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def token_chunks_file(filepath: str, chunk_tokens: int = 400,
                      overlap_tokens: int = 60) -> list[dict]:
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return token_chunks_from_text(
        text, filepath=filepath, chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
