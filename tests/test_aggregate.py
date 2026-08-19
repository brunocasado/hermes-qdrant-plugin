"""Tests for core.aggregate_hits_by_file — the per-file dedup helper."""
import sys
from pathlib import Path

# Make the plugin dir importable when run standalone (not as a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import aggregate_hits_by_file


class FakeHit:
    """Minimal stand-in for a Qdrant ScoredPoint (only .score + .payload are used)."""
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


def _hit(file, score, chunk, line_start=1, line_end=10, chunk_index=0):
    return FakeHit(score, {
        "file": file,
        "chunk": chunk,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_index": chunk_index,
    })


def test_single_hit_single_file():
    hits = [_hit("/a.py", 0.9, "print('hello')")]
    out = aggregate_hits_by_file(hits)
    assert len(out) == 1
    assert out[0]["file"] == "/a.py"
    assert out[0]["chunk_count"] == 1
    assert out[0]["best_score"] == 0.9
    assert out[0]["best_chunk"] == "print('hello')"


def test_multiple_chunks_same_file_collapses():
    hits = [
        _hit("/a.py", 0.7, "line A", line_start=1, line_end=10, chunk_index=0),
        _hit("/a.py", 0.9, "line B", line_start=11, line_end=20, chunk_index=1),
        _hit("/a.py", 0.5, "line C", line_start=21, line_end=30, chunk_index=2),
    ]
    out = aggregate_hits_by_file(hits)
    assert len(out) == 1
    entry = out[0]
    assert entry["file"] == "/a.py"
    assert entry["chunk_count"] == 3
    assert entry["best_score"] == 0.9
    # The best (highest-scoring) chunk is surfaced, not the first.
    assert entry["best_chunk"] == "line B"
    # Line range spans all matched chunks of that file.
    assert entry["line_start"] == 1
    assert entry["line_end"] == 30


def test_sorted_by_best_score_desc():
    hits = [
        _hit("/low.py", 0.3, "x"),
        _hit("/high.py", 0.8, "y"),
        _hit("/mid.py", 0.6, "z"),
    ]
    out = aggregate_hits_by_file(hits)
    files = [e["file"] for e in out]
    assert files == ["/high.py", "/mid.py", "/low.py"]


def test_top_chunks_per_file_respected():
    hits = [
        _hit("/a.py", 0.9, "best", chunk_index=0),
        _hit("/a.py", 0.8, "second", chunk_index=1),
        _hit("/a.py", 0.7, "third", chunk_index=2),
    ]
    out = aggregate_hits_by_file(hits, top_chunks_per_file=2)
    entry = out[0]
    # best_chunk is still the single highest.
    assert entry["best_chunk"] == "best"
    # The extra chunks are preserved in score-desc order.
    assert [c["chunk"] for c in entry["chunks"]] == ["best", "second"]


def test_empty_input():
    assert aggregate_hits_by_file([]) == []


def test_payload_missing_chunk_defaults_empty():
    hits = [FakeHit(0.5, {"file": "/x.py", "line_start": 1, "line_end": 2})]
    out = aggregate_hits_by_file(hits)
    assert out[0]["best_chunk"] == ""
