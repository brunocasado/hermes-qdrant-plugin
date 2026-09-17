"""Auto-search (pre_llm_call) gating + formatting — pure logic, no Qdrant."""
import importlib.util
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PLUGIN_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


auto_search = _load("auto_search")


def _summaries(file_score, **over):
    base = {
        "file": "core.py", "rel_path": "core.py", "chunk_count": 1,
        "best_score": file_score, "file_score": file_score,
        "symbols": ["search_qdrant"], "line_start": 10, "line_end": 24,
        "best_chunk": "def search_qdrant(collection_name, query): ...",
        "chunks": [],
    }
    base.update(over)
    return [base]


# --- should_run gating -------------------------------------------------

def test_run_when_real_question_and_indexed():
    assert auto_search.should_run(
        "how does the reindex debounce work",
        auto_search_enabled=True, has_collection=True,
    )


def test_skip_short_messages():
    assert not auto_search.should_run("ok", auto_search_enabled=True, has_collection=True)
    assert not auto_search.should_run("  ", auto_search_enabled=True, has_collection=True)
    assert not auto_search.should_run("", auto_search_enabled=True, has_collection=True)


def test_skip_slash_commands():
    assert not auto_search.should_run(
        "/qdrant status", auto_search_enabled=True, has_collection=True,
    )


def test_skip_when_disabled_or_not_indexed():
    msg = "how does the reindex debounce work"
    assert not auto_search.should_run(msg, auto_search_enabled=False, has_collection=True)
    assert not auto_search.should_run(msg, auto_search_enabled=True, has_collection=False)


def test_skip_non_string_messages():
    assert not auto_search.should_run(None, auto_search_enabled=True, has_collection=True)
    assert not auto_search.should_run(42, auto_search_enabled=True, has_collection=True)


# --- build_context: route-aware score gating ---------------------------

def test_semantic_route_requires_dense_bar():
    assert auto_search.build_context("q", _summaries(0.28), "semantic") is None
    assert auto_search.build_context("q", _summaries(0.35), "semantic") is not None


def test_lexical_and_hybrid_routes_inject_without_dense_bar():
    # RRF/sparse scores are incomparable with cosine — a top-ranked lexical
    # match is evidence by construction (small score, must still inject).
    assert auto_search.build_context("q", _summaries(0.016), "lexical") is not None
    assert auto_search.build_context("q", _summaries(0.03), "hybrid") is not None


def test_empty_summaries_never_inject():
    assert auto_search.build_context("q", [], "hybrid") is None


def test_block_is_additive_and_names_other_tools():
    block = auto_search.build_context("q", _summaries(0.4), "semantic")
    assert auto_search.INJECTION_MARKER in block
    assert "core.py" in block
    assert "qdrant_search" in block      # deeper queries stay available
    assert "search_files" in block       # exact matches stay available
    assert "All tools remain available" in block


def test_block_respects_char_budget():
    big = _summaries(0.4, best_chunk="x " * 5000)
    block = auto_search.build_context("q", big, "semantic")
    assert len(block) <= auto_search.MAX_EVIDENCE_CHARS + 32  # cap + truncation marker
