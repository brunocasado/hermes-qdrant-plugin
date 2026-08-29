from pathlib import Path


SOURCE = (Path(__file__).resolve().parent.parent / "__init__.py").read_text()


def test_search_retrieves_broadly_and_returns_narrowly():
    assert "fetch_limit = max(60, limit * 6)" in SOURCE
    assert "summaries[:min(limit, 8)]" in SOURCE
    assert "core.format_file_results(summaries, query)" in SOURCE


def test_search_tool_tells_agent_to_read_real_files():
    assert "Read the returned real files before reasoning or editing" in SOURCE


def test_min_score_default_filters_low_signal_dense_hits():
    import core
    # core.py search_qdrant signature default
    assert core.search_qdrant.__defaults__[-1] == 0.25
    # tool schema default shown to the model
    assert '"min_score": {"type": "number", "default": 0.25' in SOURCE
