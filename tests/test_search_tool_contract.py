from pathlib import Path


SOURCE = (Path(__file__).resolve().parent.parent / "__init__.py").read_text()


def test_search_retrieves_broadly_and_returns_narrowly():
    assert "fetch_limit = max(60, limit * 6)" in SOURCE
    assert "summaries[:min(limit, 8)]" in SOURCE
    assert "core.format_file_results(summaries, query)" in SOURCE


def test_search_tool_tells_agent_to_read_real_files():
    assert "Read the returned real files before reasoning or editing" in SOURCE
