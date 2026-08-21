from types import SimpleNamespace

import core


def hit(point_id, score, rel_path, *, symbols=None, chunk="code", start=1, end=2):
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "file": "/project/" + rel_path,
            "rel_path": rel_path,
            "basename": rel_path.rsplit("/", 1)[-1],
            "symbols": symbols or [],
            "chunk": chunk,
            "line_start": start,
            "line_end": end,
        },
    )


def test_multi_chunk_and_symbol_bonus_beats_accidental_match():
    hits = [
        hit("a1", 0.81, "scheduler.go", symbols=["ScheduleCampaign"]),
        hit("a2", 0.60, "scheduler.go", symbols=["calculateSendAt"]),
        hit("b1", 0.82, "unrelated.go"),
    ]

    ranked = core.aggregate_hits_by_file(hits, query="ScheduleCampaign")

    assert ranked[0]["rel_path"] == "scheduler.go"
    assert ranked[0]["file_score"] > ranked[1]["file_score"]
    assert set(ranked[0]["symbols"]) == {"ScheduleCampaign", "calculateSendAt"}


def test_rrf_rewards_items_present_in_both_rankings():
    fused = core.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)

    assert fused[:2] == ["a", "b"] or fused[:2] == ["b", "a"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_query_router_distinguishes_identifier_semantic_and_mixed():
    assert core.route_query("FollowUpAfter") == "lexical"
    assert core.route_query("where is follow-up scheduling calculated?") == "semantic"
    assert core.route_query("where is FollowUpAfter used to calculate next send time?") == "hybrid"
    assert core.route_query("internal/campaign/scheduler.go") == "lexical"


def test_sparse_vector_rewards_exact_code_identifiers():
    query = core.sparse_vector("FollowUpAfter")
    exact = core.sparse_vector("func FollowUpAfter() error")
    unrelated = core.sparse_vector("billing subscription items")

    query_indices = set(query.indices)
    assert query_indices & set(exact.indices)
    assert not (query_indices & set(unrelated.indices))


def test_result_formatter_is_navigation_evidence_not_code_dump():
    results = [{
        "rel_path": "internal/campaign/scheduler.go",
        "file_score": 0.91,
        "symbols": ["ScheduleCampaign", "calculateSendAt"],
        "line_start": 122,
        "line_end": 167,
        "best_chunk": "func ScheduleCampaign() {\n" + ("x" * 1000),
        "chunk_count": 3,
    }]

    rendered = core.format_file_results(results, query="campaign scheduling")

    assert "internal/campaign/scheduler.go" in rendered
    assert "ScheduleCampaign" in rendered
    assert "lines 122-167" in rendered
    assert len(rendered) < 800
