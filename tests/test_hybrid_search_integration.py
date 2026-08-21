import asyncio
from types import SimpleNamespace

import core


class FakeSearchClient:
    def __init__(self):
        self.calls = []
        self.dense = [
            SimpleNamespace(id="a", score=0.90, payload={"rel_path": "a.py", "file": "/p/a.py"}),
            SimpleNamespace(id="b", score=0.80, payload={"rel_path": "b.py", "file": "/p/b.py"}),
        ]
        self.lexical = [
            SimpleNamespace(id="b", score=4.0, payload={"rel_path": "b.py", "file": "/p/b.py"}),
            SimpleNamespace(id="c", score=3.0, payload={"rel_path": "c.py", "file": "/p/c.py"}),
        ]

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.dense if kwargs.get("using") == "dense" else self.lexical)


def test_exact_identifier_search_uses_lexical_without_embedding(monkeypatch):
    fake = FakeSearchClient()
    monkeypatch.setattr(core, "get_client", lambda: fake)

    async def embedding_must_not_run(_texts):
        raise AssertionError("dense embedding should not run for exact identifier")

    monkeypatch.setattr(core, "get_embeddings", embedding_must_not_run)

    hits = asyncio.run(core.search_qdrant("project", "FollowUpAfter", limit=20))

    assert [call["using"] for call in fake.calls] == ["lexical"]
    assert [hit.id for hit in hits] == ["b", "c"]


def test_mixed_query_runs_dense_and_lexical_then_rrf(monkeypatch):
    fake = FakeSearchClient()
    monkeypatch.setattr(core, "get_client", lambda: fake)

    async def fake_embeddings(_texts):
        return [[0.1] * 768]

    monkeypatch.setattr(core, "get_embeddings", fake_embeddings)

    hits = asyncio.run(core.search_qdrant(
        "project", "where is FollowUpAfter used to schedule sends?", limit=20,
    ))

    assert [call["using"] for call in fake.calls] == ["dense", "lexical"]
    assert hits[0].id == "b"  # corroborated by both rankings
    assert {hit.id for hit in hits} == {"a", "b", "c"}


def test_semantic_query_prefers_dense_only(monkeypatch):
    fake = FakeSearchClient()
    monkeypatch.setattr(core, "get_client", lambda: fake)

    async def fake_embeddings(_texts):
        return [[0.1] * 768]

    monkeypatch.setattr(core, "get_embeddings", fake_embeddings)

    hits = asyncio.run(core.search_qdrant(
        "project", "where is the desktop statusbar pill rendered?", limit=20,
    ))

    assert [call["using"] for call in fake.calls] == ["dense"]
    assert {hit.id for hit in hits} == {"a", "b"}
