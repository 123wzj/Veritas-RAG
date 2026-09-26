import asyncio

from backend.agent.schemas import ToolExecutionContext
from backend.agent.tools.knowledge_search import KnowledgeSearchTool
from backend.agent.tools.web_search import WebSearchTool


def test_knowledge_search_wraps_existing_retrieval(monkeypatch):
    async def fake_retrieve_async(**kwargs):
        assert kwargs["user_id"] == 3
        assert kwargs["kb_id"] == 8
        return [{
            "doc_id": "uuid-doc",
            "chunk_id": "c1",
            "content": "child",
            "parent_content": "parent context",
            "title": "Title",
            "rrf_score": 0.8,
        }]

    monkeypatch.setattr(
        "backend.agent.tools.knowledge_search.hybrid_retriever.retrieve_async",
        fake_retrieve_async,
    )
    monkeypatch.setattr(
        "backend.agent.tools.knowledge_search.reranker.rerank",
        lambda **kwargs: kwargs["documents"],
    )
    observation = asyncio.run(KnowledgeSearchTool().execute(
        {"query": "where", "document_ids": ["uuid-doc"], "target_slots": ["slot-1"]},
        ToolExecutionContext(
            run_id="r", request_id="r", user_id=3, session_id="s", kb_id=8
        ),
    ))
    assert observation.status == "success"
    assert observation.evidence[0].doc_id == "uuid-doc"
    assert observation.supported_slots == []
    assert observation.evidence[0].target_slots == ["slot-1"]


def test_web_search_wraps_provider_and_filters_domains(monkeypatch):
    async def fake_search(**kwargs):
        return [
            {"title": "Official", "url": "https://docs.example.com/a", "snippet": "ok"},
            {"title": "Other", "url": "https://other.example/b", "snippet": "skip"},
        ]

    monkeypatch.setattr(
        "backend.agent.tools.web_search.web_search_service.search_with_snippets",
        fake_search,
    )
    observation = asyncio.run(WebSearchTool().execute(
        {"query": "latest", "domains": ["docs.example.com"], "target_slots": ["slot-1"]},
        ToolExecutionContext(
            run_id="r", request_id="r", user_id=3, session_id="s", web_enabled=True
        ),
    ))
    assert observation.status == "success"
    assert [item.title for item in observation.evidence] == ["Official"]
