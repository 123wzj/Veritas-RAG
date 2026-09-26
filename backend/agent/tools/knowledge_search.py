"""Knowledge-base search tool backed by the existing hybrid retrieval stack."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from backend.agent.schemas import EvidenceItem, ToolExecutionContext, ToolObservation
from backend.agent.tools.base import AgentTool, ToolSpec
from backend.services.retrieval.diversity import select_diverse_results
from backend.services.retrieval.hybrid import hybrid_retriever
from backend.services.retrieval.reranker import reranker


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    search_intent: Literal["fact", "comparison", "procedure", "summary", "analysis"] = "fact"
    document_ids: List[str | int] = Field(default_factory=list)
    section_paths: List[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=12)
    need_parent_context: bool = True
    query_variants: List[str] = Field(default_factory=list)
    target_slots: List[str] = Field(default_factory=list)


class KnowledgeSearchTool(AgentTool):
    spec = ToolSpec(
        name="knowledge_search",
        description="Search the selected private knowledge base using Dense, BM25, RRF and reranking.",
        input_schema=KnowledgeSearchInput.model_json_schema(),
        required_permissions=["kb:read"],
        timeout_ms=45000,
        max_retries=1,
        idempotent=True,
    )

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolObservation:
        params = KnowledgeSearchInput.model_validate(arguments)
        variants = list(dict.fromkeys([
            params.query,
            *[item.strip() for item in params.query_variants if item.strip()],
        ]))[:4]
        raw = await hybrid_retriever.retrieve_async(
            query=params.query,
            user_id=context.user_id,
            kb_id=int(context.kb_id),
            query_variants=variants,
            top_k=max(params.top_k * 3, 12),
        )
        if params.document_ids:
            allowed_docs = {str(item) for item in params.document_ids}
            raw = [item for item in raw if str(item.get("doc_id")) in allowed_docs]
        if params.section_paths:
            prefixes = tuple(params.section_paths)
            raw = [
                item for item in raw
                if str(item.get("section_path") or item.get("parent_section_path") or "").startswith(prefixes)
            ]

        try:
            ranked = reranker.rerank(
                query=params.query,
                documents=[item.copy() for item in raw],
                top_k=max(params.top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )
        except Exception:
            ranked = select_diverse_results(
                [item.copy() for item in raw],
                top_k=max(params.top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )
        ranked = select_diverse_results(
            ranked,
            top_k=params.top_k,
            max_per_doc=3,
            max_per_parent=1,
        )

        evidence: List[EvidenceItem] = []
        for doc in ranked:
            child = str(doc.get("content") or "")
            parent = str(doc.get("parent_content") or "")
            score = float(doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0))) or 0.0)
            evidence.append(EvidenceItem(
                source_type="knowledge_base",
                title=str(doc.get("parent_title") or doc.get("title") or ""),
                support_snippet=child[:600],
                snippet=(parent[:1600] if params.need_parent_context and parent else child[:1000]),
                doc_id=str(doc.get("doc_id")) if doc.get("doc_id") is not None else None,
                chunk_id=str(doc.get("chunk_id")) if doc.get("chunk_id") is not None else None,
                parent_id=str(doc.get("parent_id")) if doc.get("parent_id") is not None else None,
                section_path=doc.get("parent_section_path") or doc.get("section_path"),
                page_no=doc.get("parent_page_no") or doc.get("page_no"),
                score=score,
                retrieval_score=float(doc.get("rrf_score", doc.get("score", 0.0)) or 0.0),
                rerank_score=float(doc.get("rerank_score", 0.0) or 0.0),
                query=params.query,
                target_slots=params.target_slots,
                metadata={
                    "retrieval_type": doc.get("retrieval_type"),
                    "match_query": doc.get("match_query"),
                    "kb_id": context.kb_id,
                },
            ))

        status = "success" if evidence else "empty"
        return ToolObservation(
            observation_id=str(uuid.uuid4()),
            tool_call_id="",
            tool_name=self.spec.name,
            status=status,
            summary=(
                f"Knowledge search returned {len(evidence)} evidence items for: {params.query}"
                if evidence
                else f"Knowledge search found no usable evidence for: {params.query}"
            ),
            evidence=evidence,
            # Retrieval produces candidate evidence.  Slot support is decided
            # only after final claims are explicitly mapped to evidence.
            supported_slots=[],
            missing_slots=[] if evidence else params.target_slots,
            next_hint="answer" if evidence else "refine_query",
            metrics={
                "query": params.query,
                "candidate_count": len(raw),
                "evidence_count": len(evidence),
            },
        )
