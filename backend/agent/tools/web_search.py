"""Read-only public web search tool."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.agent.schemas import EvidenceItem, ToolExecutionContext, ToolObservation
from backend.agent.tools.base import AgentTool, ToolSpec
from backend.services.web_search.search_providers import web_search_service


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    recency_days: Optional[int] = Field(default=None, ge=1, le=3650)
    domains: List[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=8)
    target_slots: List[str] = Field(default_factory=list)


class WebSearchTool(AgentTool):
    spec = ToolSpec(
        name="web_search",
        description="Search public web sources when current or external information is needed.",
        input_schema=WebSearchInput.model_json_schema(),
        required_permissions=["web:read"],
        timeout_ms=30000,
        max_retries=1,
        idempotent=True,
    )

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolObservation:
        params = WebSearchInput.model_validate(arguments)
        results = await web_search_service.search_with_snippets(
            query=params.query,
            max_results=params.max_results,
        )
        allowed_domains = {item.lower().removeprefix("www.") for item in params.domains}
        evidence: List[EvidenceItem] = []
        seen_urls = set()
        for item in results:
            url = str(item.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            if allowed_domains and not any(domain in url.lower() for domain in allowed_domains):
                continue
            snippet = str(item.get("snippet") or item.get("content") or "")
            evidence.append(EvidenceItem(
                source_type="web",
                title=str(item.get("title") or ""),
                url=url,
                support_snippet=snippet[:1000],
                snippet=snippet[:1600],
                score=float(item.get("score") or 0.0),
                query=params.query,
                target_slots=params.target_slots,
                fetched_at=datetime.now(timezone.utc),
                metadata={"recency_days": params.recency_days},
            ))
        status = "success" if evidence else "empty"
        return ToolObservation(
            observation_id=str(uuid.uuid4()),
            tool_call_id="",
            tool_name=self.spec.name,
            status=status,
            summary=(
                f"Web search returned {len(evidence)} sources for: {params.query}"
                if evidence
                else f"Web search found no usable sources for: {params.query}"
            ),
            evidence=evidence,
            # Search hits are candidates, not proof that every requested slot
            # has already been answered.
            supported_slots=[],
            missing_slots=[] if evidence else params.target_slots,
            next_hint="answer" if evidence else "clarify",
            metrics={"query": params.query, "evidence_count": len(evidence)},
        )
