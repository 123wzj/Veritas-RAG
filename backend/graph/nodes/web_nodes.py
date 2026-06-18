# -*- coding: utf-8 -*-
"""
联网搜索节点。
"""

from typing import Dict, Any, List

from backend.graph.state.state import RAGState
from backend.services.web_search.search_providers import web_search_service


def _close_web_attempt(
    sub_query_plans: List[Dict[str, Any]],
    error: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    updated_plans: List[Dict[str, Any]] = []
    all_evidence: List[Dict[str, Any]] = []

    for plan in sub_query_plans:
        current_plan = plan.copy()
        evidence = list(current_plan.get("selected_evidence") or [])
        if current_plan.get("need_web_search"):
            current_plan["need_web_search"] = False
            current_plan["need_retrieval"] = False
            current_plan["evidence_sufficient"] = len(evidence) > 0
            if error:
                current_plan["web_search_error"] = error
        updated_plans.append(current_plan)
        all_evidence.extend(evidence)

    return updated_plans, all_evidence


async def web_search(state: RAGState) -> Dict[str, Any]:
    """
    当知识库证据不足且允许联网时，对未覆盖的子问题分别联网补充证据。
    """
    events = state.get("events", [])
    web_enabled = state.get("web_enabled", False)
    sub_query_plans = state.get("sub_query_plans") or []

    if not web_enabled:
        updated_plans, all_evidence = _close_web_attempt(
            sub_query_plans,
            error="web_search_not_enabled",
        )
        return {
            "sub_query_plans": updated_plans,
            "selected_evidence": all_evidence,
            "evidence_sufficient": bool(updated_plans) and all(
                (plan.get("selected_evidence") or []) or plan.get("route_type") == "chat"
                for plan in updated_plans
            ),
            "need_web_search": False,
            "events": events + [{
                "event": "websearch.skipped",
                "data": {"reason": "not_enabled"},
            }],
        }

    try:
        updated_plans: List[Dict[str, Any]] = []
        all_evidence: List[Dict[str, Any]] = []
        search_count = 0
        next_index = 1

        events.append({
            "event": "websearch.started",
            "data": {
                "sub_query_count": sum(1 for plan in sub_query_plans if plan.get("need_web_search")),
            },
        })

        for plan in sub_query_plans:
            current_plan = plan.copy()
            plan_evidence = list(current_plan.get("selected_evidence") or [])
            all_evidence.extend(plan_evidence)
            for item in plan_evidence:
                evidence_id = str(item.get("evidence_id", ""))
                if evidence_id.startswith("E") and evidence_id[1:].isdigit():
                    next_index = max(next_index, int(evidence_id[1:]) + 1)

        for plan in sub_query_plans:
            current_plan = plan.copy()
            if not current_plan.get("need_web_search"):
                updated_plans.append(current_plan)
                continue

            search_queries = current_plan.get("retrieval_queries") or [current_plan.get("query_rewritten") or current_plan.get("sub_question") or state.get("query", "")]
            merged_results: List[Dict[str, Any]] = []
            seen_urls = set()

            for query in search_queries[:3]:
                search_count += 1
                results = await web_search_service.search_with_snippets(query=query, max_results=4)
                for result in results:
                    url = result.get("url") or f"{query}:{result.get('title')}"
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged_results.append(result)

            web_evidence: List[Dict[str, Any]] = []
            for result in merged_results:
                snippet = result.get("snippet")
                web_evidence.append({
                    "evidence_id": f"E{next_index}",
                    "source_type": "web",
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "snippet": snippet,
                    "support_snippet": snippet,
                    "page_no": None,
                    "section_path": None,
                    "score": result.get("score", 0),
                    "sub_question": current_plan.get("sub_question"),
                })
                next_index += 1

            current_plan["selected_evidence"] = (current_plan.get("selected_evidence") or []) + web_evidence
            current_plan["need_web_search"] = False
            current_plan["need_retrieval"] = False
            current_plan["evidence_sufficient"] = len(current_plan["selected_evidence"]) > 0
            updated_plans.append(current_plan)

        all_evidence = []
        for plan in updated_plans:
            all_evidence.extend(plan.get("selected_evidence") or [])

        events.append({
            "event": "websearch.completed",
            "data": {"count": search_count, "sub_query_count": len(updated_plans)},
        })

        prompt_segments = []
        for plan in updated_plans:
            evidence = plan.get("selected_evidence") or []
            if not evidence:
                continue
            prompt_segments.append(
                f"子问题：{plan.get('sub_question')}\n" + "\n".join([
                    f"{item.get('evidence_id')} | {item.get('source_type')} | title={item.get('title', '')} | snippet={(item.get('support_snippet') or item.get('snippet') or '')[:220]}"
                    for item in evidence
                ])
            )

        return {
            "sub_query_plans": updated_plans,
            "selected_evidence": all_evidence,
            "prompt_context": "\n\n".join(prompt_segments),
            "evidence_sufficient": bool(updated_plans) and all((plan.get("selected_evidence") or []) for plan in updated_plans),
            "used_web_search": True,
            "need_web_search": False,
            "events": events,
        }
    except Exception as e:
        events.append({
            "event": "websearch.failed",
            "data": {"error": str(e)},
        })
        updated_plans, all_evidence = _close_web_attempt(
            sub_query_plans,
            error=str(e),
        )
        return {
            "sub_query_plans": updated_plans,
            "selected_evidence": all_evidence,
            "evidence_sufficient": bool(updated_plans) and all(
                (plan.get("selected_evidence") or []) or plan.get("route_type") == "chat"
                for plan in updated_plans
            ),
            "used_web_search": True,
            "need_web_search": False,
            "events": events,
        }
