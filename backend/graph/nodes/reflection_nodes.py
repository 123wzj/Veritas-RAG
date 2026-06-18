# -*- coding: utf-8 -*-
"""
反思与纠错节点
"""

from typing import Dict, Any, List
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state.state import RAGState
from backend.graph.llm_factory import get_llm


llm = get_llm()


def _safe_load_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _dedupe_queries(queries: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for query in queries:
        normalized = " ".join((query or "").split())
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


async def reflection(state: RAGState) -> Dict[str, Any]:
    """
    反思节点
    根据证据评审和答案验证结果生成下一轮检索策略。
    """
    query = state.get("query", "")
    query_rewritten = state.get("query_rewritten") or query
    evidence_grade = state.get("evidence_grade") or {}
    verification = state.get("verification") or {}
    reflection_count = state.get("reflection_count", 0)
    memory_context = state.get("memory_context") or {}
    prompt_context = memory_context.get("prompt_context") or ""
    sub_query_plans = state.get("sub_query_plans") or []

    updated_plans = []
    reflection_notes: List[str] = []
    retrieval_needed = False
    web_needed = False

    for plan in sub_query_plans:
        current_plan = plan.copy()
        if current_plan.get("evidence_sufficient"):
            updated_plans.append(current_plan)
            continue

        sub_question = current_plan.get("sub_question") or query
        plan_query = current_plan.get("query_rewritten") or sub_question
        retrieved_docs = current_plan.get("retrieved_docs") or []
        selected_evidence = current_plan.get("selected_evidence") or []
        plan_grade = current_plan.get("evidence_grade") or {}

        results_summary = "\n".join([
            f"- title={doc.get('title', '')}, score={doc.get('rerank_score', doc.get('rrf_score', doc.get('score', 0))):.3f}, "
            f"query={doc.get('match_query', '')}, snippet={(doc.get('content') or '')[:120]}"
            for doc in retrieved_docs[:6]
        ])
        evidence_summary = "\n".join([
            f"- {item.get('evidence_id')}: {item.get('title', '')} | {(item.get('support_snippet') or item.get('snippet') or '')[:100]}"
            for item in selected_evidence[:4]
        ])

        system_prompt = """你是 RAG 子问题反思规划器。你需要根据当前子问题的检索、证据评审和答案验证结果，给出下一轮策略。

输出 JSON：
{
  "should_retry_retrieval": true,
  "new_query": "改进后的子问题查询，可选",
  "additional_queries": ["补充检索词1", "补充检索词2"],
  "need_web_search": false,
  "focus": "下一轮重点补哪部分",
  "reason": "简短原因"
}

要求：
1. 只针对当前子问题规划，不要按整体问题输出
2. 如果当前子问题缺的是实时或外部公开信息，可设置 need_web_search=true
3. 补充检索词要更具体，不要只是重复原句
4. 如果证据基本够，只是答案表达不稳，可以不重检索"""

        user_prompt = f"""原始问题：{query}
当前子问题：{sub_question}
当前子问题查询：{plan_query}
会话压缩上下文：{prompt_context or '无'}

整体证据评审：{evidence_grade}
当前子问题证据评审：{plan_grade}
答案验证：{verification}

检索摘要：
{results_summary or "无"}

证据摘要：
{evidence_summary or "无"}

请输出下一步策略。"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            result = _safe_load_json(response.content)
        except Exception as e:
            result = {"should_retry_retrieval": False, "reason": str(e)}

        new_query = result.get("new_query") or plan_query
        additional_queries = result.get("additional_queries") or []
        deduped_queries = _dedupe_queries([new_query] + additional_queries + current_plan.get("retrieval_queries", []) + [sub_question])
        should_retry = bool(result.get("should_retry_retrieval")) and bool(deduped_queries)
        need_web_search = bool(result.get("need_web_search")) and state.get("web_enabled", False)
        need_retrieval = should_retry and not need_web_search and state.get("kb_id") is not None

        current_plan["query_rewritten"] = new_query
        current_plan["retrieval_queries"] = deduped_queries
        current_plan["need_web_search"] = need_web_search
        current_plan["need_retrieval"] = need_retrieval

        if current_plan["need_retrieval"]:
            current_plan["route_type"] = "knowledge_base" if state.get("kb_id") is not None else current_plan.get("route_type")
            current_plan["retrieved_docs"] = []
            current_plan["reranked_docs"] = []
            current_plan["selected_evidence"] = []
        if need_web_search:
            current_plan["route_type"] = "hybrid" if state.get("kb_id") is not None else "web_search"

        updated_plans.append(current_plan)
        retrieval_needed = retrieval_needed or current_plan["need_retrieval"]
        web_needed = web_needed or current_plan["need_web_search"]
        reflection_notes.append(
            f"{sub_question}: focus={result.get('focus', '')}; reason={result.get('reason', '')}"
        )

    if not sub_query_plans:
        deduped_queries = _dedupe_queries([query_rewritten, query])
        updated_plans = []
        retrieval_needed = False
        web_needed = False

    return {
        "reflection_notes": " | ".join(note for note in reflection_notes if note),
        "reflection_count": reflection_count + 1,
        "need_reflection": retrieval_needed and not web_needed,
        "need_retrieval": retrieval_needed,
        "need_web_search": web_needed,
        "query_rewritten": query_rewritten,
        "sub_query_plans": updated_plans or sub_query_plans,
        "events": state.get("events", []) + [{
            "event": "reflection.completed",
            "data": {
                "reflection_count": reflection_count + 1,
                "focus": " | ".join(note for note in reflection_notes if note),
                "retry_sub_query_count": sum(1 for plan in (updated_plans or sub_query_plans) if plan.get("need_retrieval")),
                "web_sub_query_count": sum(1 for plan in (updated_plans or sub_query_plans) if plan.get("need_web_search")),
            },
        }],
    }


async def route_reflection(state: RAGState) -> str:
    """
    反思路由节点
    """
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 2)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 8)
    sub_query_plans = state.get("sub_query_plans") or []

    if reflection_count >= max_reflections or step_count >= max_steps:
        return "proceed"

    if any(plan.get("need_retrieval") for plan in sub_query_plans):
        return "retrieve"
    if any(plan.get("need_web_search") for plan in sub_query_plans):
        return "web_search"
    return "proceed"


async def route_evidence(state: RAGState) -> str:
    """
    证据路由节点
    """
    evidence_sufficient = state.get("evidence_sufficient", False)
    need_reflection = state.get("need_reflection", False)
    need_web_search = state.get("need_web_search", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 2)
    sub_query_plans = state.get("sub_query_plans") or []

    if any(plan.get("need_web_search") for plan in sub_query_plans) or need_web_search:
        return "web_search"
    if evidence_sufficient:
        return "generate"
    if reflection_count >= max_reflections:
        return "generate"
    if any(plan.get("need_retrieval") for plan in sub_query_plans) or need_reflection:
        return "reflect"
    return "generate"
