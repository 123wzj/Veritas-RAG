# -*- coding: utf-8 -*-
"""
查询处理节点。
"""

from typing import Dict, Any, List
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state.state import RAGState
from backend.graph.llm_factory import get_llm


llm = get_llm()
logger = logging.getLogger(__name__)

CHAT_PATTERNS = (
    "你好", "您好", "hi", "hello", "在吗", "谢谢", "哈哈",
)
TIME_SENSITIVE_PATTERNS = (
    "今天", "最新", "最近", "实时", "当前", "刚刚", "天气", "新闻", "股价", "汇率",
)


# 安全获取模型返回的json数据
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


# 统一问题格式
def _dedupe_queries(queries: List[str]) -> List[str]:
    seen = set()
    results: List[str] = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", (query or "").strip())
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(normalized)
    return results



def _is_chat_like(query: str) -> bool:
    stripped = (query or "").strip().lower()
    if not stripped:
        return True
    if len(stripped) <= 12 and any(pattern in stripped for pattern in CHAT_PATTERNS):
        return True
    return False



def _is_time_sensitive(query: str) -> bool:
    normalized = (query or "").lower()
    return any(pattern in normalized for pattern in TIME_SENSITIVE_PATTERNS)



def _build_sub_query_plan(
    question: str,
    kb_id: int | None,
    web_enabled: bool,
    query_intent: str,
    fallback_queries: List[str] | None = None,
) -> Dict[str, Any]:
    normalized_question = " ".join((question or "").split())
    retrieval_queries = _dedupe_queries((fallback_queries or []) + [normalized_question])

    if _is_chat_like(normalized_question):
        route_type = "chat"
        need_retrieval = False
        need_web_search = False
    elif _is_time_sensitive(normalized_question):
        if kb_id is not None and web_enabled:
            route_type = "hybrid"
            need_retrieval = True
            need_web_search = True
        elif kb_id is not None:
            route_type = "knowledge_base"
            need_retrieval = True
            need_web_search = False
        elif web_enabled:
            route_type = "web_search"
            need_retrieval = False
            need_web_search = True
        else:
            route_type = "chat"
            need_retrieval = False
            need_web_search = False
    elif kb_id is not None:
        route_type = "knowledge_base"
        need_retrieval = True
        need_web_search = False
    elif web_enabled and query_intent != "chat":
        route_type = "web_search"
        need_retrieval = False
        need_web_search = True
    else:
        route_type = "chat"
        need_retrieval = False
        need_web_search = False

    return {
        "sub_question": normalized_question,
        "query_rewritten": normalized_question,
        "retrieval_queries": retrieval_queries,
        "route_type": route_type,
        "route_reason": f"sub_query_heuristic={route_type}",
        "need_retrieval": need_retrieval,
        "need_web_search": need_web_search,
        "planned_tools": [tool for tool, enabled in (("retrieve", need_retrieval), ("web_search", need_web_search)) if enabled],
        "retrieved_docs": [],
        "reranked_docs": [],
        "selected_evidence": [],
        "evidence_grade": None,
        "evidence_sufficient": route_type == "chat",
        "draft_answer": None,
        "answer": None,
        "citations": [],
        "confidence": 0.0,
    }


def _summarize_route_from_plans(sub_query_plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sub_query_plans:
        return {
            "route_type": "chat",
            "need_retrieval": False,
            "need_web_search": False,
            "planned_tools": [],
        }

    need_retrieval = any(plan.get("need_retrieval") for plan in sub_query_plans)
    need_web_search = any(plan.get("need_web_search") for plan in sub_query_plans)
    route_types = {plan.get("route_type") for plan in sub_query_plans if plan.get("route_type")}

    if route_types == {"chat"}:
        route_type = "chat"
    elif need_retrieval and need_web_search:
        route_type = "hybrid"
    elif need_retrieval:
        route_type = "knowledge_base"
    elif need_web_search:
        route_type = "web_search"
    elif "chat" in route_types and len(route_types) == 1:
        route_type = "chat"
    else:
        route_type = next(iter(route_types), "chat")

    planned_tools = []
    for tool in ("retrieve", "web_search"):
        if any(tool in (plan.get("planned_tools") or []) for plan in sub_query_plans):
            planned_tools.append(tool)

    return {
        "route_type": route_type,
        "need_retrieval": need_retrieval,
        "need_web_search": need_web_search,
        "planned_tools": planned_tools,
    }


async def rewrite_query(state: RAGState) -> Dict[str, Any]:
    """
    标准化查询表达，补全适合检索的关键约束。
    """
    query = state.get("query", "")

    memory_context = state.get("memory_context") or {}
    profile = memory_context.get("profile", {})
    preferred_language = (
        memory_context.get("preferred_language")
        or profile.get("preferred_language")
        or "zh-CN"
    )
    interests = (
        memory_context.get("interests")
        or profile.get("interests")
        or []
    )
    session_summary = memory_context.get("session_summary") or ""
    recent_conversations = memory_context.get("recent_conversations") or []
    long_term_facts = memory_context.get("long_term_facts") or []

    system_prompt = """你是企业知识库检索优化助手。请把用户问题改写成更适合检索的表达。
输出 JSON：
{
  "rewritten_query": "...",
  "search_intent": "fact|comparison|procedure|summary|analysis|chat",
  "missing_facets": ["..."]
}

要求：
1. 保持原意，不要过度扩写
2. 优先补足实体名、限定条件、时间范围、目标对象
3. 兼顾 dense 检索和关键词检索
4. 如果本来就是闲聊或非常短的问候，可以保持原样
"""

    user_prompt = f"""原始问题：{query}
用户语言偏好：{preferred_language}
用户兴趣：{interests}
会话摘要：{session_summary or '无'}
相关历史：{recent_conversations[:3] if recent_conversations else []}
长期事实：{long_term_facts[:4] if long_term_facts else []}
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        parsed = _safe_load_json(response.content)
        query_rewritten = parsed.get("rewritten_query") or query
        search_intent = parsed.get("search_intent") or ("chat" if _is_chat_like(query) else "analysis")
        missing_facets = parsed.get("missing_facets") or []

        return {
            "query_rewritten": query_rewritten,
            "query_intent": search_intent,
            "retrieval_queries": _dedupe_queries([query_rewritten, query]),
            "reasoning_trace_summary": f"query_intent={search_intent}; missing_facets={missing_facets}",
            "events": state.get("events", []) + [{
                "event": "query.rewritten",
                "data": {
                    "original": query,
                    "rewritten": query_rewritten,
                    "intent": search_intent,
                },
            }],
        }
    except Exception as e:
        logger.exception("LLM failed in rewrite_query: query=%s", query)
        fallback_intent = "chat" if _is_chat_like(query) else "analysis"
        return {
            "query_rewritten": query,
            "query_intent": fallback_intent,
            "retrieval_queries": [query],
            "events": state.get("events", []) + [{
                "event": "query.rewrite.failed",
                "data": {"error": str(e)},
            }],
        }


async def decompose_query(state: RAGState) -> Dict[str, Any]:
    """
    将复杂问题拆成多个可检索子问题和检索表达。
    """
    query = state.get("query_rewritten") or state.get("query", "")
    original_query = state.get("query", "")
    query_intent = state.get("query_intent") or "analysis"
    kb_id = state.get("kb_id")
    web_enabled = state.get("web_enabled", False)

    if query_intent == "chat":
        base_queries = _dedupe_queries([query, original_query])
        return {
            "sub_questions": [],
            "retrieval_queries": base_queries,
            "sub_query_plans": [_build_sub_query_plan(query, kb_id, web_enabled, query_intent, base_queries)],
        }

    needs_decomposition = any([
        "和" in query and len(query) > 24,
        "以及" in query,
        "还有" in query,
        "分别" in query,
        "对比" in query,
        query.count("？") + query.count("?") > 1,
    ])

    if not needs_decomposition:
        base_queries = _dedupe_queries(state.get("retrieval_queries") or [query, original_query])
        return {
            "sub_questions": [],
            "retrieval_queries": base_queries,
            "sub_query_plans": [_build_sub_query_plan(query, kb_id, web_enabled, query_intent, base_queries)],
        }

    system_prompt = """你是查询规划助手。请把复杂问题拆成可独立检索的子问题，并生成检索表达。
输出 JSON：
{
  "sub_questions": ["...", "..."],
  "retrieval_queries": ["...", "..."],
  "notes": "..."
}

要求：
1. 子问题控制在 2 到 4 个
2. 检索表达比子问题更适合搜索
3. 保留原问题中的比较维度、对象范围和时间约束
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"问题：{query}"),
        ])
        result = _safe_load_json(response.content)
        sub_questions = _dedupe_queries(result.get("sub_questions") or [])
        retrieval_queries = _dedupe_queries(
            (state.get("retrieval_queries") or [])
            + result.get("retrieval_queries", [])
            + sub_questions
            + [query, original_query]
        )
        plans = [
            _build_sub_query_plan(sub_question, kb_id, web_enabled, query_intent, [sub_question])
            for sub_question in sub_questions
        ]
        return {
            "sub_questions": sub_questions,
            "retrieval_queries": retrieval_queries,
            "sub_query_plans": plans,
            "reasoning_trace_summary": (state.get("reasoning_trace_summary") or "") + f"; decomposition_notes={result.get('notes', '')}",
            "events": state.get("events", []) + [{
                "event": "query.decomposed",
                "data": {
                    "count": len(sub_questions),
                    "questions": sub_questions,
                    "retrieval_queries": retrieval_queries,
                },
            }],
        }
    except Exception as e:
        logger.exception("LLM failed in decompose_query: query=%s", query)
        base_queries = _dedupe_queries((state.get("retrieval_queries") or []) + [query, original_query])
        return {
            "sub_questions": [],
            "retrieval_queries": base_queries,
            "sub_query_plans": [_build_sub_query_plan(query, kb_id, web_enabled, query_intent, base_queries)],
            "events": state.get("events", []) + [{
                "event": "query.decompose.failed",
                "data": {"error": str(e)},
            }],
        }


async def plan_query_route(state: RAGState) -> Dict[str, Any]:
    """
    基于意图、知识库上下文和联网能力，规划查询路由。
    """
    query = state.get("query", "")
    query_rewritten = state.get("query_rewritten") or query
    query_intent = state.get("query_intent") or "analysis"
    kb_id = state.get("kb_id")
    web_enabled = state.get("web_enabled", False)
    sub_questions = state.get("sub_questions") or []
    existing_plans = state.get("sub_query_plans") or []

    heuristic_route = "knowledge_base" if kb_id is not None else "chat"
    if _is_chat_like(query):
        heuristic_route = "chat"
    elif _is_time_sensitive(query_rewritten) and web_enabled and kb_id is None:
        heuristic_route = "web_search"
    elif _is_time_sensitive(query_rewritten) and web_enabled and kb_id is not None:
        heuristic_route = "hybrid"

    system_prompt = """你是 Agentic RAG 路由规划器。请判断当前问题最适合走哪条路由。
输出 JSON：
{
  "route_type": "knowledge_base|web_search|chat|hybrid",
  "route_reason": "...",
  "need_retrieval": true,
  "need_web_search": false,
  "planned_tools": ["rewrite", "retrieve", "rerank"]
}

规则：
1. 闲聊、问候、非知识型对话优先 chat
2. 明显依赖最新信息且允许联网时优先 web_search 或 hybrid
3. 知识库内问题优先 knowledge_base
4. hybrid 表示先走知识库检索，再结合联网补充
"""

    user_prompt = f"""原始问题：{query}
改写问题：{query_rewritten}
意图：{query_intent}
子问题：{sub_questions or []}
是否有知识库：{kb_id is not None}
是否允许联网：{web_enabled}
启发式建议：{heuristic_route}
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result = _safe_load_json(response.content)
    except Exception:
        logger.exception("LLM failed in plan_query_route: query=%s", query)
        result = {}

    route_type = result.get("route_type") or heuristic_route
    if route_type not in {"knowledge_base", "web_search", "chat", "hybrid"}:
        route_type = heuristic_route

    need_retrieval = bool(result.get("need_retrieval", route_type in {"knowledge_base", "hybrid"} and kb_id is not None))
    need_web_search = bool(result.get("need_web_search", route_type in {"web_search", "hybrid"} and web_enabled))

    if route_type == "chat":
        need_retrieval = False
        need_web_search = False
    elif route_type == "web_search":
        need_retrieval = False
        need_web_search = web_enabled
    elif route_type == "hybrid":
        need_retrieval = kb_id is not None
        need_web_search = web_enabled
    else:
        need_retrieval = kb_id is not None
        need_web_search = False

    sub_query_plans = [
        plan.copy() for plan in (
            existing_plans or [
                _build_sub_query_plan(query_rewritten, kb_id, web_enabled, query_intent, state.get("retrieval_queries") or [query_rewritten, query])
            ]
        )
    ]

    if route_type in {"hybrid", "web_search", "knowledge_base"}:
        adjusted_plans = []
        for plan in sub_query_plans:
            current_plan = plan.copy()
            if current_plan.get("route_type") == "chat":
                adjusted_plans.append(current_plan)
                continue

            if route_type == "hybrid":
                current_plan["route_type"] = "hybrid"
                current_plan["need_retrieval"] = kb_id is not None
                current_plan["need_web_search"] = web_enabled
            elif route_type == "web_search":
                current_plan["route_type"] = "web_search"
                current_plan["need_retrieval"] = False
                current_plan["need_web_search"] = web_enabled
            elif route_type == "knowledge_base":
                current_plan["route_type"] = "knowledge_base"
                current_plan["need_retrieval"] = kb_id is not None
                current_plan["need_web_search"] = False

            current_plan["planned_tools"] = [
                tool
                for tool, enabled in (
                    ("retrieve", current_plan["need_retrieval"]),
                    ("web_search", current_plan["need_web_search"]),
                )
                if enabled
            ]
            adjusted_plans.append(current_plan)
        sub_query_plans = adjusted_plans

    route_summary = _summarize_route_from_plans(sub_query_plans)
    planned_tools = route_summary["planned_tools"]
    route_type = route_summary["route_type"]
    need_retrieval = route_summary["need_retrieval"]
    need_web_search = route_summary["need_web_search"]

    llm_route_type = result.get("route_type")
    if llm_route_type and llm_route_type in {"knowledge_base", "web_search", "chat", "hybrid"}:
        route_reason = result.get("route_reason") or f"llm={llm_route_type}; plan_summary={route_type}"
    else:
        route_reason = result.get("route_reason") or f"heuristic={heuristic_route}; plan_summary={route_type}"

    return {
        "route_type": route_type,
        "route_reason": route_reason,
        "need_retrieval": need_retrieval,
        "need_web_search": need_web_search,
        "planned_tools": planned_tools,
        "sub_query_plans": sub_query_plans,
        "events": state.get("events", []) + [{
            "event": "route.planned",
            "data": {
                "route_type": route_type,
                "need_retrieval": need_retrieval,
                "need_web_search": need_web_search,
                "planned_tools": planned_tools,
                "sub_query_count": len(sub_query_plans),
            },
        }],
    }


async def route_planned_query(state: RAGState) -> str:
    sub_query_plans = state.get("sub_query_plans") or []
    if any(plan.get("need_retrieval") for plan in sub_query_plans):
        return "retrieve"
    if any(plan.get("need_web_search") for plan in sub_query_plans):
        return "web_search"

    route_type = state.get("route_type") or "knowledge_base"
    if route_type == "chat":
        return "generate"
    if route_type == "web_search" and state.get("need_web_search", False):
        return "web_search"
    if state.get("need_retrieval", False):
        return "retrieve"
    if state.get("need_web_search", False):
        return "web_search"
    return "generate"
