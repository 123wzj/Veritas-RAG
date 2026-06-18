# -*- coding: utf-8 -*-
"""
答案生成与验证节点。
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



def _extract_cited_evidence_ids(answer: str) -> List[str]:
    return sorted(set(re.findall(r"\[(E\d+)\]", answer or "")))



def _build_citations(
    selected_evidence: List[Dict[str, Any]],
    cited_ids: List[str],
) -> List[Dict[str, Any]]:
    evidence_map = {item["evidence_id"]: item for item in selected_evidence}
    citations = []
    seen = set()
    ids = cited_ids or [item["evidence_id"] for item in selected_evidence[:3]]
    for evidence_id in ids:
        item = evidence_map.get(evidence_id)
        if not item:
            continue
        dedupe_key = (item.get("doc_id"), item.get("parent_id") or item.get("chunk_id"), evidence_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        citations.append({
            "source_type": item.get("source_type", "knowledge_base"),
            "doc_id": item.get("doc_id"),
            "chunk_id": item.get("chunk_id"),
            "parent_id": item.get("parent_id"),
            "title": item.get("title"),
            "page_no": item.get("page_no"),
            "section_path": item.get("section_path"),
            "url": item.get("url"),
            "snippet": item.get("support_snippet") or item.get("snippet", "")[:200],
            "score": item.get("score"),
            "evidence_id": evidence_id,
        })
    return citations


async def _generate_chat_response(
    query: str,
    preferred_language: str,
    interaction_style: str,
    session_summary: str,
) -> Dict[str, Any]:
    system_prompt = """你是友好、直接的中文助手。
当前问题被路由为闲聊或通用对话，不需要引用知识库证据。
请直接回答，不要编造检索来源，不要输出 JSON 以外的内容。

输出 JSON：
{
  "answer": "...",
  "confidence": 0-1
}
"""
    user_prompt = f"""用户问题：{query}
用户语言偏好：{preferred_language}
回答风格：{interaction_style}
会话摘要：{session_summary or '无'}
"""
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    parsed = _safe_load_json(response.content)
    return {
        "final_answer": parsed.get("answer") or "你好，我在。",
        "citations": [],
        "confidence": round(float(parsed.get("confidence", 0.85)), 2),
    }


async def _generate_sub_answer(
    sub_question: str,
    route_type: str,
    selected_evidence: List[Dict[str, Any]],
    preferred_language: str,
    interaction_style: str,
    session_summary: str,
    prompt_context: str,
    working_memory: List[Any],
    long_term_facts: List[Any],
) -> Dict[str, Any]:
    if route_type == "chat" and not selected_evidence:
        result = await _generate_chat_response(sub_question, preferred_language, interaction_style, session_summary)
        return {
            "sub_question": sub_question,
            "answer": result["final_answer"],
            "citations": [],
            "confidence": result["confidence"],
            "reasoning_summary": "chat_route",
        }

    evidence_text = "\n\n".join([
        f"{ev['evidence_id']}\n"
        f"title={ev.get('title', '')}\n"
        f"page={ev.get('page_no')}\n"
        f"section={ev.get('section_path')}\n"
        f"snippet={((ev.get('support_snippet') or ev.get('snippet') or '')[:320])}"
        for ev in selected_evidence[:6]
    ])

    system_prompt = """你是企业知识库问答助手。你必须严格依据当前子问题对应的证据作答。
输出 JSON：
{
  "answer": "最终答案，关键句必须带 [E1][E2] 这种内联引用",
  "used_evidence_ids": ["E1", "E3"],
  "confidence": 0-1,
  "reasoning_summary": "一句话说明答案主要建立在哪些证据上"
}

要求：
1. 只能引用给定证据编号
2. 关键结论必须带 [E#]
3. 如果证据不足，要明确说明不足
4. 只回答当前子问题，不要扩展到其他子问题
"""

    user_prompt = f"""当前子问题：{sub_question}
路由类型：{route_type}
用户语言偏好：{preferred_language}
回答风格：{interaction_style}
会话摘要：{session_summary or '无'}
长期事实：{long_term_facts[:4] if long_term_facts else []}
工作记忆：{working_memory[:4] if working_memory else []}
补充上下文：{prompt_context or '无'}

证据：
{evidence_text}
"""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    parsed = _safe_load_json(response.content)
    answer = parsed.get("answer") or "当前证据不足，暂时无法给出可靠答案。"
    used_evidence_ids = parsed.get("used_evidence_ids") or _extract_cited_evidence_ids(answer)
    citations = _build_citations(selected_evidence, used_evidence_ids)
    confidence = float(parsed.get("confidence", _calculate_confidence(selected_evidence, citations)))
    return {
        "sub_question": sub_question,
        "answer": answer,
        "citations": citations,
        "confidence": round(min(max(confidence, 0.0), 1.0), 2),
        "reasoning_summary": parsed.get("reasoning_summary") or "",
    }


async def generate_answer(state: RAGState) -> Dict[str, Any]:
    query = state.get("query", "")
    selected_evidence = state.get("selected_evidence", [])
    memory_context = state.get("memory_context", {}) or {}
    profile = memory_context.get("profile", {})
    preferred_language = (
        memory_context.get("preferred_language")
        or profile.get("preferred_language")
        or "zh-CN"
    )
    interaction_style = (
        memory_context.get("interaction_style")
        or profile.get("interaction_style")
        or "detailed"
    )
    prompt_context = state.get("prompt_context") or memory_context.get("prompt_context") or ""
    session_summary = state.get("session_summary") or memory_context.get("session_summary") or ""
    working_memory = memory_context.get("working_memory") or []
    long_term_facts = memory_context.get("long_term_facts") or []
    sub_query_plans = state.get("sub_query_plans") or []

    if not sub_query_plans:
        route_type = state.get("route_type") or "knowledge_base"
        if route_type == "chat" and not selected_evidence:
            try:
                result = await _generate_chat_response(query, preferred_language, interaction_style, session_summary)
                return {
                    **result,
                    "events": state.get("events", []) + [{
                        "event": "answer.completed",
                        "data": {"citations_count": 0, "route_type": route_type},
                    }],
                }
            except Exception as e:
                logger.exception("LLM failed in chat mode: query=%s", query)
                return {
                    "final_answer": f"抱歉，当前无法完成回答：{str(e)}",
                    "citations": [],
                    "confidence": 0.0,
                    "error": str(e),
                }

        if not selected_evidence:
            return {
                "final_answer": "当前没有足够证据支撑回答这个问题。",
                "citations": [],
                "confidence": 0.0,
            }

    try:
        sub_answers: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, Any]] = []
        confidence_values: List[float] = []
        answer_segments: List[str] = []
        reasoning_segments: List[str] = []
        updated_plans: List[Dict[str, Any]] = []

        for plan in sub_query_plans:
            current_plan = plan.copy()
            result = await _generate_sub_answer(
                sub_question=current_plan.get("sub_question") or query,
                route_type=current_plan.get("route_type") or state.get("route_type") or "knowledge_base",
                selected_evidence=current_plan.get("selected_evidence") or [],
                preferred_language=preferred_language,
                interaction_style=interaction_style,
                session_summary=session_summary,
                prompt_context=prompt_context,
                working_memory=working_memory,
                long_term_facts=long_term_facts,
            )
            current_plan["answer"] = result["answer"]
            current_plan["citations"] = result["citations"]
            current_plan["confidence"] = result["confidence"]
            updated_plans.append(current_plan)
            sub_answers.append(result)
            all_citations.extend(result["citations"])
            confidence_values.append(result["confidence"])
            reasoning_segments.append(f"{result['sub_question']}: {result.get('reasoning_summary', '')}".strip())
            if len(sub_query_plans) > 1:
                answer_segments.append(f"关于“{result['sub_question']}”：{result['answer']}")
            else:
                answer_segments.append(result["answer"])

        final_answer = "\n\n".join(answer_segments)
        confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0

        return {
            "final_answer": final_answer,
            "citations": all_citations,
            "confidence": confidence,
            "sub_query_plans": updated_plans,
            "sub_query_answers": sub_answers,
            "reasoning_trace_summary": " | ".join(segment for segment in reasoning_segments if segment),
            "events": state.get("events", []) + [{
                "event": "answer.completed",
                "data": {"citations_count": len(all_citations), "sub_query_count": len(sub_answers)},
            }],
        }
    except Exception as e:
        logger.exception("LLM failed in generate_answer: query=%s evidence_count=%s", query, len(selected_evidence))
        fallback_citations = _build_citations(selected_evidence, [])
        return {
            "final_answer": f"抱歉，生成答案时出错：{str(e)}",
            "citations": fallback_citations,
            "confidence": 0.0,
            "error": str(e),
        }



def _calculate_confidence(
    evidence: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> float:
    if not evidence:
        return 0.0

    avg_score = sum(float(ev.get("score", 0.0)) for ev in evidence) / len(evidence)
    citation_coverage = len(citations) / max(1, min(4, len(evidence)))
    confidence = avg_score * 0.6 + min(citation_coverage, 1.0) * 0.4
    return round(min(max(confidence, 0.0), 1.0), 2)


async def verify_answer(state: RAGState) -> Dict[str, Any]:
    final_answer = state.get("final_answer", "")
    selected_evidence = state.get("selected_evidence", [])
    query = state.get("query", "")
    route_type = state.get("route_type") or "knowledge_base"
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 3)
    sub_query_plans = state.get("sub_query_plans") or []

    if route_type == "chat" and not selected_evidence:
        return {
            "confidence": state.get("confidence", 0.85),
            "verification": {"grounded": True, "useful": True, "reason": "chat_route"},
            "need_reflection": False,
        }

    if not final_answer or (sub_query_plans and not all(plan.get("answer") for plan in sub_query_plans)):
        verification = {
            "grounded": False,
            "useful": False,
            "reason": "存在未完成的子问题答案",
        }
        return {
            "confidence": 0.0,
            "verification": verification,
            "need_reflection": reflection_count < max_reflections,
        }

    if sub_query_plans and not all((plan.get("selected_evidence") or []) or plan.get("route_type") == "chat" for plan in sub_query_plans):
        if state.get("used_web_search"):
            return {
                "confidence": state.get("confidence", 0.0),
                "verification": {
                    "grounded": True,
                    "useful": True,
                    "reason": "web_search_exhausted_no_evidence",
                },
                "need_reflection": False,
            }
        return {
            "confidence": 0.0,
            "verification": {
                "grounded": False,
                "useful": False,
                "reason": "存在未覆盖的子问题证据",
            },
            "need_reflection": reflection_count < max_reflections,
        }

    evidence_text = "\n\n".join([
        f"{item['evidence_id']}: {item.get('support_snippet') or item.get('snippet', '')[:220]}"
        for item in selected_evidence[:10]
    ])

    system_prompt = """你是答案验证器。请判断答案是否被证据支撑，以及是否回答了问题。
输出 JSON：
{
  "grounded": true,
  "useful": true,
  "missing_or_unsupported": ["..."],
  "reason": "...",
  "confidence_adjustment": 0-1
}
"""

    user_prompt = f"""问题：{query}

证据：
{evidence_text}

答案：
{final_answer}
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        verification = _safe_load_json(response.content)
    except Exception:
        logger.exception("LLM failed in verify_answer: query=%s", query)
        verification = {}

    if verification:
        grounded = bool(verification.get("grounded"))
        useful = bool(verification.get("useful"))
        adjustment = float(verification.get("confidence_adjustment", 0.75 if grounded else 0.35))
        original_confidence = state.get("confidence", 0.5)
        adjusted_confidence = round(min(max(original_confidence * adjustment, 0.0), 1.0), 2)
        need_reflection = (not grounded or not useful) and reflection_count < max_reflections
        return {
            "confidence": adjusted_confidence,
            "verification": verification,
            "need_reflection": need_reflection,
        }

    cited_ids = _extract_cited_evidence_ids(final_answer)
    citations = state.get("citations", [])
    citation_coverage = len(cited_ids or citations) / max(1, min(4, len(selected_evidence)))
    original_confidence = state.get("confidence", 0.5)
    adjusted_confidence = round(original_confidence * (0.7 + min(citation_coverage, 1.0) * 0.3), 2)
    need_reflection = citation_coverage < 0.5 and reflection_count < max_reflections
    return {
        "confidence": adjusted_confidence,
        "verification": {
            "grounded": not need_reflection,
            "useful": True,
            "reason": "fallback_verification",
        },
        "need_reflection": need_reflection,
    }


async def route_verification(state: RAGState) -> str:
    verification = state.get("verification") or {}
    need_reflection = state.get("need_reflection", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 3)

    if need_reflection and reflection_count < max_reflections:
        return "reflect"

    if verification and (not verification.get("grounded", True) or not verification.get("useful", True)):
        return "reflect" if reflection_count < max_reflections else "write_memory"

    return "write_memory"
