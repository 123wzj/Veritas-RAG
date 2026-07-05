# -*- coding: utf-8 -*-
"""
检索相关节点
"""

from typing import Dict, Any, List
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state.state import RAGState
from backend.graph.llm_factory import get_llm
from backend.services.retrieval.hybrid import hybrid_retriever
from backend.services.retrieval.diversity import select_diverse_results
from backend.services.retrieval.reranker import reranker


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


async def retrieve_hybrid(state: RAGState) -> Dict[str, Any]:
    """
    混合检索节点

    对每个需要知识库检索的子问题独立执行多查询检索。
    """
    user_id = state.get("user_id", 0)
    kb_id = state.get("kb_id")
    top_k = state.get("top_k", 6)
    sub_query_plans = state.get("sub_query_plans") or []

    if kb_id is None or not sub_query_plans:
        return {"retrieved_docs": [], "sub_query_plans": sub_query_plans}

    try:
        all_docs: List[Dict[str, Any]] = []
        updated_plans: List[Dict[str, Any]] = []
        active_queries = []

        for plan in sub_query_plans:
            current_plan = plan.copy()
            if not current_plan.get("need_retrieval"):
                updated_plans.append(current_plan)
                continue

            query = current_plan.get("query_rewritten") or current_plan.get("sub_question") or state.get("query", "")
            retrieval_queries = current_plan.get("retrieval_queries") or [query]
            active_queries.append({
                "sub_question": current_plan.get("sub_question"),
                "queries": retrieval_queries,
            })
            results = await hybrid_retriever.retrieve_async(
                query=query,
                user_id=user_id,
                kb_id=kb_id,
                query_variants=retrieval_queries,
                top_k=max(top_k * 3, 12),
            )
            current_plan["retrieved_docs"] = results
            all_docs.extend(results)
            updated_plans.append(current_plan)

        return {
            "retrieved_docs": all_docs,
            "sub_query_plans": updated_plans,
            "events": state.get("events", []) + [
                {"event": "retrieval.started", "data": {"queries": active_queries}},
                {
                    "event": "retrieval.completed",
                    "data": {
                        "count": len(all_docs),
                        "sub_query_count": len(active_queries),
                    },
                },
            ],
            "step_count": state.get("step_count", 0) + 1,
        }
    except Exception as e:
        return {
            "error": f"检索失败: {str(e)}",
            "events": state.get("events", []) + [{
                "event": "retrieval.failed",
                "data": {"error": str(e)},
            }],
        }


async def rerank_candidates(state: RAGState) -> Dict[str, Any]:
    """
    重排候选节点
    """
    top_k = state.get("top_k", 6)
    sub_query_plans = state.get("sub_query_plans") or []

    if not sub_query_plans:
        return {"reranked_docs": []}

    reranked_docs: List[Dict[str, Any]] = []
    updated_plans: List[Dict[str, Any]] = []

    for plan in sub_query_plans:
        current_plan = plan.copy()
        retrieved_docs = current_plan.get("retrieved_docs") or []
        query = current_plan.get("query_rewritten") or current_plan.get("sub_question") or state.get("query", "")

        if not retrieved_docs:
            current_plan["reranked_docs"] = []
            updated_plans.append(current_plan)
            continue

        try:
            ranked = reranker.rerank(
                query=query,
                documents=[doc.copy() for doc in retrieved_docs],
                top_k=max(top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )
        except Exception:
            ranked = select_diverse_results(
                [doc.copy() for doc in retrieved_docs],
                top_k=max(top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )

        for idx, doc in enumerate(ranked):
            doc.setdefault("rank", idx)
            doc["sub_question"] = current_plan.get("sub_question")

        current_plan["reranked_docs"] = ranked
        reranked_docs.extend(ranked)
        updated_plans.append(current_plan)

    return {
        "reranked_docs": reranked_docs,
        "sub_query_plans": updated_plans,
        "events": state.get("events", []) + [{
            "event": "rerank.completed",
            "data": {"count": len(reranked_docs), "sub_query_count": len(updated_plans)},
        }],
    }


async def pack_evidence(state: RAGState) -> Dict[str, Any]:
    """
    打包证据节点

    把每个子问题的 rerank 结果打包为可引用证据结构。
    """
    top_k = state.get("top_k", 6)
    updated_plans: List[Dict[str, Any]] = []
    all_evidence: List[Dict[str, Any]] = []
    prompt_segments: List[str] = []
    evidence_index = 1

    for plan in state.get("sub_query_plans") or []:
        current_plan = plan.copy()
        reranked_docs = current_plan.get("reranked_docs") or []
        evidence: List[Dict[str, Any]] = []
        packing_candidates = select_diverse_results(
            reranked_docs,
            top_k=top_k,
            max_per_doc=3,
            max_per_parent=1,
        )

        for doc in packing_candidates:
            parent_content = doc.get("parent_content") or ""
            child_content = doc.get("content") or ""
            snippet = parent_content[:1200] if parent_content else child_content[:600]
            support_snippet = child_content[:400]
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))

            item = {
                "evidence_id": f"E{evidence_index}",
                "source_type": "knowledge_base",
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "parent_id": doc.get("parent_id"),
                "title": doc.get("parent_title") or doc.get("title") or "",
                "page_no": doc.get("parent_page_no") or doc.get("page_no"),
                "section_path": doc.get("parent_section_path") or doc.get("section_path"),
                "snippet": snippet,
                "support_snippet": support_snippet,
                "score": round(float(score or 0.0), 6),
                "match_query": doc.get("match_query"),
                "retrieval_type": doc.get("retrieval_type"),
                "sub_question": current_plan.get("sub_question"),
            }
            evidence_index += 1
            evidence.append(item)

            if len(evidence) >= top_k:
                break

        current_plan["selected_evidence"] = evidence
        current_plan["evidence_sufficient"] = len(evidence) > 0
        updated_plans.append(current_plan)
        all_evidence.extend(evidence)
        if evidence:
            prompt_segments.append(
                f"子问题：{current_plan.get('sub_question')}\n" + "\n".join([
                    f"{item['evidence_id']} | title={item.get('title', '')} | section={item.get('section_path')} | snippet={(item.get('support_snippet') or item.get('snippet', ''))[:220]}"
                    for item in evidence
                ])
            )

    return {
        "selected_evidence": all_evidence,
        "sub_query_plans": updated_plans,
        "evidence_sufficient": bool(updated_plans) and all(plan.get("evidence_sufficient") for plan in updated_plans),
        "prompt_context": "\n\n".join(prompt_segments),
    }


async def judge_evidence(state: RAGState) -> Dict[str, Any]:
    """
    判断证据是否足够。

    强约束：每个子问题都必须被覆盖。
    """
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 2)
    web_enabled = state.get("web_enabled", False)
    used_web_search = state.get("used_web_search", False)
    updated_plans: List[Dict[str, Any]] = []
    uncovered_questions: List[str] = []
    evidence_grades: List[Dict[str, Any]] = []

    for plan in state.get("sub_query_plans") or []:
        current_plan = plan.copy()
        selected_evidence = current_plan.get("selected_evidence") or []
        sub_question = current_plan.get("sub_question") or state.get("query", "")

        if current_plan.get("route_type") == "chat":
            current_plan["evidence_sufficient"] = True
            current_plan["evidence_grade"] = {
                "coverage_score": 1.0,
                "answerability_score": 1.0,
                "recommended_action": "generate",
                "missing_aspects": [],
                "reason": "chat_route",
            }
            updated_plans.append(current_plan)
            evidence_grades.append({"sub_question": sub_question, **current_plan["evidence_grade"]})
            continue

        if not selected_evidence:
            current_plan["evidence_sufficient"] = False
            current_plan["need_web_search"] = web_enabled and not used_web_search
            current_plan["need_retrieval"] = (
                state.get("kb_id") is not None
                and not current_plan["need_web_search"]
                and reflection_count < max_reflections
            )
            current_plan["evidence_grade"] = {
                "coverage_score": 0.0,
                "answerability_score": 0.0,
                "recommended_action": "web_search" if current_plan["need_web_search"] else "reflect",
                "missing_aspects": ["未检索到有效证据"],
                "reason": "empty_evidence",
            }
            uncovered_questions.append(sub_question)
            updated_plans.append(current_plan)
            evidence_grades.append({"sub_question": sub_question, **current_plan["evidence_grade"]})
            continue

        evidence_digest = "\n\n".join([
            f"{item['evidence_id']} | title={item.get('title', '')} | page={item.get('page_no')}\n"
            f"score={item.get('score', 0):.3f}\n"
            f"snippet={(item.get('support_snippet') or item.get('snippet', '')[:300])}"
            for item in selected_evidence[:6]
        ])

        system_prompt = """你是 RAG 子问题证据评审器。你需要判断当前证据是否足以回答这个子问题。

输出 JSON：
{
  "coverage_score": 0-1,
  "answerability_score": 0-1,
  "recommended_action": "generate|reflect|web_search",
  "missing_aspects": ["仍缺失的信息点"],
  "reason": "简短理由"
}

要求：
1. 只针对当前子问题评审，不要按整体问题判断
2. 只有 coverage_score 和 answerability_score 都足够时，才允许 generate
3. 如果知识库证据不足且允许联网，优先 web_search
"""

        user_prompt = f"""子问题：{sub_question}

证据摘要：
{evidence_digest}

请输出评审结果。"""

        try:
            result = _safe_load_json((await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])).content)
        except Exception:
            logger.exception("LLM failed in judge_evidence: sub_question=%s evidence_count=%s", sub_question, len(selected_evidence))
            result = {}

        if result:
            coverage_score = float(result.get("coverage_score", 0.0))
            answerability_score = float(result.get("answerability_score", 0.0))
            recommended_action = result.get("recommended_action", "reflect")
            missing_aspects = result.get("missing_aspects", [])
            evidence_sufficient = coverage_score >= 0.72 and answerability_score >= 0.72
        else:
            avg_score = sum(item.get("score", 0.0) for item in selected_evidence) / len(selected_evidence)
            evidence_sufficient = len(selected_evidence) >= 2 and avg_score >= 0.18
            coverage_score = min(1.0, len(selected_evidence) / 3)
            answerability_score = min(1.0, avg_score)
            recommended_action = "generate" if evidence_sufficient else ("web_search" if web_enabled else "reflect")
            missing_aspects = []

        planned_hybrid_web = (
            current_plan.get("route_type") == "hybrid"
            and web_enabled
            and not used_web_search
            and "web_search" in (current_plan.get("planned_tools") or [])
        )

        current_plan["evidence_sufficient"] = evidence_sufficient
        current_plan["need_web_search"] = (
            planned_hybrid_web
            or (
                not evidence_sufficient
                and web_enabled
                and not used_web_search
                and recommended_action == "web_search"
            )
        )
        current_plan["need_retrieval"] = (
            not evidence_sufficient
            and state.get("kb_id") is not None
            and not current_plan["need_web_search"]
            and reflection_count < max_reflections
        )
        current_plan["evidence_grade"] = {
            "coverage_score": round(coverage_score, 3),
            "answerability_score": round(answerability_score, 3),
            "recommended_action": recommended_action,
            "missing_aspects": missing_aspects,
            "reason": result.get("reason", "") if result else "fallback_heuristic",
        }

        if not evidence_sufficient:
            uncovered_questions.append(sub_question)

        updated_plans.append(current_plan)
        evidence_grades.append({"sub_question": sub_question, **current_plan["evidence_grade"]})

    all_covered = bool(updated_plans) and not uncovered_questions
    need_web_search = any(plan.get("need_web_search") for plan in updated_plans)
    need_reflection = (not all_covered) and not need_web_search and reflection_count < max_reflections

    return {
        "sub_query_plans": updated_plans,
        "evidence_sufficient": all_covered,
        "need_reflection": need_reflection,
        "need_web_search": need_web_search,
        "evidence_grade": {
            "per_sub_question": evidence_grades,
            "uncovered_questions": uncovered_questions,
            "recommended_action": "generate" if all_covered else ("web_search" if need_web_search else "reflect"),
        },
    }
