# -*- coding: utf-8 -*-
"""
Agentic RAG 状态图定义。
"""

from langgraph.graph import StateGraph, END

from backend.graph.state.state import RAGState, create_initial_state
from backend.graph.nodes.query_nodes import (
    rewrite_query,
    decompose_query,
    plan_query_route,
    route_planned_query,
)
from backend.graph.nodes.retrieval_nodes import (
    retrieve_hybrid,
    rerank_candidates,
    pack_evidence,
    judge_evidence_slots as judge_evidence,
)
from backend.graph.nodes.reflection_nodes import reflection, route_reflection, route_evidence
from backend.graph.nodes.generation_nodes import generate_answer, verify_answer, route_verification
from backend.graph.nodes.memory_nodes import (
    load_generation_memory,
    load_user_memory,
    write_memory,
)
from backend.graph.nodes.web_nodes import web_search
from backend.core.config import settings


GRAPH_RUN_CONFIG = {
    "recursion_limit": max(25, settings.GRAPH_RECURSION_LIMIT),
}


def create_agentic_rag_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    workflow.add_node("load_memory", load_user_memory)
    workflow.add_node("load_generation_memory", load_generation_memory)
    workflow.add_node("write_memory", write_memory)

    # 查询重写
    workflow.add_node("rewrite_query", rewrite_query)
    # 将复杂问题拆成多个可检索子问题和检索表达
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("plan_query_route", plan_query_route)

    workflow.add_node("retrieve", retrieve_hybrid)
    workflow.add_node("rerank", rerank_candidates)
    workflow.add_node("pack_evidence", pack_evidence)
    workflow.add_node("judge_evidence", judge_evidence)

    workflow.add_node("reflect", reflection)
    workflow.add_node("web_search", web_search)

    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_answer", verify_answer)

    workflow.set_entry_point("load_memory")

    workflow.add_edge("load_memory", "rewrite_query")
    workflow.add_edge("rewrite_query", "decompose_query")
    workflow.add_edge("decompose_query", "plan_query_route")

    workflow.add_conditional_edges(
        "plan_query_route",
        route_planned_query,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate": "load_generation_memory",
        },
    )

    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "pack_evidence")
    workflow.add_edge("pack_evidence", "judge_evidence")

    workflow.add_conditional_edges(
        "judge_evidence",
        route_evidence,
        {
            "generate": "load_generation_memory",
            "reflect": "reflect",
            "web_search": "web_search",
        },
    )

    workflow.add_conditional_edges(
        "reflect",
        route_reflection,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "proceed": "load_generation_memory",
        },
    )

    workflow.add_edge("web_search", "judge_evidence")
    workflow.add_edge("load_generation_memory", "generate_answer")
    workflow.add_edge("generate_answer", "verify_answer")

    workflow.add_conditional_edges(
        "verify_answer",
        route_verification,
        {
            "reflect": "reflect",
            "write_memory": "write_memory",
        },
    )

    workflow.add_edge("write_memory", END)
    return workflow

# 编译工作流
agentic_rag_graph = create_agentic_rag_graph().compile()

# 初始化工作流
async def run_agentic_rag(
    query: str,
    user_id: int,
    request_id: str | None = None,
    kb_id: int = None,
    session_id: str = None,
    web_enabled: bool = False,
    stream_events: bool = True,
    top_k: int | None = None,
    max_reflections: int | None = None,
    max_steps: int | None = None,
) -> RAGState:
    initial_state = create_initial_state(
        query=query,
        user_id=user_id,
        request_id=request_id,
        kb_id=kb_id,
        session_id=session_id,
        web_enabled=web_enabled,
        top_k=top_k,
        max_reflections=max_reflections,
        max_steps=max_steps,
    )

    if stream_events:
        async for event in agentic_rag_graph.astream(initial_state, config=GRAPH_RUN_CONFIG):
            yield event
    else:
        final_state = await agentic_rag_graph.ainvoke(initial_state, config=GRAPH_RUN_CONFIG)
        yield final_state
