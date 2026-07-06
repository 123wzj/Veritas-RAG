# -*- coding: utf-8 -*-
"""
Agentic RAG 状态定义。

LangGraph 会把 RAG 流程中的中间结果都放在这个 State 里：
用户问题、路由规划、检索结果、重排结果、证据、反思、最终回答等节点之间都通过它传递。
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from backend.core.config import settings


class RAGState(TypedDict):
    """
    Agentic RAG 的全局状态对象。

    每个字段都代表链路中的一个阶段性信息，节点只读写自己关心的字段。
    """

    # ========== 基础信息 ==========
    # 单次 RAG 请求的唯一 ID，便于日志追踪、前端事件关联和排查问题。
    request_id: str
    # 会话 ID，用于把本次问答写入同一个聊天历史；没有时后端会自动创建。
    session_id: Optional[str]
    # 当前用户 ID，用于权限校验、知识库隔离、会话记忆读取。
    user_id: int
    # 当前选择的知识库 ID；为空时通常走闲聊、联网或普通模型回答。
    kb_id: Optional[int]

    # ========== 查询理解与规划 ==========
    # 用户原始问题，所有后续改写、检索和生成都围绕它展开。
    query: str
    # 大模型改写后的检索友好问题，用于降低口语化、省略表达对召回的影响。
    query_rewritten: Optional[str]
    # 用户意图分类结果，例如知识库问答、联网查询、闲聊、总结等。
    query_intent: Optional[str]
    # 查询拆解后的子问题列表，用于复杂问题的多路召回。
    sub_questions: List[str]
    # 实际送入检索器的查询列表，通常包含原问题、改写问题、子问题。
    retrieval_queries: List[str]
    # 子问题级执行计划。每个子问题独立保存路由、检索、证据和答案。
    sub_query_plans: List[Dict[str, Any]]
    # 子问题级生成结果，用于最终聚合答案。
    sub_query_answers: List[Dict[str, Any]]
    # 路由类型，决定下一步走知识库检索、联网搜索、混合检索还是直接回答。
    route_type: Optional[str]
    # 路由原因，用于调试和前端展示“为什么走这个路径”。
    route_reason: Optional[str]
    # 计划使用的工具列表，例如 knowledge_base、web_search 等。
    planned_tools: List[str]
    # 用户是否允许联网增强；即使路由想联网，也需要这个开关允许。
    web_enabled: bool
    # 期望最终保留的证据数量上限，会影响召回、精排和 prompt 组装规模。
    top_k: int

    # ========== 检索与证据 ==========
    # 初召回候选文档/子块，来自 Dense、Sparse、RRF 融合后的结果。
    retrieved_docs: List[Dict[str, Any]]
    # Reranker 精排后的候选结果，排序质量通常高于 retrieved_docs。
    reranked_docs: List[Dict[str, Any]]
    # 最终打包给生成模型的证据，通常包含父块上下文、子块命中片段和引用元数据。
    selected_evidence: List[Dict[str, Any]]
    # 证据质量评估结果，包含覆盖度、可回答性、缺失点和推荐动作。
    evidence_grade: Optional[Dict[str, Any]]

    # ========== 推理、反思与生成 ==========
    # 对内部推理链路的摘要，只保留可展示的简要过程，不保存长篇私有推理。
    reasoning_trace_summary: Optional[str]
    # 反思节点产生的修正建议，例如需要补充检索、证据不足、答案不够 grounded。
    reflection_notes: Optional[str]
    # 草稿答案，预留字段；可用于先生成草稿再校验或重写。
    draft_answer: Optional[str]
    # 最终返回给用户的答案正文。
    final_answer: Optional[str]
    # 答案校验结果，通常包含 grounded/useful/reason/confidence_adjustment 等信息。
    verification: Optional[Dict[str, Any]]

    # ========== 输出结果 ==========
    # 引用来源列表，用于前端展示 E1/E2、文档标题、页码、片段、得分等。
    citations: List[Dict[str, Any]]
    # 答案置信度，综合证据质量、引用覆盖、校验结果得到。
    confidence: float
    # 本次回答是否实际使用了联网搜索结果。
    used_web_search: bool

    # ========== 记忆与上下文 ==========
    # 用户画像、长期记忆、工作记忆等上下文信息，由 memory 节点读取。
    memory_context: Optional[Dict[str, Any]]
    # 当前会话摘要，用于多轮对话时压缩历史上下文。
    session_summary: Optional[Dict[str, Any]]
    # LLM 生成的记忆更新计划，由 API 保存回答时在同一事务内应用。
    memory_update_plan: Optional[Dict[str, Any]]
    # 组装 prompt 时使用的额外上下文，通常由证据打包节点生成。
    prompt_context: Optional[str]
    # Context Assembler 的总体和分区 token 估算。
    context_token_usage: Optional[Dict[str, Any]]

    # ========== 流程控制与事件 ==========
    # 对外发送的事件流，前端用它展示“思考过程”和每个节点进展。
    events: List[Dict[str, Any]]
    # Per-node wall-clock timings collected by production graph nodes.
    latency_breakdown_ms: Dict[str, float]
    # 错误信息；某个节点失败时写入，后续由接口层返回给前端。
    error: Optional[str]
    # 已执行步骤数。Annotated + operator.add 表示 LangGraph 合并状态时做累加。
    step_count: Annotated[int, operator.add]
    # 已执行反思次数，用于控制反思循环，防止无限重试。
    reflection_count: int
    # 最大反思次数，默认最多 3 次，防止链路过慢或循环。
    max_reflections: int
    # 最大工具/节点执行步数，用于保护 agent 流程。
    max_steps: int

    # ========== 内部路由布尔开关 ==========
    # 是否需要知识库检索；一般有 kb_id 且路由选择知识库时为 True。
    need_retrieval: bool
    # 是否需要进入反思节点；证据不足或答案校验失败时为 True。
    need_reflection: bool
    # 是否需要联网搜索；知识库不足且用户允许联网时可能为 True。
    need_web_search: bool
    # 当前证据是否足够支撑回答；生成前和反思循环都会参考它。
    evidence_sufficient: bool


def create_initial_state(
    query: str,
    user_id: int,
    request_id: Optional[str] = None,
    kb_id: Optional[int] = None,
    session_id: Optional[str] = None,
    web_enabled: bool = False,
    top_k: Optional[int] = None,
    max_reflections: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> RAGState:
    """
    创建一次 RAG 请求的初始状态。

    Args:
        query: 用户原始问题。
        user_id: 当前用户 ID。
        kb_id: 当前知识库 ID；为空时默认不做知识库检索。
        session_id: 当前会话 ID；为空时后端会创建新会话。
        web_enabled: 是否允许联网增强。
        top_k: 最终期望保留的证据数量。

    Returns:
        可直接传入 LangGraph 的初始状态字典。
    """
    import uuid

    return {
        "request_id": request_id or str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "kb_id": kb_id,
        "query": query,
        "query_rewritten": None,
        "query_intent": None,
        "sub_questions": [],
        "retrieval_queries": [],
        "sub_query_plans": [],
        "sub_query_answers": [],
        "route_type": None,
        "route_reason": None,
        "planned_tools": [],
        "web_enabled": bool(web_enabled and settings.WEB_SEARCH_ENABLED),
        "top_k": max(3, min(top_k or 6, 12)),
        "retrieved_docs": [],
        "reranked_docs": [],
        "selected_evidence": [],
        "evidence_grade": None,
        "reasoning_trace_summary": None,
        "reflection_notes": None,
        "draft_answer": None,
        "final_answer": None,
        "verification": None,
        "citations": [],
        "confidence": 0.0,
        "used_web_search": False,
        "memory_context": None,
        "session_summary": None,
        "memory_update_plan": None,
        "prompt_context": None,
        "context_token_usage": None,
        "events": [],
        "latency_breakdown_ms": {},
        "error": None,
        "step_count": 0,
        "reflection_count": 0,
        "max_reflections": max(
            0,
            min(
                settings.MAX_REFLECTION_ROUNDS if max_reflections is None else max_reflections,
                3,
            ),
        ),
        "max_steps": max(1, min(settings.MAX_TOOL_STEPS if max_steps is None else max_steps, 20)),
        # 默认只有指定知识库时才需要检索；后续路由节点可以继续覆盖。
        "need_retrieval": kb_id is not None,
        "need_reflection": False,
        "need_web_search": False,
        # 没有知识库时通常走闲聊/直接回答，因此初始认为不缺知识库证据。
        "evidence_sufficient": kb_id is None,
    }
