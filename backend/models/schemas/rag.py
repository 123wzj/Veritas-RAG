# -*- coding: utf-8 -*-
"""
RAG 查询与响应数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


class RAGQueryRequest(BaseModel):
    """RAG 查询请求模型"""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    kb_id: Optional[int] = None
    web_enabled: bool = False
    stream: bool = True
    top_k: Optional[int] = None


class RAGQueryResponse(BaseModel):
    """RAG 查询响应模型（非流式）"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    latency_ms: int
    session_id: Optional[str] = None


# ========== 流式输出事件模型 ==========

class StreamEvent(BaseModel):
    """流式事件基类"""
    event: str
    request_id: str
    session_id: Optional[str] = None
    timestamp: int


class RunStartedEvent(StreamEvent):
    """查询开始事件"""
    event: Literal["run.started"] = "run.started"
    data: Dict[str, Any]


class MemoryLoadedEvent(StreamEvent):
    """记忆加载完成事件"""
    event: Literal["memory.loaded"] = "memory.loaded"
    data: Dict[str, Any]


class QueryRewrittenEvent(StreamEvent):
    """问题改写完成事件"""
    event: Literal["query.rewritten"] = "query.rewritten"
    data: Dict[str, Any]


class QueryDecomposedEvent(StreamEvent):
    """问题拆解完成事件"""
    event: Literal["query.decomposed"] = "query.decomposed"
    data: Dict[str, Any]


class RetrievalStartedEvent(StreamEvent):
    """检索开始事件"""
    event: Literal["retrieval.started"] = "retrieval.started"


class RetrievalCompletedEvent(StreamEvent):
    """检索完成事件"""
    event: Literal["retrieval.completed"] = "retrieval.completed"
    data: Dict[str, Any]


class RerankCompletedEvent(StreamEvent):
    """重排完成事件"""
    event: Literal["rerank.completed"] = "rerank.completed"
    data: Dict[str, Any]


class ReflectionStartedEvent(StreamEvent):
    """反思开始事件"""
    event: Literal["reflection.started"] = "reflection.started"


class ReflectionCompletedEvent(StreamEvent):
    """反思完成事件"""
    event: Literal["reflection.completed"] = "reflection.completed"
    data: Dict[str, Any]


class WebSearchStartedEvent(StreamEvent):
    """联网搜索开始事件"""
    event: Literal["websearch.started"] = "websearch.started"


class WebSearchCompletedEvent(StreamEvent):
    """联网搜索完成事件"""
    event: Literal["websearch.completed"] = "websearch.completed"
    data: Dict[str, Any]


class AnswerDeltaEvent(StreamEvent):
    """答案增量事件"""
    event: Literal["answer.delta"] = "answer.delta"
    data: Dict[str, str]


class CitationDeltaEvent(StreamEvent):
    """引用增量事件"""
    event: Literal["citation.delta"] = "citation.delta"
    data: Dict[str, Any]


class AnswerCompletedEvent(StreamEvent):
    """答案完成事件"""
    event: Literal["answer.completed"] = "answer.completed"
    data: Dict[str, Any]


class MemoryUpdatedEvent(StreamEvent):
    """记忆更新事件"""
    event: Literal["memory.updated"] = "memory.updated"
    data: Dict[str, Any]


class RunFailedEvent(StreamEvent):
    """查询失败事件"""
    event: Literal["run.failed"] = "run.failed"
    data: Dict[str, Any]


# ========== LangGraph State 模型 ==========

class RAGGraphState(BaseModel):
    """RAG 状态机状态模型"""
    request_id: str
    session_id: Optional[str] = None
    user_id: int
    kb_id: Optional[int] = None
    query: str
    query_rewritten: Optional[str] = None
    query_intent: Optional[str] = None
    sub_questions: List[str] = []
    retrieval_queries: List[str] = []
    sub_query_plans: List[Dict[str, Any]] = []
    sub_query_answers: List[Dict[str, Any]] = []
    route_type: Optional[str] = None
    route_reason: Optional[str] = None
    planned_tools: List[str] = []
    web_enabled: bool = False
    top_k: int = 6
    retrieved_docs: List[Dict[str, Any]] = []
    reranked_docs: List[Dict[str, Any]] = []
    selected_evidence: List[Dict[str, Any]] = []
    evidence_grade: Optional[Dict[str, Any]] = None
    reasoning_trace_summary: Optional[str] = None
    reflection_notes: Optional[str] = None
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = []
    confidence: float = 0.0
    used_web_search: bool = False
    memory_context: Optional[Dict[str, Any]] = None
    session_summary: Optional[str] = None
    prompt_context: Optional[str] = None
    events: List[Dict[str, Any]] = []
    error: Optional[str] = None
    step_count: int = 0
    reflection_count: int = 0
    max_reflections: int = 0
    max_steps: int = 0
    need_retrieval: bool = False
    need_reflection: bool = False
    need_web_search: bool = False
    evidence_sufficient: bool = False
