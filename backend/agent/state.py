"""LangGraph state for the controlled ReAct runtime."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    request_id: str
    user_id: int
    session_id: str
    branch_id: Optional[int]
    kb_id: Optional[int]
    web_enabled: bool
    current_query: str
    top_k: int
    runtime_mode: str

    memory_context: Dict[str, Any]
    working_memory: Dict[str, Any]
    available_tools: List[str]
    decision: Optional[Dict[str, Any]]
    pending_tool_calls: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    idempotency_cache: Dict[str, Dict[str, Any]]
    evidence_ledger: Dict[str, Any]
    selected_evidence: List[Dict[str, Any]]

    iteration: int
    budgets: Dict[str, Any]
    decision_feedback: Optional[Dict[str, Any]]
    stop_reason: Optional[str]
    no_progress_count: int
    last_ledger_size: int
    action_fingerprints: List[str]

    final_answer: Optional[str]
    citations: List[Dict[str, Any]]
    confidence: float
    verification: Optional[Dict[str, Any]]
    memory_update_plan: Optional[Dict[str, Any]]
    context_token_usage: Dict[str, Any]
    last_context_manifest: Dict[str, Any]
    input_tokens: int
    output_tokens: int

    trace_attempt_no: int
    trace_node_counts: Dict[str, int]
    trace_event_offset: int

    events: List[Dict[str, Any]]
    latency_breakdown_ms: Dict[str, float]
    error: Optional[str]


def create_agent_state(
    *,
    query: str,
    user_id: int,
    request_id: str,
    session_id: str,
    kb_id: Optional[int],
    web_enabled: bool,
    top_k: Optional[int],
    runtime_mode: str,
    run_id: Optional[str] = None,
    branch_id: Optional[int] = None,
    budgets: Optional[Dict[str, Any]] = None,
) -> AgentState:
    return {
        "run_id": run_id or request_id,
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "branch_id": branch_id,
        "kb_id": kb_id,
        "web_enabled": web_enabled,
        "current_query": query,
        "top_k": max(1, min(top_k or 6, 12)),
        "runtime_mode": runtime_mode,
        "memory_context": {},
        "working_memory": {},
        "available_tools": [],
        "decision": None,
        "pending_tool_calls": [],
        "tool_calls": [],
        "observations": [],
        "idempotency_cache": {},
        "evidence_ledger": {"entries": {}, "slot_coverage": {}, "conflicts": [], "next_index": 1, "next_conflict_index": 1},
        "selected_evidence": [],
        "iteration": 0,
        "budgets": budgets or {},
        "decision_feedback": None,
        "stop_reason": None,
        "no_progress_count": 0,
        "last_ledger_size": 0,
        "action_fingerprints": [],
        "final_answer": None,
        "citations": [],
        "confidence": 0.0,
        "verification": None,
        "memory_update_plan": None,
        "context_token_usage": {},
        "last_context_manifest": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "trace_attempt_no": 1,
        "trace_node_counts": {},
        "trace_event_offset": 0,
        "events": [],
        "latency_breakdown_ms": {},
        "error": None,
    }
