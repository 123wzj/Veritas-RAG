"""Controlled decide -> act -> observe -> decide LangGraph runtime."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from backend.agent.controller import agent_controller
from backend.agent.evidence.ledger import evidence_ledger_service
from backend.agent.memory.service import react_memory_service
from backend.agent.schemas import RuntimeBudget, ToolCallRequest, ToolExecutionContext
from backend.agent.state import AgentState
from backend.agent.tools.bootstrap import register_builtin_tools
from backend.agent.tools.gateway import ToolGateway
from backend.agent.tools.registry import tool_registry
from backend.agent.verification import verify_decision
from backend.db.mysql.connection import get_db
from backend.services.context.context_assembler import estimate_tokens
from backend.services.memory.memory_service import memory_service


register_builtin_tools()
tool_gateway = ToolGateway()


def _event(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": name,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def hydrate_context(state: AgentState) -> Dict[str, Any]:
    started = time.perf_counter()
    db = next(get_db())
    try:
        working = react_memory_service.initial_working_memory(
            run_id=state["run_id"], query=state["current_query"]
        )
        memory_context = await react_memory_service.load(
            run_id=state["run_id"],
            user_id=state["user_id"],
            session_id=state["session_id"],
            kb_id=state.get("kb_id"),
            query=state["current_query"],
            request_id=state["request_id"],
            db=db,
            branch_id=state.get("branch_id"),
            read_only=True,
        )
    finally:
        db.close()
    tools = []
    if state.get("kb_id") is not None:
        tools.append("knowledge_search")
    if state.get("web_enabled"):
        tools.append("web_search")
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "working_memory": working,
        "memory_context": memory_context,
        "available_tools": tools,
        "events": [*state.get("events", []), _event("memory.loaded", {
            "selected_memory_ids": memory_context.get("selected_memory_ids") or [],
            "available_tools": tools,
        })],
        "latency_breakdown_ms": {**state.get("latency_breakdown_ms", {}), "react.hydrate_context": elapsed},
    }


async def decide(state: AgentState) -> Dict[str, Any]:
    budget = RuntimeBudget.model_validate(state.get("budgets") or {})
    if int(state.get("iteration") or 0) >= budget.max_iterations:
        decision = {
            "type": "refusal",
            "tool_calls": [],
            "answer": "在当前检索预算内仍缺少足够证据，暂时无法可靠回答。",
            "cited_evidence_ids": [],
            "reason_summary": "迭代预算已用尽。",
            "unresolved_slots": (state.get("working_memory") or {}).get("unresolved_slots") or [],
            "confidence": 0.0,
        }
        return {
            "decision": decision,
            "pending_tool_calls": [],
            "stop_reason": "max_iterations",
            "events": [*state.get("events", []), _event("agent.decision", {
                "type": "refusal", "reason_summary": decision["reason_summary"]
            })],
        }
    started = time.perf_counter()
    try:
        decision, metadata = await agent_controller.decide(state)
        value = decision.model_dump(mode="json")
        pending = [item.model_dump(mode="json") for item in decision.tool_calls]
        event_data = {
            "type": decision.type,
            "reason_summary": decision.reason_summary,
            "tool_names": [item.tool_name for item in decision.tool_calls],
        }
        usage = metadata.get("model_usage") or {}
        context_usage = metadata["context"]["token_usage"]
        output_tokens = int(usage.get("output_tokens") or 0)
        latency = (time.perf_counter() - started) * 1000
        return {
            "decision": value,
            "pending_tool_calls": pending,
            "decision_feedback": None,
            "context_token_usage": context_usage,
            "output_tokens": int(state.get("output_tokens") or 0) + output_tokens,
            "events": [*state.get("events", []), _event("agent.decision", event_data)],
            "latency_breakdown_ms": {**state.get("latency_breakdown_ms", {}), f"react.decide.{state.get('iteration', 0)}": latency},
        }
    except Exception as exc:
        return {
            "error": f"agent_decision_failed: {exc}",
            "stop_reason": "controller_error",
            "events": [*state.get("events", []), _event("agent.failed", {"stage": "decide", "error": str(exc)})],
        }


async def execute_tools(state: AgentState) -> Dict[str, Any]:
    db = next(get_db())
    observations = list(state.get("observations") or [])
    tool_calls = list(state.get("tool_calls") or [])
    cache = dict(state.get("idempotency_cache") or {})
    action_fingerprints = list(state.get("action_fingerprints") or [])
    events = list(state.get("events") or [])
    budget = RuntimeBudget.model_validate(state.get("budgets") or {})
    context = ToolExecutionContext(
        run_id=state["run_id"],
        request_id=state["request_id"],
        user_id=state["user_id"],
        session_id=state["session_id"],
        branch_id=state.get("branch_id"),
        kb_id=state.get("kb_id"),
        web_enabled=bool(state.get("web_enabled")),
        budget=budget,
    )
    started = time.perf_counter()
    try:
        for raw in state.get("pending_tool_calls") or []:
            call = ToolCallRequest.model_validate(raw)
            events.append(_event("tool.requested", {
                "tool_call_id": call.tool_call_id, "tool_name": call.tool_name
            }))
            observation, fingerprint = await tool_gateway.execute(
                call,
                context=context,
                db=db,
                prior_calls=tool_calls,
                idempotency_cache=cache,
            )
            tool_calls.append({
                **call.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "status": observation.status,
                "error_code": observation.error_code,
            })
            observations.append(observation.model_dump(mode="json"))
            action_fingerprints.append(fingerprint)
            events.append(_event("tool.completed", {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "status": observation.status,
                "latency_ms": observation.metrics.get("latency_ms"),
            }))
    finally:
        db.close()
    latency = (time.perf_counter() - started) * 1000
    return {
        "tool_calls": tool_calls,
        "observations": observations,
        "idempotency_cache": cache,
        "action_fingerprints": action_fingerprints,
        "pending_tool_calls": [],
        "events": events,
        "latency_breakdown_ms": {**state.get("latency_breakdown_ms", {}), f"react.act.{state.get('iteration', 0)}": latency},
    }


async def observe(state: AgentState) -> Dict[str, Any]:
    ledger_value = state.get("evidence_ledger") or {}
    working = state.get("working_memory") or {}
    observations = list(state.get("observations") or [])
    recent_calls = len((state.get("decision") or {}).get("tool_calls") or [])
    latest = observations[-recent_calls:] if recent_calls else []
    added_total = 0
    for item in latest:
        from backend.agent.schemas import ToolObservation
        observation = ToolObservation.model_validate(item)
        ledger, normalized, added = evidence_ledger_service.add_observation(
            ledger_value, observation
        )
        ledger_value = ledger.model_dump(mode="json")
        added_total += added
        fingerprint = next(
            (
                call.get("fingerprint", "")
                for call in reversed(state.get("tool_calls") or [])
                if call.get("tool_call_id") == normalized.tool_call_id
            ),
            "",
        )
        working = react_memory_service.update_after_observation(
            working,
            observation=normalized.model_dump(mode="json"),
            fingerprint=fingerprint,
        )
    no_progress = int(state.get("no_progress_count") or 0) + (1 if added_total == 0 else -999)
    no_progress = max(0, no_progress)
    iteration = int(state.get("iteration") or 0) + 1
    budget = RuntimeBudget.model_validate(state.get("budgets") or {})
    forced_decision = None
    if (
        no_progress >= 2
        or len(state.get("tool_calls") or []) >= budget.max_tool_calls
        or iteration >= budget.max_iterations
    ):
        forced_decision = {
            "type": "refusal",
            "tool_calls": [],
            "answer": "在当前检索预算内没有获得足够的新证据，暂时无法可靠回答。",
            "cited_evidence_ids": [],
            "reason_summary": "工具预算或无进展阈值已达到。",
            "unresolved_slots": working.get("unresolved_slots") or [],
            "confidence": 0.0,
        }
    events = [*state.get("events", []), _event("observation.created", {
        "count": len(latest), "new_evidence_count": added_total, "iteration": iteration
    }), _event("evidence.updated", {
        "evidence_count": len((ledger_value.get("entries") or {})),
        "unresolved_slots": working.get("unresolved_slots") or [],
    })]
    return {
        "evidence_ledger": ledger_value,
        "working_memory": working,
        "iteration": iteration,
        "no_progress_count": no_progress,
        "last_ledger_size": len(ledger_value.get("entries") or {}),
        "events": events,
        **({"decision": forced_decision} if forced_decision else {}),
    }


async def verify_answer(state: AgentState) -> Dict[str, Any]:
    decision = state.get("decision") or {}
    verification = verify_decision(
        decision,
        state.get("evidence_ledger") or {},
        had_tool_calls=bool(state.get("tool_calls")),
    )
    budget = RuntimeBudget.model_validate(state.get("budgets") or {})
    if not verification.publishable and int(state.get("iteration") or 0) < budget.max_iterations:
        return {
            "verification": verification.model_dump(mode="json"),
            "decision_feedback": verification.model_dump(mode="json"),
            "decision": None,
            "events": [*state.get("events", []), _event("answer.verified", {
                "publishable": False, "reason": verification.reason
            })],
        }

    if not verification.publishable:
        answer = "现有证据不足以支撑可靠结论，我暂时不能给出确定答案。"
        cited_ids = []
        confidence = 0.0
        stop_reason = "verification_failed"
    else:
        answer = str(decision.get("answer") or "")
        cited_ids = list(decision.get("cited_evidence_ids") or [])
        confidence = float(decision.get("confidence") or 0.0)
        stop_reason = str(decision.get("type") or "completed")
    citations = evidence_ledger_service.citations(
        state.get("evidence_ledger") or {}, cited_ids
    )
    ledger_entries = (state.get("evidence_ledger") or {}).get("entries") or {}
    return {
        "final_answer": answer,
        "citations": citations,
        "confidence": confidence,
        "verification": verification.model_dump(mode="json"),
        "stop_reason": stop_reason,
        "selected_evidence": [ledger_entries[item] for item in cited_ids if item in ledger_entries],
        "events": [*state.get("events", []), _event("answer.verified", {
            "publishable": verification.publishable,
            "citation_count": len(citations),
        })],
    }


async def propose_memory_update(state: AgentState) -> Dict[str, Any]:
    if not state.get("final_answer"):
        return {}
    db = next(get_db())
    started = time.perf_counter()
    try:
        plan = await memory_service.build_memory_update_plan(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["current_query"],
            answer=state["final_answer"],
            request_id=state["request_id"],
            kb_id=state.get("kb_id"),
            db=db,
            verification=state.get("verification") or {},
            confidence=float(state.get("confidence") or 0.0),
            working_memory_draft=state.get("working_memory") or {},
            branch_id=state.get("branch_id"),
        )
    finally:
        db.close()
    latency = (time.perf_counter() - started) * 1000
    return {
        "memory_update_plan": plan,
        "events": [*state.get("events", []), _event("memory.update.planned", {
            "action_count": len(plan.get("long_term_actions") or [])
        })],
        "latency_breakdown_ms": {**state.get("latency_breakdown_ms", {}), "react.memory_plan": latency},
    }


def route_after_decide(state: AgentState) -> Literal["act", "verify", "end"]:
    if state.get("error"):
        return "end"
    decision = state.get("decision") or {}
    return "act" if decision.get("type") == "tool_calls" else "verify"


def route_after_observe(state: AgentState) -> Literal["decide", "verify"]:
    return "verify" if (state.get("decision") or {}).get("type") == "refusal" else "decide"


def route_after_verify(state: AgentState) -> Literal["decide", "memory", "end"]:
    if state.get("error"):
        return "end"
    if not state.get("final_answer"):
        return "decide"
    return "memory"


def create_react_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("hydrate_context", hydrate_context)
    workflow.add_node("decide", decide)
    workflow.add_node("act", execute_tools)
    workflow.add_node("observe", observe)
    workflow.add_node("verify", verify_answer)
    workflow.add_node("memory", propose_memory_update)
    workflow.set_entry_point("hydrate_context")
    workflow.add_edge("hydrate_context", "decide")
    workflow.add_conditional_edges("decide", route_after_decide, {
        "act": "act", "verify": "verify", "end": END,
    })
    workflow.add_edge("act", "observe")
    workflow.add_conditional_edges("observe", route_after_observe, {
        "decide": "decide", "verify": "verify",
    })
    workflow.add_conditional_edges("verify", route_after_verify, {
        "decide": "decide", "memory": "memory", "end": END,
    })
    workflow.add_edge("memory", END)
    return workflow.compile()


react_graph = create_react_graph()
