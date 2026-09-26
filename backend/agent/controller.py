"""LLM controller that emits auditable decisions without exposing chain of thought."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.context.builder import react_context_builder
from backend.agent.schemas import AgentDecision, RuntimeBudget, ToolCallRequest
from backend.agent.tools.registry import ToolRegistry, tool_registry
from backend.graph.llm_factory import get_llm


CONTROL_TOOLS = [
    {
        "name": "submit_final_answer",
        "description": "Submit the final user-facing answer after enough evidence is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason_summary": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "cited_evidence_ids", "reason_summary", "confidence"],
        },
    },
    {
        "name": "submit_refusal",
        "description": "Refuse when policy, authorization, or evidence limitations prevent an answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reason_summary": {"type": "string"},
            },
            "required": ["answer", "reason_summary"],
        },
    },
    {
        "name": "request_clarification",
        "description": "Ask one concise clarification when the user intent is materially ambiguous.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reason_summary": {"type": "string"},
                "unresolved_slots": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "reason_summary", "unresolved_slots"],
        },
    },
]


class AgentController:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or tool_registry

    def _tool_definitions(self, available_tools: List[str]) -> List[Dict[str, Any]]:
        result = []
        for spec in self.registry.specs(available_tools):
            result.append({
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            })
        return [*result, *CONTROL_TOOLS]

    @staticmethod
    def _runtime_policy(state: Dict[str, Any], budget: RuntimeBudget) -> Dict[str, Any]:
        return {
            "runtime_mode": state.get("runtime_mode"),
            "iteration": int(state.get("iteration") or 0),
            "max_iterations": budget.max_iterations,
            "remaining_tool_calls": max(
                0, budget.max_tool_calls - len(state.get("tool_calls") or [])
            ),
            "web_enabled": bool(state.get("web_enabled")),
            "kb_selected": state.get("kb_id") is not None,
            "decision_feedback": state.get("decision_feedback") or {},
            "rules": [
                "Use only registered tools.",
                "Never put user/session/kb/run identifiers in tool arguments.",
                "Do not repeat an action fingerprint that produced no new evidence.",
                "Cite only Evidence Ledger E# identifiers.",
                "Use a control tool to finish; never return an unstructured final answer.",
            ],
        }

    async def decide(self, state: Dict[str, Any]) -> tuple[AgentDecision, Dict[str, Any]]:
        budget = RuntimeBudget.model_validate(state.get("budgets") or {})
        available_tools = list(state.get("available_tools") or [])
        definitions = self._tool_definitions(available_tools)
        context = react_context_builder.build(
            current_query=str(state.get("current_query") or ""),
            budget=budget,
            runtime_policy=self._runtime_policy(state, budget),
            working_memory=state.get("working_memory") or {},
            memory_context=state.get("memory_context") or {},
            evidence_ledger=state.get("evidence_ledger") or {},
            observations=state.get("observations") or [],
            tool_specs=definitions,
        )
        llm = get_llm("flash").bind_tools(
            definitions,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        response = await llm.ainvoke([
            SystemMessage(content=context["system"]),
            HumanMessage(content=context["text"]),
        ])
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            raise ValueError("Controller must return a registered action or control tool call")

        first_name = str(tool_calls[0].get("name") or "")
        if first_name == "submit_final_answer":
            args = dict(tool_calls[0].get("args") or {})
            decision = AgentDecision(
                type="final_answer",
                answer=str(args.get("answer") or ""),
                cited_evidence_ids=[str(item) for item in args.get("cited_evidence_ids") or []],
                reason_summary=str(args.get("reason_summary") or "")[:500],
                confidence=float(args.get("confidence") or 0.0),
            )
        elif first_name == "submit_refusal":
            args = dict(tool_calls[0].get("args") or {})
            decision = AgentDecision(
                type="refusal",
                answer=str(args.get("answer") or "证据不足，暂时无法可靠回答。"),
                reason_summary=str(args.get("reason_summary") or "")[:500],
            )
        elif first_name == "request_clarification":
            args = dict(tool_calls[0].get("args") or {})
            decision = AgentDecision(
                type="clarification",
                answer=str(args.get("answer") or "请补充问题所需的关键信息。"),
                reason_summary=str(args.get("reason_summary") or "")[:500],
                unresolved_slots=[str(item) for item in args.get("unresolved_slots") or []],
            )
        else:
            calls: List[ToolCallRequest] = []
            for raw in tool_calls:
                name = str(raw.get("name") or "")
                if name not in available_tools:
                    continue
                args = dict(raw.get("args") or {})
                calls.append(ToolCallRequest(
                    tool_call_id=str(raw.get("id") or uuid.uuid4()),
                    tool_name=name,
                    arguments=args,
                    purpose=str(args.pop("purpose", ""))[:500],
                    target_slots=[str(item) for item in args.get("target_slots") or []],
                ))
            decision = AgentDecision(
                type="tool_calls",
                tool_calls=calls,
                reason_summary="调用只读工具补充可验证证据。",
            )
        usage = getattr(response, "usage_metadata", None) or {}
        return decision, {
            "context": context,
            "model_usage": usage,
            "raw_action_count": len(tool_calls),
        }


agent_controller = AgentController()
