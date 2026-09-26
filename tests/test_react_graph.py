import asyncio
from unittest.mock import AsyncMock

import backend.agent.graph as graph_module
import backend.agent.verification as verification_module
from backend.agent.graph import react_graph, route_after_decide, route_after_observe, verify_answer
from backend.agent.schemas import (
    AgentDecision,
    AnswerClaim,
    EvidenceItem,
    ToolCallRequest,
    ToolObservation,
)
from backend.agent.state import create_agent_state


def test_graph_routes_tool_decision_to_act():
    assert route_after_decide({"decision": {"type": "tool_calls"}}) == "act"
    assert route_after_decide({"decision": {"type": "final_answer"}}) == "verify"


def test_graph_verification_publishes_only_known_evidence_ids():
    state = {
        "decision": {
            "type": "final_answer",
            "tool_calls": [],
            "answer": "answer [E1]",
            "cited_evidence_ids": ["E1"],
            "reason_summary": "supported",
            "unresolved_slots": [],
            "confidence": 0.9,
        },
        "evidence_ledger": {
            "entries": {"E1": EvidenceItem(
                evidence_id="E1", source_type="web", title="source",
                url="https://example.com", support_snippet="proof",
            ).model_dump(mode="json")},
            "slot_coverage": {}, "conflicts": [], "next_index": 2,
        },
        "tool_calls": [{"tool_name": "web_search"}],
        "events": [],
        "budgets": {"max_iterations": 2},
        "iteration": 1,
    }
    result = asyncio.run(verify_answer(state))
    assert result["final_answer"] == "answer [E1]"
    assert result["citations"][0]["evidence_id"] == "E1"


def test_observe_route_uses_explicit_forced_decision():
    assert route_after_observe({"decision": {"type": "refusal"}}) == "verify"
    assert route_after_observe({"decision": {"type": "tool_calls"}}) == "decide"


def test_compiled_graph_runs_decide_act_observe_decide_cycle(monkeypatch):
    monkeypatch.setattr(
        verification_module.settings,
        "AGENT_SEMANTIC_VERIFICATION_ENABLED",
        False,
    )
    class FakeDb:
        def close(self):
            return None

    decisions = iter([
        AgentDecision(
            type="tool_calls",
            tool_calls=[ToolCallRequest(
                tool_call_id="call-1",
                tool_name="knowledge_search",
                arguments={"query": "answer", "target_slots": ["slot-1"]},
                target_slots=["slot-1"],
            )],
            reason_summary="need evidence",
        ),
            AgentDecision(
                type="final_answer",
                answer="grounded answer [E1]",
                cited_evidence_ids=["E1"],
                claims=[AnswerClaim(
                    claim_id="claim-1",
                    text="grounded answer",
                    slot_id="slot-1",
                    evidence_ids=["E1"],
                )],
                reason_summary="evidence is sufficient",
            confidence=0.9,
        ),
    ])

    async def fake_decide(_state):
        return next(decisions), {
            "context": {"token_usage": {"used": 10, "budget": 100}},
            "model_usage": {"output_tokens": 5},
        }

    async def fake_execute(call, **_kwargs):
        return ToolObservation(
            observation_id="obs-1",
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            status="success",
            summary="found",
            evidence=[EvidenceItem(
                source_type="knowledge_base",
                doc_id="d1",
                chunk_id="c1",
                support_snippet="proof",
                target_slots=["slot-1"],
            )],
            supported_slots=["slot-1"],
            next_hint="answer",
        ), "fingerprint-1"

    monkeypatch.setattr(graph_module, "get_db", lambda: iter([FakeDb()]))
    monkeypatch.setattr(
        graph_module.react_memory_service,
        "load",
        AsyncMock(return_value={"selected_memory_ids": [], "recent_messages": []}),
    )
    monkeypatch.setattr(graph_module.agent_controller, "decide", fake_decide)
    monkeypatch.setattr(graph_module.tool_gateway, "execute", fake_execute)
    monkeypatch.setattr(
        graph_module.memory_service,
        "build_memory_update_plan",
        AsyncMock(return_value={"long_term_actions": []}),
    )

    state = create_agent_state(
        query="question",
        user_id=1,
        request_id="req-1",
        session_id="s1",
        kb_id=7,
        web_enabled=False,
        top_k=6,
        runtime_mode="react",
        budgets={"max_iterations": 3, "max_tool_calls": 3, "max_kb_calls": 2},
    )
    final = asyncio.run(react_graph.ainvoke(state))
    assert final["final_answer"] == "grounded answer [E1]"
    assert final["citations"][0]["evidence_id"] == "E1"
    assert final["iteration"] == 1
    assert len(final["tool_calls"]) == 1
