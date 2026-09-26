import pytest
from pydantic import ValidationError

from backend.agent.schemas import AgentDecision, ToolCallRequest
from backend.agent.state import create_agent_state


def test_agent_decision_requires_calls_for_tool_action():
    with pytest.raises(ValidationError):
        AgentDecision(type="tool_calls")


def test_agent_state_keeps_server_identity_and_separate_run_id():
    state = create_agent_state(
        query="hello",
        user_id=7,
        request_id="request-1",
        run_id="request-1:shadow",
        session_id="session-7",
        kb_id=9,
        web_enabled=False,
        top_k=6,
        runtime_mode="react_shadow",
    )
    assert state["user_id"] == 7
    assert state["session_id"] == "session-7"
    assert state["run_id"] == "request-1:shadow"


def test_tool_call_contract_is_typed():
    call = ToolCallRequest(
        tool_call_id="call-1",
        tool_name="knowledge_search",
        arguments={"query": "memory"},
        target_slots=["slot-1"],
    )
    assert call.target_slots == ["slot-1"]
