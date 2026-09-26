import asyncio

import backend.agent.runtime as runtime_module
from backend.agent.runtime import build_runtime_budget, get_runtime_mode, run_rag_runtime
from backend.core.config import settings


def test_runtime_mode_defaults_to_react(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_RUNTIME_MODE", "react")
    assert get_runtime_mode() == "react"
    assert get_runtime_mode("not-a-mode") == "react"
    assert get_runtime_mode("react") == "react"
    assert get_runtime_mode("react_shadow") == "react"
    assert get_runtime_mode("legacy") == "react"


def test_runtime_budget_is_bounded_and_explicit():
    budget = build_runtime_budget()
    assert budget.max_iterations >= 1
    assert budget.max_tool_calls >= budget.max_kb_calls
    assert budget.input_token_limit >= 2048


def test_runtime_dispatcher_uses_react_even_for_legacy_override(monkeypatch):
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {"final_answer": "ok", "runtime_mode": state["runtime_mode"]}

    monkeypatch.setattr(runtime_module, "react_graph", FakeGraph())

    async def collect():
        return [
            item
            async for item in run_rag_runtime(
                query="question",
                user_id=1,
                request_id="req-react-default",
                session_id="session-1",
                runtime_mode="legacy",
                stream_events=False,
            )
        ]

    result = asyncio.run(collect())
    assert result[0]["runtime_mode"] == "react"
    assert captured["state"]["runtime_mode"] == "react"
