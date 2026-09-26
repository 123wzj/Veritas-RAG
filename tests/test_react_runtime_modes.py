import asyncio
from pathlib import Path

import backend.agent.runtime as runtime_module
from backend.agent.checkpoint import ReactCheckpointManager
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


def test_runtime_resumes_unfinished_checkpoint_with_same_request_id(monkeypatch):
    captured = {}

    class Snapshot:
        values = {"iteration": 2}
        next = ("decide",)

    class FakeGraph:
        async def aget_state(self, config):
            captured["state_config"] = config
            return Snapshot()

        async def ainvoke(self, state, config):
            captured["input"] = state
            captured["invoke_config"] = config
            return {"final_answer": "resumed"}

    monkeypatch.setattr(runtime_module, "react_graph", FakeGraph())

    async def collect():
        return [
            item
            async for item in run_rag_runtime(
                query="question",
                user_id=7,
                request_id="resume-me",
                session_id="session-1",
                stream_events=False,
            )
        ]

    result = asyncio.run(collect())
    assert result[0]["final_answer"] == "resumed"
    assert captured["input"] is None
    assert captured["invoke_config"]["configurable"]["thread_id"] == "user:7:run:resume-me"


def test_runtime_returns_completed_checkpoint_without_reexecution(monkeypatch):
    completed = {"final_answer": "already complete", "iteration": 3}

    class Snapshot:
        values = completed
        next = ()

    class FakeGraph:
        async def aget_state(self, _config):
            return Snapshot()

        async def ainvoke(self, _state, _config):
            raise AssertionError("completed checkpoint must not execute again")

    monkeypatch.setattr(runtime_module, "react_graph", FakeGraph())

    async def collect():
        return [
            item
            async for item in run_rag_runtime(
                query="question",
                user_id=1,
                request_id="completed-run",
                session_id="session-1",
                stream_events=False,
            )
        ]

    assert asyncio.run(collect()) == [{"checkpoint": completed}]


def test_checkpoint_manager_creates_persistent_sqlite_file(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoints" / "react.sqlite3"
    monkeypatch.setattr(settings, "AGENT_CHECKPOINT_PATH", str(checkpoint_path))
    manager = ReactCheckpointManager()

    async def start_and_close():
        graph = await manager.start()
        assert graph is not None
        await manager.close()

    asyncio.run(start_and_close())
    assert Path(checkpoint_path).is_file()
