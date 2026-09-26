import asyncio
from pathlib import Path
from typing import TypedDict

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

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
        values = {"iteration": 2, "events": [{"event": "old-1"}, {"event": "old-2"}]}
        next = ("decide",)

    class FakeGraph:
        async def aget_state(self, config):
            captured["state_config"] = config
            return Snapshot()

        async def ainvoke(self, state, config):
            captured["input"] = state
            captured["invoke_config"] = config
            return {"final_answer": "resumed"}

        async def aupdate_state(self, config, values):
            captured["updated_config"] = config
            captured["updated_values"] = values

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
    assert captured["updated_values"] == {
        "trace_attempt_no": 1,
        "trace_event_offset": 2,
    }


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


def test_real_sqlite_checkpoint_resumes_failed_node_after_reopen(tmp_path):
    checkpoint_path = tmp_path / "real-resume.sqlite3"
    calls = {"prepare": 0, "unstable": 0}

    class State(TypedDict, total=False):
        value: int
        trace_attempt_no: int

    async def prepare(state: State):
        calls["prepare"] += 1
        return {"value": int(state.get("value") or 0) + 1}

    async def unstable(state: State):
        calls["unstable"] += 1
        if calls["unstable"] == 1:
            raise RuntimeError("simulated crash")
        return {"value": int(state.get("value") or 0) + 1}

    def build(saver):
        workflow = StateGraph(State)
        workflow.add_node("prepare", prepare)
        workflow.add_node("unstable", unstable)
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "unstable")
        workflow.add_edge("unstable", END)
        return workflow.compile(checkpointer=saver)

    async def scenario():
        config = {"configurable": {"thread_id": "user:1:run:crash"}}
        first_connection = await aiosqlite.connect(str(checkpoint_path))
        first_saver = AsyncSqliteSaver(first_connection)
        await first_saver.setup()
        first_graph = build(first_saver)
        try:
            await first_graph.ainvoke({"value": 0}, config=config)
        except RuntimeError as exc:
            assert "simulated crash" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("first execution must fail")
        await first_connection.close()

        second_connection = await aiosqlite.connect(str(checkpoint_path))
        second_saver = AsyncSqliteSaver(second_connection)
        await second_saver.setup()
        second_graph = build(second_saver)
        snapshot = await second_graph.aget_state(config)
        assert snapshot.next == ("unstable",)
        await second_graph.aupdate_state(config, {"trace_attempt_no": 2})
        result = await second_graph.ainvoke(None, config=config)
        await second_connection.close()
        return result

    assert asyncio.run(scenario())["value"] == 2
    assert calls == {"prepare": 1, "unstable": 2}
