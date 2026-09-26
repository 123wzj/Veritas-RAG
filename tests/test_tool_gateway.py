import asyncio
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agent.schemas import ToolCallRequest, ToolExecutionContext, ToolObservation
from backend.agent.tools.base import AgentTool, ToolSpec
from backend.agent.tools.gateway import ToolGateway
from backend.agent.tools.registry import ToolRegistry
from backend.db.mysql.connection import Base
from backend.models.database.user import AgentObservationTable, AgentToolCallTable, SessionTable


class FakeTool(AgentTool):
    spec = ToolSpec(name="fake_tool", description="test", input_schema={"type": "object"})

    def __init__(self):
        self.calls = 0
        self.last_arguments = None

    async def execute(self, arguments, context):
        self.calls += 1
        self.last_arguments = arguments
        return ToolObservation(
            observation_id=str(uuid.uuid4()),
            tool_call_id="",
            tool_name="fake_tool",
            status="success",
            summary="ok",
            next_hint="answer",
        )


class FlakyTool(FakeTool):
    spec = ToolSpec(
        name="flaky_tool",
        description="retry test",
        input_schema={"type": "object"},
        max_retries=1,
    )

    async def execute(self, arguments, context):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return ToolObservation(
            observation_id=str(uuid.uuid4()),
            tool_call_id="",
            tool_name="flaky_tool",
            status="success",
            summary="recovered",
            next_hint="answer",
        )


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        SessionTable.__table__, AgentToolCallTable.__table__, AgentObservationTable.__table__,
    ])
    db = sessionmaker(bind=engine)()
    db.add_all([
        SessionTable(session_id="s1", user_id=1, message_count=0),
        SessionTable(session_id="s2", user_id=2, message_count=0),
    ])
    db.commit()
    return db


def test_gateway_strips_protected_identity_and_reuses_persisted_result():
    db = _db()
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool)
    gateway = ToolGateway(registry)
    context = ToolExecutionContext(
        run_id="run-1", request_id="req-1", user_id=1, session_id="s1"
    )
    call = ToolCallRequest(
        tool_call_id="call-1",
        tool_name="fake_tool",
        arguments={"query": "q", "user_id": 2, "session_id": "s2", "kb_id": 999},
    )
    cache = {}
    first, _ = asyncio.run(gateway.execute(
        call, context=context, db=db, prior_calls=[], idempotency_cache=cache
    ))
    cache.clear()
    second, _ = asyncio.run(gateway.execute(
        call, context=context, db=db, prior_calls=[], idempotency_cache=cache
    ))
    assert first.status == "success"
    assert second.reused is True
    assert tool.calls == 1
    assert tool.last_arguments == {"query": "q"}
    db.close()


def test_gateway_denies_session_owned_by_another_user():
    db = _db()
    registry = ToolRegistry()
    registry.register(FakeTool())
    gateway = ToolGateway(registry)
    observation, _ = asyncio.run(gateway.execute(
        ToolCallRequest(tool_call_id="call-x", tool_name="fake_tool"),
        context=ToolExecutionContext(
            run_id="run-x", request_id="req-x", user_id=2, session_id="s1"
        ),
        db=db,
        prior_calls=[],
        idempotency_cache={},
    ))
    assert observation.status == "denied"
    assert observation.error_code == "session_not_owned"
    db.close()


def test_tool_idempotency_does_not_cross_runs():
    db = _db()
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool)
    gateway = ToolGateway(registry)
    call = ToolCallRequest(tool_call_id="same-call", tool_name="fake_tool")
    for run_id in ("run-1", "run-2"):
        observation, _ = asyncio.run(gateway.execute(
            call,
            context=ToolExecutionContext(
                run_id=run_id, request_id=run_id, user_id=1, session_id="s1"
            ),
            db=db,
            prior_calls=[],
            idempotency_cache={},
        ))
        assert observation.status == "success"
    assert tool.calls == 2
    db.close()


def test_gateway_retries_only_retryable_adapter_failures():
    db = _db()
    registry = ToolRegistry()
    tool = FlakyTool()
    registry.register(tool)
    observation, _ = asyncio.run(ToolGateway(registry).execute(
        ToolCallRequest(tool_call_id="flaky", tool_name="flaky_tool"),
        context=ToolExecutionContext(
            run_id="retry-run", request_id="retry-run", user_id=1, session_id="s1"
        ),
        db=db,
        prior_calls=[],
        idempotency_cache={},
    ))
    row = db.query(AgentToolCallTable).filter(
        AgentToolCallTable.tool_call_id == "flaky"
    ).one()
    assert observation.status == "success"
    assert tool.calls == 2
    assert row.retry_count == 1
    db.close()
