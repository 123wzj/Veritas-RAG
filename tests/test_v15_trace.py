from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.mysql.connection import Base
from backend.models.database.user import (
    ContextManifestTable,
    RAGEventTable,
    RAGRunAttemptTable,
    RAGRunTable,
    RAGSpanTable,
)
from backend.services.trace_service import TraceService


def test_trace_models_keep_sensitive_data_out_of_contract():
    assert {"request_id", "route_type", "selected_evidence_ids", "selected_memory_ids"}.issubset(RAGRunTable.__table__.columns.keys())
    assert {"runtime_mode", "iteration_count", "stop_reason", "tool_call_count", "budget_profile", "shadow_metrics"}.issubset(RAGRunTable.__table__.columns.keys())
    assert {"request_id", "span_name", "latency_ms", "metadata"}.issubset(RAGSpanTable.__table__.columns.keys())
    assert {"attempt_no", "node_name", "span_kind"}.issubset(RAGSpanTable.__table__.columns.keys())
    assert "content" not in RAGSpanTable.__table__.columns.keys()
    assert hasattr(TraceService, "record_span")


def test_trace_preserves_attempts_events_spans_and_context_manifests():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        RAGRunTable.__table__,
        RAGSpanTable.__table__,
        RAGRunAttemptTable.__table__,
        RAGEventTable.__table__,
        ContextManifestTable.__table__,
    ])
    db = sessionmaker(bind=engine)()
    service = TraceService()
    service.start_run(
        db,
        request_id="trace-1",
        user_id=1,
        session_id="session-1",
        kb_id=7,
    )
    first = service.start_attempt(db, request_id="trace-1")
    service.start_span(
        db,
        request_id="trace-1",
        attempt_no=first.attempt_no,
        span_name="react.decide.0",
        node_name="decide",
    )
    service.record_span(
        db,
        request_id="trace-1",
        attempt_no=first.attempt_no,
        span_name="react.decide.0",
        node_name="decide",
        latency_ms=15,
        input_tokens=100,
        output_tokens=20,
    )
    service.record_event(
        db,
        request_id="trace-1",
        attempt_no=first.attempt_no,
        event_name="agent.decision",
        node_name="decide",
    )
    service.record_context_manifest(
        db,
        request_id="trace-1",
        attempt_no=first.attempt_no,
        node_name="decide",
        iteration=0,
        manifest={
            "loaded_sections": ["current_query", "working_memory"],
            "omitted_sections": ["long_term_memory"],
            "token_usage": {"budget": 32000, "used": 100, "by_section": {}},
        },
    )
    service.finish_attempt(
        db,
        request_id="trace-1",
        attempt_no=first.attempt_no,
        status="interrupted",
        stop_reason="test_failure",
        error="boom",
    )
    second = service.start_attempt(db, request_id="trace-1")
    service.finish_attempt(
        db,
        request_id="trace-1",
        attempt_no=second.attempt_no,
        status="completed",
    )
    db.commit()

    assert first.attempt_no == 1
    assert second.attempt_no == 2
    assert second.resumed is True
    assert db.query(RAGRunAttemptTable).count() == 2
    assert db.query(RAGSpanTable).one().status == "completed"
    assert db.query(RAGEventTable).one().event_name == "agent.decision"
    assert db.query(ContextManifestTable).one().token_used == 100
