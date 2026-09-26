from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.models.database.user import (
    ContextManifestTable,
    RAGEventTable,
    RAGRunAttemptTable,
    RAGRunTable,
    RAGSpanTable,
)


class TraceService:
    def start_run(
        self,
        db: Session,
        *,
        request_id: str,
        user_id: int,
        session_id: str,
        kb_id: Optional[int],
        runtime_mode: str = "react",
        budget_profile: Optional[str] = None,
    ) -> RAGRunTable:
        run = db.query(RAGRunTable).filter(RAGRunTable.request_id == request_id).first()
        if run:
            if run.user_id != user_id or run.session_id != session_id:
                raise PermissionError("request_id belongs to another user or session")
            if run.final_status != "completed":
                run.final_status = "running"
                # Prior errors remain immutable in rag_run_attempts.
                run.error = None
                run.completed_at = None
            return run
        run = RAGRunTable(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            kb_id=kb_id,
            final_status="running",
            runtime_mode=runtime_mode,
            budget_profile=budget_profile,
        )
        db.add(run)
        db.flush()
        return run

    def start_attempt(self, db: Session, *, request_id: str) -> RAGRunAttemptTable:
        # Serialize concurrent resume requests for the same run before choosing
        # the next attempt number.
        (
            db.query(RAGRunTable)
            .filter(RAGRunTable.request_id == request_id)
            .with_for_update()
            .first()
        )
        latest = (
            db.query(RAGRunAttemptTable)
            .filter(RAGRunAttemptTable.request_id == request_id)
            .order_by(RAGRunAttemptTable.attempt_no.desc())
            .first()
        )
        attempt_no = int(latest.attempt_no or 0) + 1 if latest else 1
        attempt = RAGRunAttemptTable(
            request_id=request_id,
            attempt_no=attempt_no,
            status="running",
            resumed=bool(latest),
        )
        db.add(attempt)
        db.flush()
        return attempt

    def finish_attempt(
        self,
        db: Session,
        *,
        request_id: str,
        attempt_no: int,
        status: str,
        stop_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[RAGRunAttemptTable]:
        attempt = (
            db.query(RAGRunAttemptTable)
            .filter(
                RAGRunAttemptTable.request_id == request_id,
                RAGRunAttemptTable.attempt_no == attempt_no,
            )
            .first()
        )
        if not attempt:
            return None
        attempt.status = status
        attempt.stop_reason = stop_reason
        attempt.error = error
        attempt.ended_at = datetime.now(timezone.utc)
        return attempt

    def start_span(
        self,
        db: Session,
        *,
        request_id: str,
        attempt_no: int,
        span_name: str,
        node_name: Optional[str] = None,
        span_kind: str = "agent_node",
        metadata: Optional[dict[str, Any]] = None,
    ) -> RAGSpanTable:
        span = (
            db.query(RAGSpanTable)
            .filter(
                RAGSpanTable.request_id == request_id,
                RAGSpanTable.attempt_no == attempt_no,
                RAGSpanTable.span_name == span_name,
            )
            .first()
        )
        if span is None:
            span = RAGSpanTable(
                request_id=request_id,
                attempt_no=attempt_no,
                span_name=span_name,
            )
            db.add(span)
        span.node_name = node_name
        span.span_kind = span_kind
        span.status = "running"
        span.started_at = datetime.now(timezone.utc)
        span.ended_at = None
        span.metadata_json = metadata or {}
        span.error = None
        db.flush()
        return span

    def record_span(
        self,
        db: Session,
        *,
        request_id: str,
        span_name: str,
        attempt_no: int = 1,
        node_name: Optional[str] = None,
        span_kind: str = "agent_node",
        status: str = "completed",
        latency_ms: Optional[int] = None,
        model_name: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> RAGSpanTable:
        span = (
            db.query(RAGSpanTable)
            .filter(
                RAGSpanTable.request_id == request_id,
                RAGSpanTable.attempt_no == attempt_no,
                RAGSpanTable.span_name == span_name,
            )
            .first()
        )
        if span is None:
            span = RAGSpanTable(
                request_id=request_id,
                attempt_no=attempt_no,
                span_name=span_name,
                started_at=datetime.now(timezone.utc),
            )
            db.add(span)
        span.node_name = node_name or span.node_name
        span.span_kind = span_kind or span.span_kind
        span.status = status
        span.ended_at = datetime.now(timezone.utc)
        span.latency_ms = latency_ms
        span.model_name = model_name
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.metadata_json = metadata or span.metadata_json or {}
        span.error = error
        return span

    def record_event(
        self,
        db: Session,
        *,
        request_id: str,
        attempt_no: int,
        event_name: str,
        node_name: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RAGEventTable:
        latest = (
            db.query(RAGEventTable.sequence_no)
            .filter(
                RAGEventTable.request_id == request_id,
                RAGEventTable.attempt_no == attempt_no,
            )
            .order_by(RAGEventTable.sequence_no.desc())
            .first()
        )
        sequence_no = int(latest[0]) + 1 if latest else 1
        event = RAGEventTable(
            request_id=request_id,
            attempt_no=attempt_no,
            sequence_no=sequence_no,
            event_name=event_name,
            node_name=node_name,
            status=status,
            metadata_json=metadata or {},
        )
        db.add(event)
        return event

    def record_context_manifest(
        self,
        db: Session,
        *,
        request_id: str,
        attempt_no: int,
        node_name: str,
        iteration: int,
        manifest: dict[str, Any],
        model_name: Optional[str] = None,
    ) -> ContextManifestTable:
        row = (
            db.query(ContextManifestTable)
            .filter(
                ContextManifestTable.request_id == request_id,
                ContextManifestTable.attempt_no == attempt_no,
                ContextManifestTable.node_name == node_name,
                ContextManifestTable.iteration == iteration,
            )
            .first()
        )
        if row is None:
            row = ContextManifestTable(
                request_id=request_id,
                attempt_no=attempt_no,
                node_name=node_name,
                iteration=iteration,
            )
            db.add(row)
        usage = manifest.get("token_usage") or {}
        row.model_name = model_name
        row.token_budget = int(usage.get("budget") or 0)
        row.token_used = int(usage.get("used") or 0)
        row.loaded_sections = manifest.get("loaded_sections") or []
        row.omitted_sections = manifest.get("omitted_sections") or []
        row.section_usage = usage.get("by_section") or {}
        return row

    def finish_run(
        self,
        db: Session,
        *,
        request_id: str,
        final_status: str,
        total_latency_ms: int,
        route_type: Optional[str] = None,
        answer_mode: Optional[str] = None,
        reflection_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        selected_evidence_ids: Optional[list[str]] = None,
        selected_memory_ids: Optional[list[str]] = None,
        runtime_mode: Optional[str] = None,
        iteration_count: int = 0,
        stop_reason: Optional[str] = None,
        tool_call_count: int = 0,
        budget_profile: Optional[str] = None,
        shadow_metrics: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[RAGRunTable]:
        run = db.query(RAGRunTable).filter(RAGRunTable.request_id == request_id).first()
        if not run:
            return None
        run.final_status = final_status
        run.total_latency_ms = total_latency_ms
        run.route_type = route_type
        run.answer_mode = answer_mode
        run.reflection_count = reflection_count
        run.input_tokens = input_tokens
        run.output_tokens = output_tokens
        run.selected_evidence_ids = selected_evidence_ids or []
        run.selected_memory_ids = selected_memory_ids or []
        if runtime_mode:
            run.runtime_mode = runtime_mode
        run.iteration_count = iteration_count
        run.stop_reason = stop_reason
        run.tool_call_count = tool_call_count
        if budget_profile:
            run.budget_profile = budget_profile
        run.shadow_metrics = shadow_metrics or {}
        run.error = error
        run.completed_at = datetime.now(timezone.utc)
        return run


trace_service = TraceService()
