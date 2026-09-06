from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.models.database.user import RAGRunTable, RAGSpanTable


class TraceService:
    def start_run(self, db: Session, *, request_id: str, user_id: int, session_id: str, kb_id: Optional[int]) -> RAGRunTable:
        run = db.query(RAGRunTable).filter(RAGRunTable.request_id == request_id).first()
        if run:
            return run
        run = RAGRunTable(request_id=request_id, user_id=user_id, session_id=session_id, kb_id=kb_id, final_status="running")
        db.add(run)
        db.flush()
        return run

    def record_span(self, db: Session, *, request_id: str, span_name: str, status: str = "completed", latency_ms: Optional[int] = None, model_name: Optional[str] = None, input_tokens: int = 0, output_tokens: int = 0, metadata: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> RAGSpanTable:
        span = RAGSpanTable(request_id=request_id, span_name=span_name, status=status, ended_at=datetime.utcnow(), latency_ms=latency_ms, model_name=model_name, input_tokens=input_tokens, output_tokens=output_tokens, metadata_json=metadata or {}, error=error)
        db.add(span)
        return span

    def finish_run(self, db: Session, *, request_id: str, final_status: str, total_latency_ms: int, route_type: Optional[str] = None, answer_mode: Optional[str] = None, reflection_count: int = 0, input_tokens: int = 0, output_tokens: int = 0, selected_evidence_ids: Optional[list[str]] = None, selected_memory_ids: Optional[list[str]] = None, error: Optional[str] = None) -> Optional[RAGRunTable]:
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
        run.error = error
        run.completed_at = datetime.utcnow()
        return run


trace_service = TraceService()
