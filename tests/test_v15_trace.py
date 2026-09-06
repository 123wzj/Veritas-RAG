from backend.models.database.user import RAGRunTable, RAGSpanTable
from backend.services.trace_service import TraceService


def test_trace_models_keep_sensitive_data_out_of_contract():
    assert {"request_id", "route_type", "selected_evidence_ids", "selected_memory_ids"}.issubset(RAGRunTable.__table__.columns.keys())
    assert {"request_id", "span_name", "latency_ms", "metadata"}.issubset(RAGSpanTable.__table__.columns.keys())
    assert "content" not in RAGSpanTable.__table__.columns.keys()
    assert hasattr(TraceService, "record_span")
