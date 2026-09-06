from backend.models.database.user import LongTermMemoryTable
from backend.models.schemas.memory import LongTermMemoryPatch


def test_long_term_memory_governance_fields_and_operations_are_typed():
    columns = set(LongTermMemoryTable.__table__.columns.keys())
    assert {"memory_id", "memory_type", "content", "scope_type", "source", "confidence", "status", "last_confirmed_at", "expires_at", "created_at", "updated_at"}.issubset(columns)
    payload = LongTermMemoryPatch(operation="confirm", request_id="req-1")
    assert payload.operation == "confirm"
    assert LongTermMemoryTable.source.default.arg == "inferred"
