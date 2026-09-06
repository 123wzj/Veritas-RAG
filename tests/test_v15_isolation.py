from backend.models.database.user import MessageTable, SessionTable
from backend.services.memory.memory_service import MemoryService


def test_isolation_contract_has_session_and_user_ownership_keys():
    assert "session_id" in SessionTable.__table__.columns
    assert "user_id" in SessionTable.__table__.columns
    assert "session_id" in MessageTable.__table__.columns
    assert "request_id" in MessageTable.__table__.columns
    assert hasattr(MemoryService, "list_long_term_memories")
