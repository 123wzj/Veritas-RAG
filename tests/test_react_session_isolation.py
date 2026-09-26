import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agent.memory.service import ReactMemoryService
from backend.db.mysql.connection import Base
from backend.models.database.user import ConversationBranchTable, MessageTable, SessionTable
from backend.services.memory.memory_service import MemoryService


def test_memory_loader_rejects_cross_user_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[SessionTable.__table__])
    db = sessionmaker(bind=engine)()
    db.add(SessionTable(session_id="owned-by-1", user_id=1, message_count=0))
    db.commit()
    with pytest.raises(PermissionError):
        asyncio.run(ReactMemoryService().load(
            run_id="r",
            user_id=2,
            session_id="owned-by-1",
            kb_id=None,
            query="q",
            request_id="r",
            db=db,
        ))
    db.close()


def test_recent_messages_do_not_cross_conversation_branches():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        SessionTable.__table__, ConversationBranchTable.__table__, MessageTable.__table__,
    ])
    db = sessionmaker(bind=engine)()
    db.add(SessionTable(session_id="s1", user_id=1, message_count=3))
    db.flush()
    branch_a = ConversationBranchTable(session_id="s1", branch_name="a", is_active=True)
    branch_b = ConversationBranchTable(session_id="s1", branch_name="b", is_active=False)
    db.add_all([branch_a, branch_b])
    db.flush()
    db.add_all([
        MessageTable(session_id="s1", branch_id=branch_a.id, role="user", content="branch-a", request_id="a"),
        MessageTable(session_id="s1", branch_id=branch_b.id, role="user", content="branch-b", request_id="b"),
        MessageTable(session_id="s1", role="user", content="main", request_id="main"),
    ])
    db.commit()
    service = MemoryService()
    branch_messages = service.get_recent_messages("s1", db, branch_id=branch_a.id)
    main_messages = service.get_recent_messages("s1", db)
    assert [item["content"] for item in branch_messages] == ["branch-a"]
    assert [item["content"] for item in main_messages] == ["main"]
    db.close()
