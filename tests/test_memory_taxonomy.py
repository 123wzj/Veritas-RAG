from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.agent.context.builder import ReactContextBuilder
from backend.agent.memory.service import ReactMemoryService
from backend.db.mysql.connection import Base
from backend.models.database.user import (
    ConversationBranchTable,
    LongTermMemoryTable,
    MemoryUpdateLogTable,
    MessageTable,
    SessionMemoryTable,
    SessionTable,
)
from backend.services.memory.memory_service import MemoryService
from backend.services.chat.branch import ConversationBranchService


def _database():
    engine = create_engine("sqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _record: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(
        engine,
        tables=[
            SessionTable.__table__,
            ConversationBranchTable.__table__,
            MessageTable.__table__,
            SessionMemoryTable.__table__,
            LongTermMemoryTable.__table__,
            MemoryUpdateLogTable.__table__,
        ],
    )
    return sessionmaker(bind=engine)()


def test_business_type_maps_to_cognitive_category_and_json_payload():
    action = MemoryService()._normalize_long_term_actions(
        [
            {
                "action": "create",
                "scope_type": "user",
                "memory_type": "user_preference",
                "content": "回答时先给结论",
                "source": "inferred",
            }
        ],
        kb_id=None,
    )[0]

    assert action["memory_category"] == "procedural"
    assert action["memory_payload"]["instruction"] == "回答时先给结论"
    assert action["status"] == "pending_confirmation"
    assert datetime.fromisoformat(action["stale_at"]) > datetime.now()


def test_fork_copies_messages_and_summary_into_independent_session():
    db = _database()
    try:
        db.add(SessionTable(session_id="s1", user_id=1, title="主会话", message_count=2))
        db.flush()
        first = MessageTable(session_id="s1", role="user", content="问题", request_id="r1")
        second = MessageTable(session_id="s1", role="assistant", content="回答", request_id="r1")
        db.add_all([first, second])
        db.flush()
        db.add(
            SessionMemoryTable(
                session_id="s1",
                summary={"session_goal": "主会话", "confirmed_decisions": [], "discarded_ideas": [], "open_questions": []},
                summary_through_message_id=second.id,
                version=1,
            )
        )
        db.commit()

        branch = ConversationBranchService().fork_from_message(
            session_id="s1",
            from_message_id=second.id,
            branch_name="方案 A",
            db=db,
        )

        assert branch.forked_session_id != "s1"
        copied_messages = db.query(MessageTable).filter_by(
            session_id=branch.forked_session_id
        ).order_by(MessageTable.id).all()
        copied_memory = db.query(SessionMemoryTable).filter_by(
            session_id=branch.forked_session_id
        ).one()
        source_memory = db.query(SessionMemoryTable).filter_by(session_id="s1").one()
        assert [item.content for item in copied_messages] == ["问题", "回答"]
        assert all(item.branch_id is None for item in copied_messages)
        assert copied_memory.summary == source_memory.summary
        assert copied_memory.summary_through_message_id == copied_messages[-1].id
    finally:
        db.close()


def test_deleting_fork_removes_its_session_but_preserves_descendant_session():
    db = _database()
    service = ConversationBranchService()
    try:
        db.add(SessionTable(session_id="s1", user_id=1, title="主会话"))
        db.flush()
        root_message = MessageTable(
            session_id="s1", role="user", content="根消息", request_id="root"
        )
        db.add(root_message)
        db.flush()
        db.add(SessionMemoryTable(session_id="s1", summary={}, version=1))
        db.commit()

        first_fork = service.fork_from_message(
            session_id="s1",
            from_message_id=root_message.id,
            branch_name="第一层",
            db=db,
        )
        first_fork_session_id = first_fork.forked_session_id
        copied_message = db.query(MessageTable).filter_by(
            session_id=first_fork_session_id
        ).one()
        second_fork = service.fork_from_message(
            session_id=first_fork_session_id,
            from_message_id=copied_message.id,
            branch_name="第二层",
            db=db,
        )
        second_fork_id = second_fork.id
        descendant_session_id = second_fork.forked_session_id

        assert service.delete_branch(first_fork.id, db=db) is True
        assert db.query(SessionTable).filter_by(
            session_id=first_fork_session_id
        ).first() is None
        assert db.query(SessionTable).filter_by(
            session_id=descendant_session_id
        ).one().title == "第二层"
        assert db.query(ConversationBranchTable).filter_by(
            id=second_fork_id
        ).first() is None
    finally:
        db.close()


def test_working_memory_decomposes_multi_part_question():
    value = ReactMemoryService.initial_working_memory(
        run_id="r1",
        query="当前记忆怎么存？什么时候加载？另外冲突如何处理？",
    )
    assert len(value["answer_slots"]) == 3
    assert value["unresolved_slots"] == ["slot-1", "slot-2", "slot-3"]


def test_context_builder_keeps_truncated_structures_as_valid_json():
    builder = ReactContextBuilder()
    value = [{"id": index, "content": "内容" * 100} for index in range(10)]
    fitted = builder._fit_value(value, 80)
    assert isinstance(fitted, list)
    assert len(fitted) < len(value)
    assert all(isinstance(item, dict) for item in fitted)
