from datetime import datetime

from sqlalchemy import create_engine
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


def _database():
    engine = create_engine("sqlite:///:memory:")
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


def test_branch_summary_updates_without_overwriting_session_summary():
    db = _database()
    try:
        db.add(SessionTable(session_id="s1", user_id=1, title="新对话"))
        branch = ConversationBranchTable(session_id="s1", branch_name="方案 A")
        db.add(branch)
        db.flush()
        db.add(
            SessionMemoryTable(
                session_id="s1",
                summary={"session_goal": "主会话", "confirmed_decisions": [], "discarded_ideas": [], "open_questions": []},
                version=1,
            )
        )
        db.commit()

        MemoryService().apply_memory_update_plan(
            {
                "request_id": "req-branch",
                "user_id": 1,
                "session_id": "s1",
                "branch_id": branch.id,
                "base_memory_version": 1,
                "summary_updated": True,
                "session_summary": {
                    "session_goal": "分支方案 A",
                    "confirmed_decisions": ["使用方案 A"],
                    "discarded_ideas": [],
                    "open_questions": [],
                },
                "long_term_actions": [],
                "allowed_target_memory_ids": [],
            },
            db=db,
        )

        db.flush()
        db.refresh(branch)
        session_memory = db.query(SessionMemoryTable).filter_by(session_id="s1").one()
        assert branch.summary["session_goal"] == "分支方案 A"
        assert branch.memory_version == 2
        assert session_memory.summary["session_goal"] == "主会话"
        assert session_memory.version == 1
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
