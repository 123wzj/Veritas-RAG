from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.mysql.connection import Base
from backend.models.database.user import (
    LongTermMemoryTable,
    MemoryUpdateLogTable,
    SessionMemoryTable,
    SessionTable,
)
from backend.services.memory.memory_service import MemoryService


def test_long_term_selection_excludes_expired_stale_and_conflicting_duplicates():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[LongTermMemoryTable.__table__])
    db = sessionmaker(bind=engine)()
    now = datetime.now()
    db.add_all([
        LongTermMemoryTable(
            memory_id="current", user_id=1, scope_type="user", memory_type="user_preference",
            content="用户喜欢详细回答", normalized_key="answer_style", keywords=["详细回答"],
            confidence=0.9, source="confirmed", status="active", conflict_group="cg1",
            valid_from=now - timedelta(days=2), last_confirmed_at=now - timedelta(days=1),
        ),
        LongTermMemoryTable(
            memory_id="conflict", user_id=1, scope_type="user", memory_type="user_preference",
            content="用户喜欢简短回答", normalized_key="answer_style", keywords=["简短回答"],
            confidence=0.8, source="inferred", status="active", conflict_group="cg1",
            valid_from=now - timedelta(days=2),
        ),
        LongTermMemoryTable(
            memory_id="expired", user_id=1, scope_type="user", memory_type="user_profile",
            content="过期事实", normalized_key="expired", keywords=["过期事实"], confidence=1,
            source="confirmed", status="active", valid_from=now - timedelta(days=20),
            expires_at=now - timedelta(days=1),
        ),
    ])
    db.commit()
    selected = MemoryService()._rank_long_term_candidates(
        user_id=1, kb_id=None, query="回答风格 详细回答", db=db
    )
    ids = [item["memory_id"] for item in selected]
    assert "current" in ids
    assert "conflict" not in ids
    assert "expired" not in ids
    db.close()


def test_project_memory_never_crosses_kb_scope():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[LongTermMemoryTable.__table__])
    db = sessionmaker(bind=engine)()
    now = datetime.now()
    db.add_all([
        LongTermMemoryTable(
            memory_id="project-7", user_id=1, kb_id=7, scope_type="project",
            memory_type="project_context", content="项目七规则", normalized_key="rule",
            keywords=["项目七"], confidence=1, source="confirmed", status="active", valid_from=now,
        ),
        LongTermMemoryTable(
            memory_id="project-8", user_id=1, kb_id=8, scope_type="project",
            memory_type="project_context", content="项目八规则", normalized_key="rule",
            keywords=["项目八"], confidence=1, source="confirmed", status="active", valid_from=now,
        ),
    ])
    db.commit()
    ids = [item["memory_id"] for item in MemoryService()._rank_long_term_candidates(
        user_id=1, kb_id=7, query="项目规则", db=db
    )]
    assert "project-7" in ids
    assert "project-8" not in ids
    db.close()


def _conflict_database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[
        SessionTable.__table__,
        SessionMemoryTable.__table__,
        LongTermMemoryTable.__table__,
        MemoryUpdateLogTable.__table__,
    ])
    db = sessionmaker(bind=engine)()
    db.add(SessionTable(session_id="s1", user_id=1, title="冲突测试"))
    db.commit()
    return db


def _replacement_plan(*, request_id, target_id, normalized_key, source, category="procedural"):
    return {
        "request_id": request_id,
        "user_id": 1,
        "session_id": "s1",
        "kb_id": None,
        "summary_updated": False,
        "long_term_actions": [{
            "action": "replace",
            "target_memory_id": target_id,
            "scope_type": "user",
            "memory_category": category,
            "memory_type": "user_preference",
            "content": "用户现在偏好简短回答",
            "normalized_key": normalized_key,
            "keywords": ["简短回答"],
            "confidence": 0.95,
            "source": source,
            "status": "active" if source == "explicit" else "pending_confirmation",
            "conflict_resolution": "supersede",
            "conflict_reason": "用户表达了新的回答风格",
        }],
        "allowed_target_memory_ids": [target_id],
        "reason": "conflict test",
    }


def test_conflict_guard_cannot_replace_unrelated_normalized_key():
    db = _conflict_database()
    try:
        db.add(LongTermMemoryTable(
            memory_id="old", user_id=1, scope_type="user",
            memory_category="procedural", memory_type="user_preference",
            content="用户偏好详细回答", normalized_key="answer_style",
            keywords=["详细回答"], confidence=0.9, source="confirmed",
            status="active", valid_from=datetime.now() - timedelta(days=1),
        ))
        db.commit()
        result = MemoryService().apply_memory_update_plan(
            _replacement_plan(
                request_id="unrelated",
                target_id="old",
                normalized_key="response_language",
                source="explicit",
            ),
            db=db,
        )
        db.commit()
        old = db.query(LongTermMemoryTable).filter_by(memory_id="old").one()
        assert result["actions"] == []
        assert old.status == "active"
        assert db.query(LongTermMemoryTable).count() == 1
    finally:
        db.close()


def test_explicit_newer_memory_supersedes_old_version_but_keeps_history():
    db = _conflict_database()
    try:
        db.add(LongTermMemoryTable(
            memory_id="old", user_id=1, scope_type="user",
            memory_category="procedural", memory_type="user_preference",
            content="用户偏好详细回答", normalized_key="answer_style",
            keywords=["详细回答"], confidence=0.9, source="confirmed",
            status="active", valid_from=datetime.now() - timedelta(days=1),
        ))
        db.commit()
        result = MemoryService().apply_memory_update_plan(
            _replacement_plan(
                request_id="explicit-supersede",
                target_id="old",
                normalized_key="answer_style",
                source="explicit",
            ),
            db=db,
        )
        db.commit()
        old = db.query(LongTermMemoryTable).filter_by(memory_id="old").one()
        replacement_id = result["actions"][0]["replacement_memory_id"]
        replacement = db.query(LongTermMemoryTable).filter_by(
            memory_id=replacement_id
        ).one()
        assert old.status == "superseded"
        assert old.valid_to is not None
        assert old.superseded_by == replacement.memory_id
        assert replacement.status == "active"
    finally:
        db.close()


def test_inferred_conflict_waits_for_user_confirmation():
    db = _conflict_database()
    try:
        db.add(LongTermMemoryTable(
            memory_id="old", user_id=1, scope_type="user",
            memory_category="procedural", memory_type="user_preference",
            content="用户偏好详细回答", normalized_key="answer_style",
            keywords=["详细回答"], confidence=0.9, source="confirmed",
            status="active", valid_from=datetime.now() - timedelta(days=1),
        ))
        db.commit()
        result = MemoryService().apply_memory_update_plan(
            _replacement_plan(
                request_id="inferred-confirm",
                target_id="old",
                normalized_key="answer_style",
                source="inferred",
            ),
            db=db,
        )
        db.commit()
        old = db.query(LongTermMemoryTable).filter_by(memory_id="old").one()
        replacement_id = result["actions"][0]["replacement_memory_id"]
        replacement = db.query(LongTermMemoryTable).filter_by(
            memory_id=replacement_id
        ).one()
        assert old.status == "active"
        assert old.superseded_by is None
        assert replacement.status == "pending_confirmation"
    finally:
        db.close()
