from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.mysql.connection import Base
from backend.models.database.user import LongTermMemoryTable
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
