# -*- coding: utf-8 -*-
"""
User-related database models.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from backend.db.mysql.connection import Base


class UserTable(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    preferences = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UserProfileTable(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, unique=True, index=True)
    preferred_language = Column(String(10), default="zh-CN")
    interests = Column(JSON, nullable=True)
    interaction_style = Column(String(20), default="concise")
    frequently_asked_topics = Column(JSON, nullable=True)
    long_term_facts = Column(JSON, nullable=True)
    working_preferences = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SessionTable(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    kb_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_active = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    message_count = Column(Integer, default=0)
    title = Column(String(100), nullable=True)
    # Legacy compatibility field. New memory summaries live in session_memories.
    summary = Column(Text, nullable=True)
    context = Column(JSON, nullable=True)
    category = Column(String(50), nullable=True, index=True)
    archived = Column(Boolean, default=False, nullable=False, index=True)


class MessageTable(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    token_count = Column(Integer, default=0)
    request_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    branch_id = Column(Integer, ForeignKey("conversation_branches.id"), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint("request_id", "role", name="uq_messages_request_role"),
    )


class ConversationBranchTable(Base):
    __tablename__ = "conversation_branches"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    branch_name = Column(String(255), nullable=True)
    parent_branch_id = Column(Integer, ForeignKey("conversation_branches.id"), nullable=True, index=True)
    parent_message_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True)


class SessionMemoryTable(Base):
    __tablename__ = "session_memories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id"),
        nullable=False,
        unique=True,
        index=True,
    )
    summary = Column(JSON, nullable=True)
    summary_text = Column(Text, nullable=True)
    summary_through_message_id = Column(Integer, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class LongTermMemoryTable(Base):
    __tablename__ = "long_term_memories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    memory_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    kb_id = Column(Integer, nullable=True, index=True)
    scope_type = Column(String(20), default="user", nullable=False, index=True)
    memory_type = Column(String(40), nullable=False, index=True)
    content = Column(Text, nullable=False)
    normalized_key = Column(String(255), nullable=False)
    keywords = Column(JSON, nullable=True)
    confidence = Column(Float, default=0.8, nullable=False)
    source = Column(String(30), default="inferred", nullable=False, index=True)
    status = Column(String(20), default="active", nullable=False, index=True)
    last_confirmed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    source_session_id = Column(String(64), nullable=True, index=True)
    source_message_id = Column(Integer, nullable=True, index=True)
    superseded_by = Column(String(64), nullable=True)
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    valid_from = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    valid_to = Column(DateTime(timezone=True), nullable=True)
    memory_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_long_term_memory_scope",
            "user_id",
            "kb_id",
            "scope_type",
            "status",
        ),
        Index(
            "ix_long_term_memory_lookup",
            "user_id",
            "memory_type",
            "normalized_key",
        ),
    )


class MemoryUpdateLogTable(Base):
    __tablename__ = "memory_update_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    memory_id = Column(String(64), nullable=True, index=True)
    action = Column(String(30), nullable=False)
    before_value = Column(JSON, nullable=True)
    after_value = Column(JSON, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "memory_id",
            "action",
            name="uq_memory_log_request_action",
        ),
    )


class AnswerFeedbackTable(Base):
    __tablename__ = "answer_feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    rating = Column(String(20), nullable=False)
    comment = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("request_id", "user_id", name="uq_answer_feedback_request_user"),
    )


class RAGRunTable(Base):
    __tablename__ = "rag_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    kb_id = Column(Integer, nullable=True, index=True)
    route_type = Column(String(40), nullable=True)
    final_status = Column(String(30), nullable=False, default="running")
    answer_mode = Column(String(40), nullable=True)
    reflection_count = Column(Integer, nullable=False, default=0)
    total_latency_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    selected_evidence_ids = Column(JSON, nullable=True)
    selected_memory_ids = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class RAGSpanTable(Base):
    __tablename__ = "rag_spans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    span_name = Column(String(60), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="completed")
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    model_name = Column(String(100), nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    metadata_json = Column("metadata", JSON, nullable=True)
    error = Column(Text, nullable=True)
