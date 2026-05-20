# -*- coding: utf-8 -*-
"""
User-related database models.
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.sql import func

from db.mysql.connection import Base


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
    summary = Column(Text, nullable=True)
    context = Column(JSON, nullable=True)
    category = Column(String(50), nullable=True, index=True)


class MessageTable(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    branch_id = Column(Integer, ForeignKey("conversation_branches.id"), nullable=True, index=True)


class ConversationBranchTable(Base):
    __tablename__ = "conversation_branches"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, index=True)
    branch_name = Column(String(255), nullable=True)
    parent_branch_id = Column(Integer, ForeignKey("conversation_branches.id"), nullable=True, index=True)
    parent_message_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True)

