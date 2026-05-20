# -*- coding: utf-8 -*-
"""
Knowledge-base related database models.
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.sql import func

from db.mysql.connection import Base
from models.schemas.knowledge import DocumentStatus, ModalityType


class KnowledgeBaseTable(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    acl_tags = Column(JSON, nullable=True)
    visibility = Column(String(20), default="private", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DocumentTable(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    doc_id = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    file_size = Column(Integer, nullable=False)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    modality = Column(SQLEnum(ModalityType), default=ModalityType.TEXT, nullable=False)
    language = Column(String(10), default="zh")
    total_chunks = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ChunkTable(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    doc_id = Column(String(64), nullable=False, index=True)
    chunk_id = Column(String(64), unique=True, nullable=False, index=True)
    parent_id = Column(String(64), nullable=True, index=True)
    is_parent = Column(Boolean, default=False, nullable=False, index=True)
    modality = Column(SQLEnum(ModalityType), default=ModalityType.TEXT, nullable=False)
    language = Column(String(10), default="zh")
    title = Column(String(255), nullable=True)
    section_path = Column(String(500), nullable=True)
    page_no = Column(Integer, nullable=True)
    content = Column(Text, nullable=True)
    sparse_vector = Column(JSON, nullable=True)
    caption = Column(Text, nullable=True)
    source_uri = Column(String(500), nullable=True)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DocumentIndexTaskTable(Base):
    __tablename__ = "document_index_tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    doc_id = Column(String(64), nullable=False, unique=True, index=True)
    status = Column(String(20), default="pending", nullable=False)
    progress = Column(Integer, default=0)
    current_step = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class KnowledgeBasePermissionTable(Base):
    __tablename__ = "kb_permissions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    permission_type = Column(String(20), nullable=False)
    granted_by = Column(Integer, nullable=False)
    granted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
