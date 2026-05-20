# -*- coding: utf-8 -*-
"""
知识库相关数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """文档状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModalityType(str, Enum):
    """数据模态类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"


class PermissionType(str, Enum):
    """权限类型枚举"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class KnowledgeBaseCreate(BaseModel):
    """创建知识库模型"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    acl_tags: Optional[List[str]] = None


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    acl_tags: Optional[List[str]] = None


class KnowledgeBase(BaseModel):
    """知识库模型"""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    acl_tags: Optional[List[str]] = None
    document_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None


class DocumentCreate(BaseModel):
    """创建文档模型"""
    kb_id: int
    filename: str
    file_path: str


class Document(BaseModel):
    """文档模型"""
    id: int
    kb_id: int
    doc_id: str
    filename: str
    file_path: str
    file_size: int
    status: DocumentStatus
    modality: ModalityType
    language: str = "zh"
    total_chunks: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class ChunkMetadata(BaseModel):
    """Chunk 元数据模型"""
    chunk_id: str
    parent_id: Optional[str] = None
    doc_id: str
    modality: ModalityType
    language: str
    title: Optional[str] = None
    section_path: Optional[str] = None
    page_no: Optional[int] = None
    caption: Optional[str] = None
    source_uri: Optional[str] = None


class ChunkWithVector(ChunkMetadata):
    """带向量的 Chunk 模型（用于入库）"""
    content: str
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[dict] = None
    image_vector: Optional[List[float]] = None
