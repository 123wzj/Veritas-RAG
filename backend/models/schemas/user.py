# -*- coding: utf-8 -*-
"""
用户相关数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = None


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """用户更新模型"""
    email: Optional[str] = None
    preferences: Optional[dict] = None


class UserInDB(UserBase):
    """数据库中的用户模型"""
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class User(UserInDB):
    """返回给前端的用户模型（不包含敏感信息）"""
    pass


class UserProfile(BaseModel):
    """用户画像模型（用于记忆模块）"""
    user_id: int
    preferred_language: str = "zh-CN"
    interests: list[str] = []
    interaction_style: str = "concise"  # concise, detailed, friendly
    frequently_asked_topics: list[str] = []
    long_term_facts: list[str] = []
    working_preferences: dict[str, Any] = {}


class UserProfileUpdate(BaseModel):
    """用户画像更新模型"""
    preferred_language: str = "zh-CN"
    interests: list[str] = []
    interaction_style: str = "concise"
    frequently_asked_topics: list[str] = []
    long_term_facts: list[str] = []
    working_preferences: dict[str, Any] = {}


class SessionContext(BaseModel):
    """会话上下文模型"""
    session_id: str
    user_id: int
    kb_id: Optional[int] = None
    created_at: datetime
    last_active: datetime
    message_count: int = 0
    title: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    archived: bool = False


class SessionRenameRequest(BaseModel):
    """会话重命名请求"""
    title: str = Field(..., min_length=1, max_length=100)
