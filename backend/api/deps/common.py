# -*- coding: utf-8 -*-
"""
API 依赖注入模块。
"""

from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.db.mysql.connection import get_db
from backend.models.database.user import UserTable


def get_current_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> Optional[UserTable]:
    """
    获取当前用户（可选认证）。

    当前开发模式下，如未接入真实鉴权则返回 None。
    """
    if credentials is None:
        return None
    return None


def get_required_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> UserTable:
    """
    获取当前用户（必需认证）。

    开发模式下固定回落到默认用户 ID=1。
    """
    default_user = db.query(UserTable).filter(UserTable.id == 1).first()

    if not default_user:
        from backend.models.database.user import UserProfileTable

        default_user = UserTable(
            id=1,
            username="default_user",
            email="default@example.com",
            hashed_password="test_password",
            is_active=True,
        )
        db.add(default_user)
        db.commit()

        profile = UserProfileTable(
            user_id=1,
            preferred_language="zh-CN",
            interaction_style="detailed",
        )
        db.add(profile)
        db.commit()
        db.refresh(default_user)

    return default_user


def get_redis_cache():
    """获取 Redis 缓存依赖。"""
    from backend.db.redis.connection import redis_client
    return redis_client


def get_vector_client():
    """获取 Chroma 客户端依赖。"""
    from backend.db.chroma.connection import chroma_client
    return chroma_client
