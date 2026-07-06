# -*- coding: utf-8 -*-
"""
Redis 缓存连接管理
"""

import redis
from typing import Optional
from backend.core.config import settings


class RedisClient:
    """Redis 客户端封装"""

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """获取 Redis 客户端（懒加载）"""
        if self._client is None:
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
        return self._client

    def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        return self.client.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """设置缓存"""
        return self.client.set(key, value, ex=ex)

    def delete(self, key: str) -> int:
        """删除缓存"""
        return self.client.delete(key)

    def exists(self, key: str) -> bool:
        """检查 key 是否存在"""
        return self.client.exists(key) > 0

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None


# 全局 Redis 客户端实例
redis_client = RedisClient()
