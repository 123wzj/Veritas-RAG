# -*- coding: utf-8 -*-
"""
Chroma 向量数据库连接管理
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

from backend.core.config import settings


class ChromaClientManager:
    """Chroma 客户端封装。"""

    def __init__(self):
        self._client = None

    def get_client(self):
        """获取 PersistentClient。"""
        if self._client is None:
            Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        return self._client

    def get_collection(self, collection_name: Optional[str] = None):
        """获取或创建 Collection。"""
        client = self.get_client()
        name = collection_name or settings.CHROMA_COLLECTION_NAME
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, collection_name: Optional[str] = None):
        client = self.get_client()
        name = collection_name or settings.CHROMA_COLLECTION_NAME
        try:
            client.delete_collection(name)
        except Exception:
            pass

    def close(self):
        """PersistentClient 无显式 close，这里清理引用即可。"""
        self._client = None


chroma_client = ChromaClientManager()
