# -*- coding: utf-8 -*-
"""数据库表模型"""

from backend.models.database.user import (
    LongTermMemoryTable,
    MemoryUpdateLogTable,
    AnswerFeedbackTable,
    MessageTable,
    SessionMemoryTable,
    SessionTable,
    UserProfileTable,
    UserTable,
)
from backend.models.database.knowledge import KnowledgeBaseTable, DocumentTable, ChunkTable, DocumentIndexTaskTable

__all__ = [
    "UserTable",
    "UserProfileTable",
    "SessionTable",
    "MessageTable",
    "SessionMemoryTable",
    "LongTermMemoryTable",
    "MemoryUpdateLogTable",
    "AnswerFeedbackTable",
    "KnowledgeBaseTable",
    "DocumentTable",
    "ChunkTable",
    "DocumentIndexTaskTable",
]
