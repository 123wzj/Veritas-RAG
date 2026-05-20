# -*- coding: utf-8 -*-
"""
记忆相关节点
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.graph.state.state import RAGState
from backend.db.mysql.connection import get_db
from backend.services.memory.memory_service import memory_service


async def load_user_memory(state: RAGState) -> Dict[str, Any]:
    """
    加载用户记忆节点

    加载用户画像、当前会话压缩摘要、短期工作记忆。
    """
    user_id = state.get("user_id", 0)
    session_id = state.get("session_id")
    query = state.get("query", "")

    events = state.get("events", [])
    db: Optional[Session] = None

    try:
        db = next(get_db())
        memory_context = memory_service.get_session_memory(
            user_id=user_id,
            session_id=session_id,
            query=query,
            db=db,
        )

        events.append({
            "event": "memory.loaded",
            "data": {
                "context_keys": list(memory_context.keys()),
                "has_prompt_context": bool(memory_context.get("prompt_context")),
            },
        })
    except Exception as e:
        memory_context = {}
        events.append({
            "event": "memory.load.failed",
            "data": {"error": str(e)},
        })
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass

    return {
        "memory_context": memory_context,
        "events": events,
    }


async def write_memory(state: RAGState) -> Dict[str, Any]:
    """
    写入记忆节点

    将本次交互的信息写入用户短期/长期记忆，并刷新压缩上下文。
    """
    user_id = state.get("user_id", 0)
    session_id = state.get("session_id")
    query = state.get("query", "")
    final_answer = state.get("final_answer", "")

    db: Optional[Session] = None
    try:
        db = next(get_db())
        refreshed_memory = state.get("memory_context") or {}

        if session_id and final_answer:
            memory_service.save_conversation_to_memory(
                user_id=user_id,
                session_id=session_id,
                query=query,
                answer=final_answer,
                db=db,
            )
            refreshed_memory = memory_service.get_session_memory(
                user_id=user_id,
                session_id=session_id,
                query=query,
                db=db,
            )

        return {
            "memory_context": refreshed_memory,
            "events": state.get("events", []) + [{
                "event": "memory.updated",
                "data": {"user_id": user_id, "session_id": session_id},
            }],
        }
    except Exception as e:
        return {
            "events": state.get("events", []) + [{
                "event": "memory.update.failed",
                "data": {"error": str(e)},
            }],
        }
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass
