# -*- coding: utf-8 -*-
"""
用户记忆模块 API 路由
"""

from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps.common import get_required_user
from db.mysql.connection import get_db
from models.database.user import SessionTable
from services.memory.memory_service import memory_service

router = APIRouter()


@router.get("/")
async def get_user_memory(
    session_id: Optional[str] = None,
    query: str = "",
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取用户记忆（画像、会话压缩摘要、短期工作记忆、长期结构化记忆）
    """
    memory = memory_service.get_session_memory(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        db=db,
    )
    recent_sessions = memory_service.get_recent_sessions(
        user_id=current_user.id,
        limit=5,
        db=db,
    )
    return {
        "profile": memory.get("profile") or {},
        "session": memory.get("session") or {},
        "session_summary": memory.get("session_summary") or "",
        "working_memory": memory.get("working_memory") or [],
        "open_loops": memory.get("open_loops") or [],
        "recent_conversations": memory.get("recent_conversations") or [],
        "long_term_facts": memory.get("long_term_facts") or [],
        "working_preferences": memory.get("working_preferences") or {},
        "prompt_context": memory.get("prompt_context") or "",
        "recent_sessions": recent_sessions,
    }


@router.post("/sync")
async def sync_user_memory(
    data: Dict[str, Any],
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    同步用户记忆（从指定 query/answer 中更新短期/长期结构化记忆）
    """
    session_id = data.get("session_id")
    query = (data.get("query") or "").strip()
    answer = (data.get("answer") or "").strip()

    if not session_id or not query or not answer:
        raise HTTPException(status_code=400, detail="session_id, query, answer are required")

    session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    memory_service.save_conversation_to_memory(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        answer=answer,
        db=db,
    )

    memory = memory_service.get_session_memory(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        db=db,
    )
    return {
        "message": "Memory synced successfully",
        "user_id": current_user.id,
        "memory": memory,
    }


@router.delete("/sessions/{session_id}")
async def delete_session_memory(
    session_id: str,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    删除指定会话的压缩上下文记忆，不删除原始消息。
    """
    session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    session.summary = None
    session.context = {}
    db.commit()

    return {
        "message": f"Session {session_id} memory deleted",
        "user_id": current_user.id,
    }
