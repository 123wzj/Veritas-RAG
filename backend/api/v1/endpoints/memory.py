# -*- coding: utf-8 -*-
"""
用户记忆模块 API 路由
"""

import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps.common import get_required_user
from backend.db.mysql.connection import get_db
from backend.models.database.user import MemoryUpdateLogTable, SessionMemoryTable, SessionTable, LongTermMemoryTable
from sqlalchemy import and_, or_
from backend.services.memory.memory_service import memory_service
from backend.models.schemas.memory import LongTermMemoryPatch, MemoryListResponse, LongTermMemoryResponse

router = APIRouter()


@router.get("/")
async def get_user_memory(
    session_id: Optional[str] = None,
    query: str = "",
    kb_id: Optional[int] = None,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取用户记忆（画像、会话压缩摘要、短期工作记忆、长期结构化记忆）
    """
    memory = await memory_service.aget_session_memory(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        db=db,
        kb_id=kb_id,
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
        "session_summary_text": memory.get("session_summary_text") or "",
        "working_memory": memory.get("working_memory") or [],
        "open_loops": memory.get("open_loops") or [],
        "recent_conversations": memory.get("recent_conversations") or [],
        "long_term_facts": memory.get("long_term_facts") or [],
        "long_term_memories": memory.get("long_term_memories") or [],
        "selected_memory_ids": memory.get("selected_memory_ids") or [],
        "working_preferences": memory.get("working_preferences") or {},
        "prompt_context": memory.get("prompt_context") or "",
        "context_token_usage": memory.get("context_token_usage") or {},
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

    request_id = str(data.get("request_id") or uuid.uuid4())
    plan = await memory_service.build_memory_update_plan(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        answer=answer,
        request_id=request_id,
        kb_id=session.kb_id,
        db=db,
    )
    result = memory_service.apply_memory_update_plan(plan, db=db)
    db.commit()

    memory = await memory_service.aget_session_memory(
        user_id=current_user.id,
        session_id=session_id,
        query=query,
        db=db,
        kb_id=session.kb_id,
    )
    return {
        "message": "Memory synced successfully",
        "user_id": current_user.id,
        "update_result": result,
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

    db.query(SessionMemoryTable).filter(
        SessionMemoryTable.session_id == session_id
    ).delete()
    session.context = {}
    db.commit()

    return {
        "message": f"Session {session_id} memory deleted",
        "user_id": current_user.id,
    }


@router.get("/long-term", response_model=MemoryListResponse)
async def list_long_term_memories(
    kb_id: Optional[int] = None,
    status: Optional[str] = "active",
    memory_type: Optional[str] = None,
    scope_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    limit: int = 100,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    page = max(1, page)
    page_size = max(1, min(page_size, 500))
    count_query = db.query(LongTermMemoryTable).filter(LongTermMemoryTable.user_id == current_user.id)
    if kb_id is not None:
        count_query = count_query.filter(or_(LongTermMemoryTable.scope_type == "user", and_(LongTermMemoryTable.scope_type == "project", LongTermMemoryTable.kb_id == kb_id)))
    else:
        count_query = count_query.filter(LongTermMemoryTable.scope_type == "user")
    if status:
        count_query = count_query.filter(LongTermMemoryTable.status == status)
    if memory_type:
        count_query = count_query.filter(LongTermMemoryTable.memory_type == memory_type)
    if scope_type:
        count_query = count_query.filter(LongTermMemoryTable.scope_type == scope_type)
    total = count_query.count()
    items = memory_service.list_long_term_memories(
            user_id=current_user.id,
            kb_id=kb_id,
            status=status,
            memory_type=memory_type,
            scope_type=scope_type,
            page=page,
            page_size=page_size,
            limit=limit,
            db=db,
        )
    return {"items": items, "page": page, "page_size": page_size, "total": total}


@router.patch("/long-term/{memory_id}", response_model=LongTermMemoryResponse)
async def update_long_term_memory(
    memory_id: str,
    data: LongTermMemoryPatch,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    try:
        return memory_service.update_long_term_memory_record(
            user_id=current_user.id,
            memory_id=memory_id,
            content=data.content,
            status=data.status,
            confidence=data.confidence,
            expires_at=data.expires_at,
            request_id=data.request_id or str(uuid.uuid4()),
            operation=data.operation,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/long-term/{memory_id}")
async def delete_long_term_memory(
    memory_id: str,
    request_id: Optional[str] = None,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    try:
        memory = memory_service.update_long_term_memory_record(
            user_id=current_user.id,
            memory_id=memory_id,
            status="deleted",
            operation="delete",
            request_id=request_id or str(uuid.uuid4()),
            db=db,
        )
        return {"message": "Memory deleted", "memory": memory}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/updates")
async def list_memory_update_logs(
    session_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    limit: int = 100,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    query = db.query(MemoryUpdateLogTable).filter(
        MemoryUpdateLogTable.user_id == current_user.id
    )
    if session_id:
        query = query.filter(MemoryUpdateLogTable.session_id == session_id)
    if memory_id:
        query = query.filter(MemoryUpdateLogTable.memory_id == memory_id)
    rows = (
        query.order_by(MemoryUpdateLogTable.created_at.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    return {
        "items": [
            {
                "id": row.id,
                "request_id": row.request_id,
                "session_id": row.session_id,
                "memory_id": row.memory_id,
                "action": row.action,
                "before_value": row.before_value,
                "after_value": row.after_value,
                "reason": row.reason,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }
