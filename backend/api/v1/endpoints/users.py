# -*- coding: utf-8 -*-
"""
用户管理 API 路由
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List, Optional
from datetime import datetime
import json

from backend.db.mysql.connection import get_db
from backend.models.schemas.user import (
    User,
    UserCreate,
    UserProfile,
    UserProfileUpdate,
    SessionContext,
    SessionRenameRequest,
)
from backend.models.database.user import (
    MessageTable,
    SessionMemoryTable,
    SessionTable,
    UserProfileTable,
    UserTable,
    RAGRunTable,
    RAGSpanTable,
)
from backend.api.deps.common import get_required_user
from backend.services.chat.branch import branch_service
from backend.models.schemas.memory import TraceResponse

router = APIRouter()


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user = Depends(get_required_user),
):
    """
    获取当前用户信息
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"获取当前用户: id={current_user.id}, username={current_user.username}")

    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )


@router.get("/debug")
async def debug_info(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    调试端点 - 返回当前用户和会话信息
    """
    import logging
    logger = logging.getLogger(__name__)

    # 获取所有会话
    all_sessions = db.query(SessionTable).filter(
        SessionTable.user_id == current_user.id
    ).all()

    sessions_info = [
        {
            "session_id": s.session_id,
            "title": s.title or s.summary or "新对话",
            "summary": s.summary,
            "category": s.category,
            "message_count": s.message_count,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in all_sessions
    ]

    return {
        "current_user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
        },
        "sessions_count": len(all_sessions),
        "sessions": sessions_info,
    }


@router.get("/me/profile", response_model=UserProfile)
async def get_user_profile(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取当前用户画像
    """
    profile = db.query(UserProfileTable).filter(
        UserProfileTable.user_id == current_user.id
    ).first()

    if not profile:
        # 创建默认画像
        profile = UserProfileTable(
            user_id=current_user.id,
            preferred_language="zh-CN",
            interests=[],
            interaction_style="concise",
            frequently_asked_topics=[],
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return UserProfile(
        user_id=profile.user_id,
        preferred_language=profile.preferred_language,
        interests=profile.interests or [],
        interaction_style=profile.interaction_style,
        frequently_asked_topics=profile.frequently_asked_topics or [],
    )


@router.put("/me/profile", response_model=UserProfile)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    更新当前用户画像
    """
    profile = db.query(UserProfileTable).filter(
        UserProfileTable.user_id == current_user.id
    ).first()

    if not profile:
        profile = UserProfileTable(user_id=current_user.id)
        db.add(profile)

    profile.preferred_language = profile_update.preferred_language
    profile.interests = profile_update.interests
    profile.interaction_style = profile_update.interaction_style
    profile.frequently_asked_topics = profile_update.frequently_asked_topics

    db.commit()
    db.refresh(profile)

    return UserProfile(
        user_id=profile.user_id,
        preferred_language=profile.preferred_language,
        interests=profile.interests or [],
        interaction_style=profile.interaction_style,
        frequently_asked_topics=profile.frequently_asked_topics or [],
    )


# ========== 会话管理 ==========

@router.get("/sessions", response_model=List[SessionContext])
async def list_sessions(
    include_archived: bool = Query(False),
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取用户的所有会话
    """
    session_query = db.query(SessionTable).filter(
        SessionTable.user_id == current_user.id,
        SessionTable.message_count > 0,
    )
    if not include_archived:
        session_query = session_query.filter(SessionTable.archived.is_(False))
    sessions = session_query.order_by(SessionTable.last_active.desc()).all()

    return [
        SessionContext(
            session_id=s.session_id,
            user_id=s.user_id,
            kb_id=s.kb_id,
            created_at=s.created_at,
            last_active=s.last_active,
            message_count=s.message_count,
            title=s.title or s.summary or "新对话",
            summary=s.summary,
            category=s.category,
            archived=bool(s.archived),
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=SessionContext)
async def create_session(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    创建新会话
    """
    import uuid
    session_id = str(uuid.uuid4())

    session = SessionTable(
        session_id=session_id,
        user_id=current_user.id,
        message_count=0,
        title="新对话",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionContext(
        session_id=session.session_id,
        user_id=session.user_id,
        kb_id=session.kb_id,
        created_at=session.created_at,
        last_active=session.last_active,
        message_count=session.message_count,
        title=session.title or session.summary or "新对话",
        summary=session.summary,
        category=session.category,
        archived=bool(session.archived),
    )


@router.get("/session-categories")
async def get_session_categories_alias(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    Static alias for session categories to avoid being swallowed by /sessions/{session_id}.
    """
    categories = db.query(SessionTable.category).filter(
        SessionTable.user_id == current_user.id,
        SessionTable.category.isnot(None),
    ).distinct().all()

    category_counts = {}
    for (cat,) in categories:
        count = db.query(SessionTable).filter(
            SessionTable.user_id == current_user.id,
            SessionTable.category == cat,
        ).count()
        category_counts[cat] = count

    return {
        "categories": sorted(category_counts.keys()),
        "counts": category_counts,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    删除指定会话及其所有关联数据（消息、分支）
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"删除会话: session_id={session_id}, user_id={current_user.id}")

    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        logger.warning(f"会话未找到: session_id={session_id}, user_id={current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )

    try:
        # 先删除关联的对话分支
        from backend.models.database.user import ConversationBranchTable
        deleted_branches = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.session_id == session_id
        ).delete()
        logger.info(f"删除对话分支: {deleted_branches} 条")

        # 删除关联的消息
        deleted_messages = db.query(MessageTable).filter(
            MessageTable.session_id == session_id
        ).delete()
        logger.info(f"删除消息: {deleted_messages} 条")

        db.query(SessionMemoryTable).filter(
            SessionMemoryTable.session_id == session_id
        ).delete()

        # 最后删除会话本身
        db.delete(session)
        db.commit()
        logger.info(f"会话删除成功: session_id={session_id}")

        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"会话删除失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
    )


@router.patch("/sessions/{session_id}/title", response_model=SessionContext)
async def rename_session(
    session_id: str,
    payload: SessionRenameRequest,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    Rename a session.
    """
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session.title = payload.title.strip()
    session.last_active = func.now()
    db.commit()
    db.refresh(session)

    return SessionContext(
        session_id=session.session_id,
        user_id=session.user_id,
        kb_id=session.kb_id,
        created_at=session.created_at,
        last_active=session.last_active,
        message_count=session.message_count,
        title=session.title or "新对话",
        summary=session.summary,
        category=session.category,
        archived=bool(session.archived),
    )


@router.get("/sessions/{session_id}", response_model=SessionContext)
async def get_session(
    session_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取指定会话详情
    """
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return SessionContext(
        session_id=session.session_id,
        user_id=session.user_id,
        kb_id=session.kb_id,
        created_at=session.created_at,
        last_active=session.last_active,
        message_count=session.message_count,
        title=session.title or session.summary or "新对话",
        summary=session.summary,
        category=session.category,
        archived=bool(session.archived),
    )


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    category: Optional[str] = Query(None, description="会话分类（__null__ 表示清除分类）"),
    archived: Optional[bool] = Query(None),
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    更新会话信息（如分类）
    """
    # 添加调试日志
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"更新会话分类: session_id={session_id}, category={category}, user_id={current_user.id}")

    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        logger.warning(f"会话未找到: session_id={session_id}, user_id={current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )

    if category is not None:
        # 特殊值 __null__ 表示清除分类
        if category == "__null__":
            session.category = None
            logger.info(f"清除会话分类: session_id={session_id}")
        else:
            session.category = category
            logger.info(f"设置会话分类: session_id={session_id}, category={category}")
    if archived is not None:
        session.archived = archived

    try:
        db.commit()
        db.refresh(session)
        logger.info(f"会话更新成功: session_id={session_id}")
    except Exception as e:
        logger.error(f"会话更新失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}"
        )

    return SessionContext(
        session_id=session.session_id,
        user_id=session.user_id,
        kb_id=session.kb_id,
        created_at=session.created_at,
        last_active=session.last_active,
        message_count=session.message_count,
        title=session.title or session.summary or "新对话",
        summary=session.summary,
        category=session.category,
        archived=bool(session.archived),
    )


@router.get("/sessions/categories")
async def get_session_categories(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取用户的所有会话分类
    """
    # 获取所有不重复的分类
    categories = db.query(SessionTable.category).filter(
        SessionTable.user_id == current_user.id,
        SessionTable.category.isnot(None),
    ).distinct().all()

    # 统计每个分类的会话数量
    category_counts = {}
    for (cat,) in categories:
        count = db.query(SessionTable).filter(
            SessionTable.user_id == current_user.id,
            SessionTable.category == cat,
        ).count()
        category_counts[cat] = count

    return {
        "categories": sorted(category_counts.keys()),
        "counts": category_counts,
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取会话的所有消息
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # 获取消息列表
    messages = db.query(MessageTable).filter(
        MessageTable.session_id == session_id
    ).order_by(MessageTable.created_at).all()

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations,
                "token_count": msg.token_count,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ]
    }


@router.get("/sessions/{session_id}/trace", response_model=TraceResponse)
async def get_session_trace(
    session_id: str,
    request_id: Optional[str] = None,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """Return the persisted, user-safe run trace available for a session."""
    session = db.query(SessionTable).filter(SessionTable.session_id == session_id, SessionTable.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    run_query = db.query(RAGRunTable).filter(RAGRunTable.session_id == session_id, RAGRunTable.user_id == current_user.id)
    if request_id:
        run_query = run_query.filter(RAGRunTable.request_id == request_id)
    runs = run_query.order_by(RAGRunTable.created_at.asc()).all()
    payload = []
    for run in runs:
        spans = db.query(RAGSpanTable).filter(RAGSpanTable.request_id == run.request_id).order_by(RAGSpanTable.started_at.asc()).all()
        payload.append({"id": run.id, "request_id": run.request_id, "user_id": run.user_id, "session_id": run.session_id, "kb_id": run.kb_id, "route_type": run.route_type, "final_status": run.final_status, "answer_mode": run.answer_mode, "reflection_count": run.reflection_count, "total_latency_ms": run.total_latency_ms, "input_tokens": run.input_tokens, "output_tokens": run.output_tokens, "selected_evidence_ids": run.selected_evidence_ids or [], "selected_memory_ids": run.selected_memory_ids or [], "error": run.error, "created_at": run.created_at.isoformat() if run.created_at else None, "completed_at": run.completed_at.isoformat() if run.completed_at else None, "spans": [{"id": span.id, "request_id": span.request_id, "span_name": span.span_name, "status": span.status, "started_at": span.started_at.isoformat() if span.started_at else None, "ended_at": span.ended_at.isoformat() if span.ended_at else None, "latency_ms": span.latency_ms, "model_name": span.model_name, "input_tokens": span.input_tokens, "output_tokens": span.output_tokens, "metadata": span.metadata_json or {}, "error": span.error} for span in spans]})
    return {"session_id": session_id, "runs": payload}


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    format: str = "json",  # json, markdown, txt
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    导出会话记录
    支持格式: json, markdown, txt
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # 获取会话消息
    messages = db.query(MessageTable).filter(
        MessageTable.session_id == session_id
    ).order_by(MessageTable.created_at).all()

    # 构建导出数据
    export_data = {
        "session_id": session.session_id,
        "title": session.title or session.summary or "未命名对话",
        "created_at": session.created_at.isoformat(),
        "last_active": session.last_active.isoformat(),
        "message_count": len(messages),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ]
    }

    if format == "json":
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_{datetime.now().strftime("%Y%m%d")}.json"'
            }
        )

    elif format == "markdown":
        # 生成 Markdown 格式
        md_lines = [
            f"# {export_data['title']}\n",
            f"**会话 ID**: `{session_id}`  \n",
            f"**创建时间**: {export_data['created_at']}  \n",
            f"**消息数量**: {export_data['message_count']}\n",
            "---\n",
        ]

        for msg in export_data["messages"]:
            role_name = "用户" if msg["role"] == "user" else "助手"
            md_lines.append(f"## {role_name}\n")
            md_lines.append(f"{msg['content']}\n")
            md_lines.append(f"*{msg['created_at']}*\n")
            md_lines.append("---\n")

        md_content = "\n".join(md_lines)

        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_{datetime.now().strftime("%Y%m%d")}.md"'
            }
        )

    elif format == "txt":
        # 生成纯文本格式
        txt_lines = [
            f"会话: {export_data['title']}",
            f"会话 ID: {session_id}",
            f"创建时间: {export_data['created_at']}",
            f"消息数量: {export_data['message_count']}",
            "=" * 50,
            "",
        ]

        for msg in export_data["messages"]:
            role_name = "用户" if msg["role"] == "user" else "助手"
            txt_lines.append(f"[{role_name}] - {msg['created_at']}")
            txt_lines.append(msg["content"])
            txt_lines.append("")
            txt_lines.append("-" * 50)
            txt_lines.append("")

        txt_content = "\n".join(txt_lines)

        return Response(
            content=txt_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_{datetime.now().strftime("%Y%m%d")}.txt"'
            }
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}. Supported formats: json, markdown, txt"
        )


# ========== 对话分支管理 ==========

@router.get("/sessions/{session_id}/branches")
async def list_branches(
    session_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取会话的所有对话分支
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    branches = branch_service.list_branches(session_id, db)
    return {"branches": branches}


@router.post("/sessions/{session_id}/branches")
async def create_branch(
    session_id: str,
    from_message_id: int = Query(..., description="从哪条消息开始创建分支"),
    branch_name: Optional[str] = Query(None, description="分支名称"),
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    创建新的对话分支
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    try:
        branch = branch_service.fork_from_message(
            session_id=session_id,
            from_message_id=from_message_id,
            branch_name=branch_name,
            db=db,
        )

        return {
            "id": branch.id,
            "branch_name": branch.branch_name,
            "parent_branch_id": branch.parent_branch_id,
            "parent_message_id": branch.parent_message_id,
            "is_active": branch.is_active,
            "created_at": branch.created_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/sessions/{session_id}/branches/{branch_id}/switch")
async def switch_branch(
    session_id: str,
    branch_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    切换到指定对话分支
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # 验证分支存在
    from backend.models.database.user import ConversationBranchTable
    branch = db.query(ConversationBranchTable).filter(
        ConversationBranchTable.id == branch_id,
        ConversationBranchTable.session_id == session_id,
    ).first()

    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found"
        )

    try:
        branch_service.switch_branch(branch_id, db)
        return {"message": "Branch switched successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/sessions/{session_id}/branches/{branch_id}")
async def delete_branch(
    session_id: str,
    branch_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    删除对话分支
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # 验证分支存在
    from backend.models.database.user import ConversationBranchTable
    branch = db.query(ConversationBranchTable).filter(
        ConversationBranchTable.id == branch_id,
        ConversationBranchTable.session_id == session_id,
    ).first()

    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found"
        )

    try:
        branch_service.delete_branch(branch_id, db)
        return {"message": "Branch deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/branches/{branch_id}/messages")
async def get_branch_messages(
    session_id: str,
    branch_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取对话分支的所有消息
    """
    # 验证会话存在且属于当前用户
    session = db.query(SessionTable).filter(
        SessionTable.session_id == session_id,
        SessionTable.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # 验证分支存在
    from backend.models.database.user import ConversationBranchTable
    branch = db.query(ConversationBranchTable).filter(
        ConversationBranchTable.id == branch_id,
        ConversationBranchTable.session_id == session_id,
    ).first()

    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found"
        )

    messages = branch_service.get_branch_messages(branch_id, db)
    return {"messages": messages}
