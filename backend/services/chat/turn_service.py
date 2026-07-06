# -*- coding: utf-8 -*-
"""Transactional persistence for a user/assistant turn."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from backend.models.database.user import MessageTable, SessionTable
from backend.services.context.context_assembler import estimate_tokens
from backend.services.memory.memory_service import memory_service


class TurnService:
    def ensure_session(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: Optional[str],
        kb_id: Optional[int],
    ) -> Tuple[SessionTable, bool]:
        if session_id:
            existing = (
                db.query(SessionTable)
                .filter(SessionTable.session_id == session_id)
                .first()
            )
            if existing:
                if existing.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Session does not belong to current user",
                    )
                if kb_id is not None and existing.kb_id is None:
                    existing.kb_id = kb_id
                return existing, False

        session = SessionTable(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            kb_id=kb_id,
            message_count=0,
            title="新对话",
            summary=None,
            context={},
        )
        db.add(session)
        db.flush()
        return session, True

    def begin_turn(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: Optional[str],
        kb_id: Optional[int],
        request_id: str,
        query: str,
    ) -> Dict[str, Any]:
        session, created = self.ensure_session(
            db=db,
            user_id=user_id,
            session_id=session_id,
            kb_id=kb_id,
        )
        existing = (
            db.query(MessageTable)
            .filter(
                MessageTable.request_id == request_id,
                MessageTable.role == "user",
            )
            .first()
        )
        if existing:
            db.commit()
            return {
                "session": session,
                "message": existing,
                "created_session": created,
                "idempotent": True,
            }

        message = MessageTable(
            session_id=session.session_id,
            role="user",
            content=query,
            citations=None,
            token_count=estimate_tokens(query),
            request_id=request_id,
        )
        db.add(message)
        session.last_active = func.now()
        session.message_count = int(session.message_count or 0) + 1
        try:
            db.commit()
            db.refresh(session)
            db.refresh(message)
        except IntegrityError:
            db.rollback()
            message = (
                db.query(MessageTable)
                .filter(
                    MessageTable.request_id == request_id,
                    MessageTable.role == "user",
                )
                .first()
            )
            if not message:
                raise
            session = (
                db.query(SessionTable)
                .filter(SessionTable.session_id == message.session_id)
                .first()
            )
            return {
                "session": session,
                "message": message,
                "created_session": False,
                "idempotent": True,
            }
        return {
            "session": session,
            "message": message,
            "created_session": created,
            "idempotent": False,
        }

    def finalize_turn(
        self,
        *,
        db: Session,
        user_id: int,
        session_id: str,
        request_id: str,
        answer: str,
        citations: list,
        memory_update_plan: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session = (
            db.query(SessionTable)
            .filter(
                SessionTable.session_id == session_id,
                SessionTable.user_id == user_id,
            )
            .first()
        )
        if not session:
            raise ValueError("Session not found")

        existing = (
            db.query(MessageTable)
            .filter(
                MessageTable.request_id == request_id,
                MessageTable.role == "assistant",
            )
            .first()
        )
        if existing:
            return {
                "message_id": existing.id,
                "memory": {"applied": False, "idempotent": True, "actions": []},
                "idempotent": True,
            }

        assistant_message = MessageTable(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations,
            token_count=estimate_tokens(answer),
            request_id=request_id,
        )
        db.add(assistant_message)
        db.flush()
        session.last_active = func.now()
        session.message_count = int(session.message_count or 0) + 1

        memory_result = {"applied": False, "actions": []}
        if memory_update_plan:
            memory_result = memory_service.apply_memory_update_plan(
                memory_update_plan,
                db=db,
                assistant_message_id=assistant_message.id,
            )
        db.commit()
        db.refresh(assistant_message)
        return {
            "message_id": assistant_message.id,
            "memory": memory_result,
            "idempotent": False,
        }


turn_service = TurnService()
