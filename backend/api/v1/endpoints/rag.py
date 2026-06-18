# -*- coding: utf-8 -*-
"""
RAG query endpoints.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from api.deps.common import get_current_user
from db.mysql.connection import get_db as get_db_func
from graph.graph import run_agentic_rag
from models.database.user import MessageTable, SessionTable
from models.schemas.rag import RAGQueryRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sse(event: dict) -> str:
    event_type = event.get("event", "unknown")
    data = event.get("data", {})
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _ensure_session(
    db: Session,
    user_id: int,
    session_id: str | None,
    kb_id: int | None,
) -> Tuple[SessionTable, bool]:
    """
    Ensure a session row exists before writing chat messages.
    Returns the session object and whether it was newly created.
    """
    if session_id:
        existing = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
        if existing:
            if existing.user_id != user_id:
                raise HTTPException(status_code=403, detail="Session does not belong to current user")
            if kb_id is not None and existing.kb_id is None:
                existing.kb_id = kb_id
                db.commit()
                db.refresh(existing)
            return existing, False

    resolved_session_id = session_id or str(uuid.uuid4())
    session = SessionTable(
        session_id=resolved_session_id,
        user_id=user_id,
        kb_id=kb_id,
        message_count=0,
        summary=None,
        context={},
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session, True


async def _generate_sse_from_graph(
    query: str,
    user_id: int,
    kb_id: int | None,
    web_enabled: bool,
    session_id: str | None,
    top_k: int | None,
) -> AsyncGenerator[str, None]:
    start_time = time.time()
    logger.info(
        "开始处理查询: query=%s, user_id=%s, kb_id=%s, web_enabled=%s",
        query,
        user_id,
        kb_id,
        web_enabled,
    )

    db_gen = get_db_func()
    db = next(db_gen)
    resolved_session_id: str | None = session_id

    try:
        session, created = _ensure_session(
            db=db,
            user_id=user_id,
            session_id=session_id,
            kb_id=kb_id,
        )
        resolved_session_id = session.session_id
        if created:
            logger.info("自动创建会话成功: session_id=%s", resolved_session_id)

        user_msg = MessageTable(
            session_id=resolved_session_id,
            role="user",
            content=query,
            citations=None,
            token_count=len(query),
        )
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        db.query(SessionTable).filter(SessionTable.session_id == resolved_session_id).update(
            {
                "last_active": func.now(),
                "message_count": SessionTable.message_count + 1,
            }
        )
        db.commit()

        yield _format_sse(
            {
                "event": "run.started",
                "data": {
                    "query": query,
                    "session_id": resolved_session_id,
                },
            }
        )

        final_answer = None
        final_citations = []
        final_confidence = 0
        final_state = None
        emitted_event_count = 0

        async for event in run_agentic_rag(
            query=query,
            user_id=user_id,
            kb_id=kb_id,
            session_id=resolved_session_id,
            web_enabled=web_enabled,
            stream_events=True,
            top_k=top_k,
        ):
            if not isinstance(event, dict):
                continue

            for _, state in event.items():
                if not isinstance(state, dict):
                    continue

                state_events = state.get("events", [])
                for evt in state_events[emitted_event_count:]:
                    if evt.get("event") == "answer.completed":
                        continue
                    yield _format_sse(evt)
                emitted_event_count = max(emitted_event_count, len(state_events))

                if state.get("final_answer"):
                    final_answer = state["final_answer"]
                    final_citations = state.get("citations", [])
                    final_confidence = state.get("confidence", 0)
                    final_state = state

                if state.get("error"):
                    yield _format_sse(
                        {
                            "event": "run.failed",
                            "data": {
                                "error": state["error"],
                                "session_id": resolved_session_id,
                            },
                        }
                    )
                    return

        if final_answer:
            assistant_msg = MessageTable(
                session_id=resolved_session_id,
                role="assistant",
                content=final_answer,
                citations=final_citations,
                token_count=len(final_answer),
            )
            db.add(assistant_msg)
            db.commit()

            db.query(SessionTable).filter(SessionTable.session_id == resolved_session_id).update(
                {
                    "last_active": func.now(),
                    "message_count": SessionTable.message_count + 1,
                }
            )
            db.commit()

            yield _format_sse(
                {
                    "event": "answer.completed",
                    "data": {
                        "answer": final_answer,
                        "citations": final_citations,
                        "confidence": final_confidence,
                        "verification": (final_state or {}).get("verification"),
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "session_id": resolved_session_id,
                    },
                }
            )
            return

        yield _format_sse(
            {
                "event": "run.failed",
                "data": {
                    "error": "No answer generated",
                    "session_id": resolved_session_id,
                },
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("查询处理异常: %s", exc, exc_info=True)
        yield _format_sse(
            {
                "event": "run.failed",
                "data": {
                    "error": str(exc),
                    "session_id": resolved_session_id,
                },
            }
        )
    finally:
        db.close()


@router.post("/test")
async def test_endpoint(request: RAGQueryRequest):
    return {
        "received": {
            "query": request.query,
            "kb_id": request.kb_id,
            "web_enabled": request.web_enabled,
            "stream": request.stream,
            "session_id": request.session_id,
        }
    }


@router.post("/query/stream")
async def query_rag_stream(
    request: RAGQueryRequest,
    current_user=Depends(get_current_user),
):
    try:
        logger.info(
            "收到查询请求: query=%s, kb_id=%s, web_enabled=%s",
            request.query,
            request.kb_id,
            request.web_enabled,
        )
        user_id = current_user.id if current_user else 1

        return StreamingResponse(
            _generate_sse_from_graph(
                query=request.query,
                user_id=user_id,
                kb_id=request.kb_id,
                web_enabled=request.web_enabled,
                session_id=request.session_id,
                top_k=request.top_k,
            ),
            media_type="text/event-stream",
        )
    except Exception as exc:
        logger.error("查询请求处理错误: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query")
async def query_rag(
    request: RAGQueryRequest,
    current_user=Depends(get_current_user),
):
    user_id = current_user.id if current_user else 1
    result = None
    last_error = None

    async for chunk in _generate_sse_from_graph(
        query=request.query,
        user_id=user_id,
        kb_id=request.kb_id,
        web_enabled=request.web_enabled,
        session_id=request.session_id,
        top_k=request.top_k,
    ):
        lines = chunk.strip().split("\n")
        event_type = None
        payload = None

        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    payload = None

        if event_type == "answer.completed" and payload:
            result = payload
        elif event_type == "run.failed" and payload:
            last_error = payload.get("error")

    if result:
        return result

    raise HTTPException(status_code=500, detail=last_error or "查询失败")
