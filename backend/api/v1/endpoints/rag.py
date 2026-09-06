# -*- coding: utf-8 -*-
"""
RAG query endpoints.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps.common import get_current_user
from backend.db.mysql.connection import get_db as get_db_func
from backend.graph.graph import run_agentic_rag
from backend.models.schemas.rag import RAGQueryRequest
from backend.models.database.user import AnswerFeedbackTable, MessageTable, SessionTable, RAGRunTable, RAGSpanTable
from backend.api.deps.common import get_required_user
from backend.services.chat.turn_service import turn_service
from backend.services.trace_service import trace_service
from backend.models.schemas.memory import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sse(event: dict) -> str:
    event_type = event.get("event", "unknown")
    data = event.get("data", {})
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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

    db: Session | None = next(get_db_func())
    resolved_session_id: str | None = session_id
    request_id = str(uuid.uuid4())

    try:
        turn = turn_service.begin_turn(
            db=db,
            user_id=user_id,
            session_id=session_id,
            kb_id=kb_id,
            request_id=request_id,
            query=query,
        )
        session = turn["session"]
        created = turn["created_session"]
        resolved_session_id = session.session_id
        trace_service.start_run(db, request_id=request_id, user_id=user_id, session_id=resolved_session_id, kb_id=kb_id)
        db.commit()
        if created:
            logger.info("自动创建会话成功: session_id=%s", resolved_session_id)
        db.close()
        db = None

        yield _format_sse(
            {
                "event": "run.started",
                "data": {
                    "query": query,
                    "session_id": resolved_session_id,
                    "request_id": request_id,
                },
            }
        )

        final_answer = None
        final_citations = []
        final_confidence = 0
        final_state = {}
        memory_update_plan = None
        emitted_event_count = 0

        async for event in run_agentic_rag(
            query=query,
            user_id=user_id,
            request_id=request_id,
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
                final_state.update(state)

                state_events = state.get("events", [])
                for evt in state_events[emitted_event_count:]:
                    if evt.get("event") in {"answer.completed", "memory.update.planned"}:
                        continue
                    yield _format_sse(evt)
                emitted_event_count = max(emitted_event_count, len(state_events))

                if state.get("final_answer"):
                    final_answer = state["final_answer"]
                    final_citations = state.get("citations", [])
                    final_confidence = state.get("confidence", 0)
                if state.get("memory_update_plan"):
                    memory_update_plan = state["memory_update_plan"]

                if state.get("error"):
                    error_db = next(get_db_func())
                    try:
                        trace_service.finish_run(error_db, request_id=request_id, final_status="failed", total_latency_ms=int((time.time() - start_time) * 1000), route_type=final_state.get("route_type"), reflection_count=int(final_state.get("reflection_count") or 0), error=str(state["error"]))
                        error_db.commit()
                    finally:
                        error_db.close()
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
            db = next(get_db_func())
            try:
                started_span_names = set()
                span_map = {
                    "memory.loaded": "memory.load", "query.rewritten": "query.rewrite", "query.decomposed": "query.decompose",
                    "route.planned": "route.plan", "retrieval.dense": "retrieval.dense", "retrieval.sparse": "retrieval.sparse",
                    "rrf.completed": "rrf", "rerank.completed": "rerank", "evidence.graded": "evidence.grade",
                    "reflection.completed": "reflection", "answer.delta": "generation", "answer.completed": "verification",
                    "memory.updated": "memory.update",
                }
                for graph_event in final_state.get("events", []):
                    event_name = graph_event.get("event")
                    span_name = span_map.get(event_name)
                    if span_name and span_name not in started_span_names:
                        trace_service.record_span(db, request_id=request_id, span_name=span_name, metadata={"event": event_name, "count": graph_event.get("data", {}).get("count") if isinstance(graph_event.get("data"), dict) else None})
                        started_span_names.add(span_name)
                # Persist deterministic node timings even when a node does not
                # emit a public event. Event-derived spans are supplemented,
                # never duplicated, by the graph state's measured timings.
                for span_name, latency in (final_state.get("latency_breakdown_ms") or {}).items():
                    normalized = str(span_name)
                    if normalized not in started_span_names:
                        trace_service.record_span(
                            db,
                            request_id=request_id,
                            span_name=normalized,
                            latency_ms=int(latency or 0),
                            metadata={"source": "graph.latency_breakdown"},
                        )
                        started_span_names.add(normalized)
                trace_service.finish_run(db, request_id=request_id, final_status="completed", total_latency_ms=int((time.time() - start_time) * 1000), route_type=final_state.get("route_type"), answer_mode=(final_state.get("evidence_grade") or {}).get("answer_mode") if isinstance(final_state.get("evidence_grade"), dict) else None, reflection_count=int(final_state.get("reflection_count") or 0), input_tokens=int((final_state.get("context_token_usage") or {}).get("used") or 0), output_tokens=int(final_state.get("output_tokens") or 0), selected_evidence_ids=[str(item.get("evidence_id")) for item in final_state.get("selected_evidence", []) if item.get("evidence_id")], selected_memory_ids=(final_state.get("memory_context") or {}).get("selected_memory_ids") or [])
                persisted = turn_service.finalize_turn(
                    db=db,
                    user_id=user_id,
                    session_id=resolved_session_id,
                    request_id=request_id,
                    answer=final_answer,
                    citations=final_citations,
                    memory_update_plan=memory_update_plan,
                )
            finally:
                db.close()
                db = None
            yield _format_sse({
                "event": "memory.updated",
                "data": {
                    "request_id": request_id,
                    "session_id": resolved_session_id,
                    **(persisted.get("memory") or {}),
                },
            })

            yield _format_sse(
                {
                    "event": "answer.completed",
                    "data": {
                        "answer": final_answer,
                        "citations": final_citations,
                        "confidence": final_confidence,
                        "verification": final_state.get("verification"),
                        "context_token_usage": final_state.get("context_token_usage") or {
                            "memory": (
                                (final_state.get("memory_context") or {}).get(
                                    "context_token_usage"
                                )
                                or {}
                            ),
                            "generation": [],
                        },
                        "request_id": request_id,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "session_id": resolved_session_id,
                    },
                }
            )
            return

        error_db = next(get_db_func())
        try:
            trace_service.finish_run(
                error_db,
                request_id=request_id,
                final_status="failed",
                total_latency_ms=int((time.time() - start_time) * 1000),
                route_type=final_state.get("route_type"),
                reflection_count=int(final_state.get("reflection_count") or 0),
                error="No answer generated",
            )
            error_db.commit()
        finally:
            error_db.close()
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
        try:
            error_db = next(get_db_func())
            trace_service.finish_run(error_db, request_id=request_id, final_status="failed", total_latency_ms=int((time.time() - start_time) * 1000), error=str(exc))
            error_db.commit(); error_db.close()
        except Exception:
            logger.exception("Failed to persist trace failure")
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
        if db is not None:
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


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    data: FeedbackRequest,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db_func),
):
    request_id = data.request_id
    rating = data.rating
    run = db.query(RAGRunTable).filter(RAGRunTable.request_id == request_id, RAGRunTable.user_id == current_user.id).first()
    message = db.query(MessageTable).filter(MessageTable.request_id == request_id, MessageTable.role == "assistant").first()
    if not run and not message:
        raise HTTPException(status_code=404, detail="Answer run not found")
    session_id = run.session_id if run else message.session_id
    session = db.query(SessionTable).filter(SessionTable.session_id == session_id, SessionTable.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    row = db.query(AnswerFeedbackTable).filter(AnswerFeedbackTable.request_id == request_id, AnswerFeedbackTable.user_id == current_user.id).first()
    if row:
        row.rating = rating
        row.comment = data.comment
    else:
        row = AnswerFeedbackTable(request_id=request_id, session_id=session_id, user_id=current_user.id, rating=rating, comment=data.comment)
        db.add(row)
    db.commit(); db.refresh(row)
    return {"id": row.id, "request_id": request_id, "rating": row.rating, "comment": row.comment}
