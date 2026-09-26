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

from backend.db.mysql.connection import get_db as get_db_func
from backend.agent.runtime import get_runtime_mode, run_rag_runtime
from backend.core.config import settings
from backend.models.schemas.rag import RAGQueryRequest
from backend.models.database.user import AnswerFeedbackTable, MessageTable, SessionTable, RAGRunTable
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


def _safe_trace_error(value: object) -> str:
    """Return a stable user-safe error summary; full details stay in server logs."""

    text = str(value or "").lower()
    if "deadline" in text or "timeout" in text:
        return "runtime_timeout"
    if "agent_decision_failed" in text:
        return "agent_decision_failed"
    if "semantic verifier" in text:
        return "semantic_verification_failed"
    if "checkpoint" in text:
        return "checkpoint_runtime_interrupted"
    return "react_runtime_interrupted"


async def _generate_sse_from_graph(
    query: str,
    user_id: int,
    kb_id: int | None,
    web_enabled: bool,
    session_id: str | None,
    top_k: int | None,
    branch_id: int | None = None,
    request_id: str | None = None,
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
    resolved_branch_id: int | None = branch_id
    request_id = request_id or str(uuid.uuid4())
    runtime_mode = get_runtime_mode()
    trace_attempt_no = 1

    try:
        turn = turn_service.begin_turn(
            db=db,
            user_id=user_id,
            session_id=session_id,
            kb_id=kb_id,
            request_id=request_id,
            query=query,
            branch_id=branch_id,
        )
        session = turn["session"]
        created = turn["created_session"]
        resolved_session_id = session.session_id
        resolved_branch_id = turn.get("branch_id")
        trace_service.start_run(db, request_id=request_id, user_id=user_id, session_id=resolved_session_id, kb_id=kb_id, runtime_mode=runtime_mode, budget_profile=settings.AGENT_CONTEXT_PROFILE)
        attempt = trace_service.start_attempt(db, request_id=request_id)
        trace_attempt_no = attempt.attempt_no
        trace_service.record_event(
            db,
            request_id=request_id,
            attempt_no=trace_attempt_no,
            event_name="run.started",
            status="running",
            metadata={"resumed": attempt.resumed},
        )
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
                    "attempt_no": trace_attempt_no,
                    "resumed": attempt.resumed,
                },
            }
        )

        final_answer = None
        final_citations = []
        final_confidence = 0
        final_state = {}
        memory_update_plan = None
        emitted_event_count = 0

        async for event in run_rag_runtime(
            query=query,
            user_id=user_id,
            request_id=request_id,
            kb_id=kb_id,
            session_id=resolved_session_id,
            web_enabled=web_enabled,
            stream_events=True,
            top_k=top_k,
            branch_id=resolved_branch_id,
            runtime_mode=runtime_mode,
            trace_attempt_no=trace_attempt_no,
        ):
            if not isinstance(event, dict):
                continue

            for _, state in event.items():
                if not isinstance(state, dict):
                    continue
                final_state.update(state)

                state_events = state.get("events", [])
                emitted_event_count = max(
                    emitted_event_count,
                    int(state.get("trace_event_offset") or 0),
                )
                new_events = state_events[emitted_event_count:]
                if new_events:
                    event_db = next(get_db_func())
                    try:
                        for evt in new_events:
                            event_name = str(evt.get("event") or "unknown")
                            trace_service.record_event(
                                event_db,
                                request_id=request_id,
                                attempt_no=trace_attempt_no,
                                event_name=event_name,
                                node_name={
                                    "memory.loaded": "hydrate_context",
                                    "agent.decision": "decide",
                                    "tool.requested": "act",
                                    "tool.completed": "act",
                                    "observation.created": "observe",
                                    "evidence.updated": "observe",
                                    "answer.verified": "verify",
                                    "memory.update.planned": "memory",
                                }.get(event_name),
                                metadata=evt.get("data") if isinstance(evt.get("data"), dict) else {},
                            )
                        event_db.commit()
                    finally:
                        event_db.close()
                for evt in new_events:
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

                runtime_error = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
                error_value = state.get("error") or runtime_error.get("error")
                if error_value:
                    safe_error = _safe_trace_error(error_value)
                    resumable = bool(state.get("resumable") or runtime_error.get("resumable"))
                    stop_reason = state.get("stop_reason") or runtime_error.get("stop_reason")
                    error_db = next(get_db_func())
                    try:
                        trace_service.finish_run(error_db, request_id=request_id, final_status="interrupted" if resumable else "failed", total_latency_ms=int((time.time() - start_time) * 1000), route_type=final_state.get("route_type"), reflection_count=int(final_state.get("reflection_count") or 0), stop_reason=stop_reason, error=safe_error)
                        trace_service.finish_attempt(
                            error_db,
                            request_id=request_id,
                            attempt_no=trace_attempt_no,
                            status="interrupted" if resumable else "failed",
                            stop_reason=stop_reason,
                            error=safe_error,
                        )
                        trace_service.record_event(
                            error_db,
                            request_id=request_id,
                            attempt_no=trace_attempt_no,
                            event_name="run.failed",
                            status="interrupted" if resumable else "failed",
                            metadata={"stop_reason": stop_reason, "resumable": resumable},
                        )
                        error_db.commit()
                    finally:
                        error_db.close()
                    yield _format_sse(
                        {
                            "event": "run.failed",
                            "data": {
                                "error": safe_error,
                                "session_id": resolved_session_id,
                                "request_id": request_id,
                                "resumable": resumable,
                                "stop_reason": stop_reason,
                                "attempt_no": trace_attempt_no,
                            },
                        }
                    )
                    return

        if final_answer:
            db = next(get_db_func())
            try:
                trace_service.finish_run(db, request_id=request_id, final_status="completed", total_latency_ms=int((time.time() - start_time) * 1000), route_type="react", answer_mode=(final_state.get("verification") or {}).get("recommended_action") if isinstance(final_state.get("verification"), dict) else None, reflection_count=0, input_tokens=int(final_state.get("input_tokens") or 0), output_tokens=int(final_state.get("output_tokens") or 0), selected_evidence_ids=[str(item.get("evidence_id")) for item in final_state.get("selected_evidence", []) if item.get("evidence_id")], selected_memory_ids=(final_state.get("memory_context") or {}).get("selected_memory_ids") or [], runtime_mode=runtime_mode, iteration_count=int(final_state.get("iteration") or 0), stop_reason=final_state.get("stop_reason"), tool_call_count=len(final_state.get("tool_calls") or []), budget_profile=(final_state.get("budgets") or {}).get("profile") or settings.AGENT_CONTEXT_PROFILE, shadow_metrics={})
                trace_service.finish_attempt(
                    db,
                    request_id=request_id,
                    attempt_no=trace_attempt_no,
                    status="completed",
                    stop_reason=final_state.get("stop_reason"),
                )
                trace_service.record_event(
                    db,
                    request_id=request_id,
                    attempt_no=trace_attempt_no,
                    event_name="run.completed",
                    status="completed",
                    metadata={"stop_reason": final_state.get("stop_reason")},
                )
                persisted = turn_service.finalize_turn(
                    db=db,
                    user_id=user_id,
                    session_id=resolved_session_id,
                    request_id=request_id,
                    answer=final_answer,
                    citations=final_citations,
                    memory_update_plan=memory_update_plan,
                    branch_id=resolved_branch_id,
                )
            finally:
                db.close()
                db = None
            yield _format_sse({
                "event": "memory.updated",
                "data": {
                    "request_id": request_id,
                    "session_id": resolved_session_id,
                    "attempt_no": trace_attempt_no,
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
            trace_service.finish_attempt(
                error_db,
                request_id=request_id,
                attempt_no=trace_attempt_no,
                status="failed",
                stop_reason="no_answer",
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
                    "attempt_no": trace_attempt_no,
                },
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("查询处理异常: %s", exc, exc_info=True)
        safe_error = _safe_trace_error(exc)
        try:
            error_db = next(get_db_func())
            trace_service.finish_run(error_db, request_id=request_id, final_status="failed", total_latency_ms=int((time.time() - start_time) * 1000), error=safe_error)
            trace_service.finish_attempt(error_db, request_id=request_id, attempt_no=trace_attempt_no, status="failed", stop_reason="api_error", error=safe_error)
            error_db.commit()
            error_db.close()
        except Exception:
            logger.exception("Failed to persist trace failure")
        yield _format_sse(
            {
                "event": "run.failed",
                "data": {
                    "error": safe_error,
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
    current_user=Depends(get_required_user),
):
    try:
        logger.info(
            "收到查询请求: query=%s, kb_id=%s, web_enabled=%s",
            request.query,
            request.kb_id,
            request.web_enabled,
        )
        user_id = current_user.id

        return StreamingResponse(
            _generate_sse_from_graph(
                query=request.query,
                user_id=user_id,
                kb_id=request.kb_id,
                web_enabled=request.web_enabled,
                session_id=request.session_id,
                top_k=request.top_k,
                branch_id=request.branch_id,
                request_id=request.request_id,
            ),
            media_type="text/event-stream",
        )
    except Exception as exc:
        logger.error("查询请求处理错误: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query")
async def query_rag(
    request: RAGQueryRequest,
    current_user=Depends(get_required_user),
):
    user_id = current_user.id
    result = None
    last_error = None

    async for chunk in _generate_sse_from_graph(
        query=request.query,
        user_id=user_id,
        kb_id=request.kb_id,
        web_enabled=request.web_enabled,
        session_id=request.session_id,
        top_k=request.top_k,
        branch_id=request.branch_id,
        request_id=request.request_id,
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
