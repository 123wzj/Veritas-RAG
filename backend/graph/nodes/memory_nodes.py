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
        memory_context = memory_service.get_short_term_memory(
            user_id=user_id,
            session_id=session_id,
            query=query,
            db=db,
            current_request_id=state.get("request_id"),
        )

        events.append({
            "event": "memory.loaded",
            "data": {
                "context_keys": list(memory_context.keys()),
                "has_prompt_context": bool(memory_context.get("prompt_context")),
                "selected_memory_ids": memory_context.get("selected_memory_ids") or [],
                "token_usage": memory_context.get("context_token_usage") or {},
                "stage": "short_term",
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


async def load_generation_memory(state: RAGState) -> Dict[str, Any]:
    """Select long-term memory after retrieval/rerank and before generation."""
    current = state.get("memory_context") or {}
    query_parts = [
        state.get("query_rewritten") or state.get("query", ""),
        *[
            plan.get("query_rewritten") or plan.get("sub_question") or ""
            for plan in (state.get("sub_query_plans") or [])
        ],
    ]
    query = " | ".join(dict.fromkeys(
        part.strip()
        for part in query_parts
        if isinstance(part, str) and part.strip()
    ))
    if (
        current.get("long_term_selection_complete")
        and current.get("long_term_selection_query") == query
    ):
        return {"memory_context": current}

    user_id = state.get("user_id", 0)
    session_id = state.get("session_id")
    db: Optional[Session] = None
    try:
        db = next(get_db())
        memory_context = await memory_service.aget_session_memory(
            user_id=user_id,
            session_id=session_id,
            query=query,
            db=db,
            kb_id=state.get("kb_id"),
            current_request_id=state.get("request_id"),
            working_memory=current.get("working_memory") or {},
        )
        if current.get("working_memory_generated_at"):
            memory_context["working_memory"] = current.get("working_memory") or {}
            memory_context["working_memory_generated_at"] = current[
                "working_memory_generated_at"
            ]
        return {
            "memory_context": memory_context,
            "events": state.get("events", []) + [{
                "event": "memory.long_term.selected",
                "data": {
                    "stage": "generation",
                    "selected_memory_ids": (
                        memory_context.get("selected_memory_ids") or []
                    ),
                    "token_usage": memory_context.get("context_token_usage") or {},
                },
            }],
        }
    except Exception as exc:
        return {
            "memory_context": current,
            "events": state.get("events", []) + [{
                "event": "memory.long_term.selection_failed",
                "data": {"error": str(exc)},
            }],
        }
    finally:
        if db is not None:
            db.close()


async def write_memory(state: RAGState) -> Dict[str, Any]:
    """
    写入记忆节点

    生成短期/长期记忆更新计划。数据库提交由 API 最终持久化事务完成。
    """
    user_id = state.get("user_id", 0)
    session_id = state.get("session_id")
    query = state.get("query", "")
    final_answer = state.get("final_answer", "")
    verification = state.get("verification") or {}

    db: Optional[Session] = None
    try:
        db = next(get_db())
        memory_update_plan = None
        memory_update_allowed = (
            not verification
            or (
                verification.get("grounded", True)
                and verification.get("useful", True)
            )
        )
        if session_id and final_answer and memory_update_allowed:
            memory_update_plan = await memory_service.build_memory_update_plan(
                user_id=user_id,
                session_id=session_id,
                query=query,
                answer=final_answer,
                request_id=state.get("request_id", ""),
                kb_id=state.get("kb_id"),
                db=db,
                verification=verification,
                confidence=state.get("confidence"),
                working_memory_draft=(
                    (state.get("memory_context") or {}).get("working_memory")
                ),
            )

        return {
            "memory_update_plan": memory_update_plan,
            "events": state.get("events", []) + [{
                "event": (
                    "memory.update.planned"
                    if memory_update_allowed
                    else "memory.update.skipped"
                ),
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "reason": (
                        ""
                        if memory_update_allowed
                        else "answer_verification_failed"
                    ),
                    "model_controlled": bool(
                        memory_update_plan and memory_update_plan.get("model_controlled")
                    ),
                    "long_term_action_count": len(
                        (memory_update_plan or {}).get("long_term_actions") or []
                    ),
                },
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
