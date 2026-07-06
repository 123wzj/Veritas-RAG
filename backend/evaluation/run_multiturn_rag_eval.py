# -*- coding: utf-8 -*-
"""Evaluate multi-turn RAG through real session, message, memory, and graph flow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.db.mysql.connection import SessionLocal, init_db
from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR
from backend.evaluation.run_rgb_e2e_global_pool_eval import patch_graph_llms_with_deepseek, percentile
from backend.evaluation.run_rgb_fixed_context_eval import DeepSeekDirectChat, safe_json_object
from backend.models.database.user import (
    ConversationBranchTable,
    LongTermMemoryTable,
    MemoryUpdateLogTable,
    MessageTable,
    SessionMemoryTable,
    SessionTable,
    UserProfileTable,
)
from backend.services.context.context_assembler import estimate_tokens
from backend.services.memory.memory_service import memory_service


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "rgb" / "rgb_multiturn_6x4.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn RAG sessions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-reflections", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--case-timeout-seconds", type=int, default=120)
    parser.add_argument("--web-enabled", action="store_true")
    return parser.parse_args()


def load_conversations(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit is not None else rows


async def main_async() -> None:
    args = parse_args()
    init_db()
    patch_graph_llms_with_deepseek()
    conversations = load_conversations(args.input, args.limit_conversations)
    judge = build_judge()
    profile_snapshot = snapshot_profile(args.user_id)
    created_session_ids: List[str] = []
    conversation_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for conversation_index, conversation in enumerate(conversations, start=1):
            session_id = f"rgbmt_{conversation['id']}_{uuid.uuid4().hex[:8]}"[:64]
            created_session_ids.append(session_id)
            create_session(session_id, args.user_id, args.kb_id)
            turn_rows: List[Dict[str, Any]] = []

            for turn_index, turn in enumerate(conversation.get("turns") or [], start=1):
                row = await run_turn(
                    conversation=conversation,
                    turn=turn,
                    turn_index=turn_index,
                    previous_turns=turn_rows,
                    session_id=session_id,
                    args=args,
                    judge=judge,
                )
                turn_rows.append(row)
                status = "ok" if not row.get("error") else "failed"
                print(
                    f"[{conversation_index}/{len(conversations)} "
                    f"turn {turn_index}/{len(conversation.get('turns') or [])}] "
                    f"{status} id={conversation['id']}"
                )

            conversation_rows.append({
                "id": conversation["id"],
                "category": conversation.get("category"),
                "session_id": session_id,
                "pass": all(turn_passed(row) for row in turn_rows),
                "turns": turn_rows,
            })
    finally:
        cleanup_sessions(created_session_ids)
        restore_profile(args.user_id, profile_snapshot)

    report = {
        "dataset": "RGB-derived multi-turn",
        "mode": "multiturn_graph",
        "config": {
            "input": str(args.input),
            "kb_id": args.kb_id,
            "user_id": args.user_id,
            "top_k": args.top_k,
            "conversation_count": len(conversations),
            "max_reflections": args.max_reflections,
            "max_steps": args.max_steps,
            "case_timeout_seconds": args.case_timeout_seconds,
            "web_enabled": args.web_enabled,
        },
        "seconds": round(time.perf_counter() - started, 3),
        "summary": summarize(conversation_rows),
        "conversations": conversation_rows,
    }
    output = save_report(report, args.report_dir)
    print_summary(report["summary"])
    print(f"report={output}")


def build_judge() -> DeepSeekDirectChat:
    from backend.core.config import settings

    return DeepSeekDirectChat(
        model=settings.DEEPSEEK_PRO_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=None,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=3,
    )


async def run_turn(
    conversation: Dict[str, Any],
    turn: Dict[str, Any],
    turn_index: int,
    previous_turns: List[Dict[str, Any]],
    session_id: str,
    args: argparse.Namespace,
    judge: DeepSeekDirectChat,
) -> Dict[str, Any]:
    from backend.graph.graph import run_agentic_rag

    started = time.perf_counter()
    query = turn["query"]
    request_id = str(uuid.uuid4())
    pre_memory = read_memory(args.user_id, session_id, query)
    persist_message(session_id, "user", query, request_id=request_id)

    try:
        async def consume_graph() -> Optional[Dict[str, Any]]:
            result: Optional[Dict[str, Any]] = None
            async for state in run_agentic_rag(
                query=query,
                user_id=args.user_id,
                request_id=request_id,
                kb_id=args.kb_id,
                session_id=session_id,
                web_enabled=args.web_enabled,
                stream_events=False,
                top_k=args.top_k,
                max_reflections=args.max_reflections,
                max_steps=args.max_steps,
            ):
                if isinstance(state, dict):
                    result = state
            return result

        final_state = await asyncio.wait_for(
            consume_graph(),
            timeout=max(1, args.case_timeout_seconds),
        )
        if not final_state:
            raise RuntimeError("RAG graph did not return final state.")
        if final_state.get("error"):
            raise RuntimeError(str(final_state["error"]))

        answer = final_state.get("final_answer") or ""
        citations = final_state.get("citations") or []
        assistant_message_id = persist_message(
            session_id,
            "assistant",
            answer,
            citations,
            request_id=request_id,
        )
        apply_memory_plan(
            final_state.get("memory_update_plan"),
            assistant_message_id=assistant_message_id,
        )
        post_session = read_session(session_id)
        selected_evidence = final_state.get("selected_evidence") or []
        judged = await judge_turn(
            judge=judge,
            conversation=conversation,
            turn=turn,
            answer=answer,
            citations=citations,
            selected_evidence=selected_evidence,
            previous_turns=previous_turns,
        )
        memory_read_success = (
            True
            if not turn.get("requires_history")
            else bool(pre_memory.get("recent_conversations") or pre_memory.get("session_summary"))
        )
        memory_write_success = session_contains_query(post_session, query)
        expected_message_count = turn_index * 2
        message_persistence_success = int(post_session.get("message_count") or 0) == expected_message_count

        return {
            "turn_index": turn_index,
            "query": query,
            "reference": turn.get("reference"),
            "expected_behavior": turn.get("expected_behavior"),
            "requires_history": bool(turn.get("requires_history")),
            "answer": answer,
            "query_rewritten": final_state.get("query_rewritten"),
            "route_type": final_state.get("route_type"),
            "selected_evidence": selected_evidence,
            "citations": citations,
            "confidence": final_state.get("confidence", 0.0),
            "verification": final_state.get("verification"),
            "evidence_grade": final_state.get("evidence_grade"),
            "used_web_search": bool(final_state.get("used_web_search")),
            "pre_memory": {
                "session_summary": pre_memory.get("session_summary") or "",
                "recent_conversations": pre_memory.get("recent_conversations") or [],
                "working_memory": pre_memory.get("working_memory") or [],
            },
            "post_session": post_session,
            "memory_read_success": memory_read_success,
            "memory_write_success": memory_write_success,
            "message_persistence_success": message_persistence_success,
            "judge": judged,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "turn_index": turn_index,
            "query": query,
            "reference": turn.get("reference"),
            "expected_behavior": turn.get("expected_behavior"),
            "requires_history": bool(turn.get("requires_history")),
            "answer": "",
            "memory_read_success": False,
            "memory_write_success": False,
            "message_persistence_success": False,
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


async def judge_turn(
    judge: DeepSeekDirectChat,
    conversation: Dict[str, Any],
    turn: Dict[str, Any],
    answer: str,
    citations: List[Dict[str, Any]],
    selected_evidence: List[Dict[str, Any]],
    previous_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    history = [
        {"query": row.get("query"), "answer": row.get("answer")}
        for row in previous_turns
    ]
    evidence_items = [
        {
            "evidence_id": item.get("evidence_id"),
            "source_type": item.get("source_type"),
            "title": item.get("title"),
            "text": item.get("support_snippet") or item.get("snippet") or "",
        }
        for item in selected_evidence
    ]
    prompt = {
        "category": conversation.get("category"),
        "history": history,
        "current_query": turn.get("query"),
        "reference": turn.get("reference"),
        "expected_behavior": turn.get("expected_behavior"),
        "requires_history": turn.get("requires_history"),
        "correction_required": turn.get("correction_required", False),
        "session_isolation_required": turn.get("session_isolation_required", False),
        "answer": answer,
        "citations": citations,
        "evidence_items": evidence_items,
    }
    system_prompt = """你是企业RAG多轮对话评估器。请结合当前会话history、reference和证据评价本轮答案。
输出严格JSON，字段：
answer_correctness, faithfulness, citation_quality, context_resolution,
history_consistency, correction_adherence, session_isolation, overall, verdict, reason。
所有适用分数为0到1，不适用字段为null。
规则：
1. requires_history=true时，context_resolution重点判断代词、省略和跨轮条件是否正确恢复。
2. correction_required=true时，不得继续采用已被用户否定的信息。
3. session_isolation_required=true时，当前会话没有指代对象，答案应澄清或拒答，不能借用其他会话内容。
4. expected_behavior包含refuse时，安全拒答是正确答案；包含conflict时，应识别冲突，不能按多数证据猜测。
5. faithfulness和citation_quality只按evidence_items及实际引用判断。
6. verdict只输出pass或fail；overall>=0.8且核心行为正确才可pass。"""
    response = await judge.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ])
    parsed = safe_json_object(response.content)
    for key in (
        "answer_correctness",
        "faithfulness",
        "citation_quality",
        "context_resolution",
        "history_consistency",
        "correction_adherence",
        "session_isolation",
        "overall",
    ):
        value = parsed.get(key)
        if value is None:
            continue
        try:
            parsed[key] = round(min(max(float(value), 0.0), 1.0), 4)
        except (TypeError, ValueError):
            parsed[key] = None
    parsed.setdefault("verdict", "unknown")
    parsed.setdefault("reason", "")
    return parsed


def create_session(session_id: str, user_id: int, kb_id: int) -> None:
    db = SessionLocal()
    try:
        db.add(SessionTable(
            session_id=session_id,
            user_id=user_id,
            kb_id=kb_id,
            message_count=0,
            summary=None,
            context={},
        ))
        db.commit()
    finally:
        db.close()


def persist_message(
    session_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
) -> int:
    db = SessionLocal()
    try:
        message = MessageTable(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations,
            token_count=estimate_tokens(content),
            request_id=request_id,
        )
        db.add(message)
        session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
        if session:
            session.message_count = int(session.message_count or 0) + 1
        db.commit()
        db.refresh(message)
        return int(message.id)
    finally:
        db.close()


def read_memory(user_id: int, session_id: str, query: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        return memory_service.get_session_memory(user_id, session_id, query, db)
    finally:
        db.close()


def apply_memory_plan(
    plan: Optional[Dict[str, Any]],
    *,
    assistant_message_id: Optional[int],
) -> None:
    if not plan:
        return
    db = SessionLocal()
    try:
        memory_service.apply_memory_update_plan(
            plan,
            db=db,
            assistant_message_id=assistant_message_id,
        )
        db.commit()
    finally:
        db.close()


def read_session(session_id: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
        if not session:
            return {}
        return {
            "summary": session.summary or "",
            "context": copy.deepcopy(session.context or {}),
            "message_count": int(session.message_count or 0),
            "session_memory_version": int(
                getattr(
                    db.query(SessionMemoryTable)
                    .filter(SessionMemoryTable.session_id == session_id)
                    .first(),
                    "version",
                    0,
                )
                or 0
            ),
        }
    finally:
        db.close()


def session_contains_query(session: Dict[str, Any], query: str) -> bool:
    return int(session.get("session_memory_version") or 0) > 1


def snapshot_profile(user_id: int) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if not profile:
            return None
        return {
            "preferred_language": profile.preferred_language,
            "interests": copy.deepcopy(profile.interests),
            "interaction_style": profile.interaction_style,
            "frequently_asked_topics": copy.deepcopy(profile.frequently_asked_topics),
            "long_term_facts": copy.deepcopy(profile.long_term_facts),
            "working_preferences": copy.deepcopy(profile.working_preferences),
        }
    finally:
        db.close()


def restore_profile(user_id: int, snapshot: Optional[Dict[str, Any]]) -> None:
    if snapshot is None:
        return
    db = SessionLocal()
    try:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if profile:
            for key, value in snapshot.items():
                setattr(profile, key, value)
            db.commit()
    finally:
        db.close()


def cleanup_sessions(session_ids: List[str]) -> None:
    if not session_ids:
        return
    db = SessionLocal()
    try:
        db.query(MemoryUpdateLogTable).filter(
            MemoryUpdateLogTable.session_id.in_(session_ids)
        ).delete(synchronize_session=False)
        db.query(LongTermMemoryTable).filter(
            LongTermMemoryTable.source_session_id.in_(session_ids)
        ).delete(synchronize_session=False)
        db.query(SessionMemoryTable).filter(
            SessionMemoryTable.session_id.in_(session_ids)
        ).delete(synchronize_session=False)
        db.query(MessageTable).filter(MessageTable.session_id.in_(session_ids)).delete(
            synchronize_session=False
        )
        db.query(ConversationBranchTable).filter(
            ConversationBranchTable.session_id.in_(session_ids)
        ).delete(synchronize_session=False)
        db.query(SessionTable).filter(SessionTable.session_id.in_(session_ids)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def turn_passed(row: Dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    judged = row.get("judge") or {}
    verdict = str(judged.get("verdict") or "").strip().lower()
    if verdict == "pass":
        return True
    overall = judged.get("overall")
    return isinstance(overall, (int, float)) and float(overall) >= 0.8


def summarize(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    turns = [turn for conversation in conversations for turn in conversation.get("turns") or []]
    successful = [turn for turn in turns if not turn.get("error")]
    latencies = [float(turn.get("latency_ms") or 0) for turn in turns]
    history_turns = [turn for turn in successful if turn.get("requires_history")]
    numeric_keys = (
        "answer_correctness",
        "faithfulness",
        "citation_quality",
        "context_resolution",
        "history_consistency",
        "correction_adherence",
        "session_isolation",
        "overall",
    )
    judge_summary: Dict[str, float] = {}
    for key in numeric_keys:
        values = [
            float((turn.get("judge") or {}).get(key))
            for turn in successful
            if isinstance((turn.get("judge") or {}).get(key), (int, float))
        ]
        if values:
            judge_summary[key] = round(sum(values) / len(values), 6)

    by_category: Dict[str, Dict[str, Any]] = {}
    for conversation in conversations:
        category = str(conversation.get("category") or "unknown")
        category_turns = conversation.get("turns") or []
        by_category[category] = {
            "turn_count": len(category_turns),
            "pass_rate": round(
                sum(1 for turn in category_turns if turn_passed(turn)) / max(1, len(category_turns)),
                6,
            ),
            "conversation_pass": bool(conversation.get("pass")),
        }

    return {
        "conversation_count": len(conversations),
        "turn_count": len(turns),
        "success_rate": round(len(successful) / max(1, len(turns)), 6),
        "turn_pass_rate": round(sum(1 for turn in turns if turn_passed(turn)) / max(1, len(turns)), 6),
        "conversation_pass_rate": round(
            sum(1 for conversation in conversations if conversation.get("pass"))
            / max(1, len(conversations)),
            6,
        ),
        "memory_read_success_rate": round(
            sum(1 for turn in history_turns if turn.get("memory_read_success"))
            / max(1, len(history_turns)),
            6,
        ),
        "memory_write_success_rate": round(
            sum(1 for turn in successful if turn.get("memory_write_success"))
            / max(1, len(successful)),
            6,
        ),
        "message_persistence_success_rate": round(
            sum(1 for turn in successful if turn.get("message_persistence_success"))
            / max(1, len(successful)),
            6,
        ),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "judge": judge_summary,
        "by_category": by_category,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("summary")
    for key in (
        "conversation_count",
        "turn_count",
        "success_rate",
        "turn_pass_rate",
        "conversation_pass_rate",
        "memory_read_success_rate",
        "memory_write_success_rate",
        "message_persistence_success_rate",
        "avg_latency_ms",
        "p95_latency_ms",
    ):
        print(f"- {key}: {summary.get(key)}")
    if summary.get("judge"):
        print("- judge: " + ", ".join(
            f"{key}={value:.4f}" for key, value in summary["judge"].items()
        ))


def save_report(report: Dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"rgb_multiturn_eval_kb{report['config']['kb_id']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    asyncio.run(main_async())
