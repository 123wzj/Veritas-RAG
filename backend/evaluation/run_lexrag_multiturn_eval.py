# -*- coding: utf-8 -*-
"""Run LexRAG through the real multi-turn graph and RAGAS evaluators."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.import_lexrag_kb import project_doc_id
from backend.evaluation.run_multiturn_rag_eval import (
    apply_memory_plan,
    build_judge,
    cleanup_sessions,
    create_session,
    judge_turn,
    persist_message,
    read_memory,
    read_session,
    restore_profile,
    session_contains_query,
    snapshot_profile,
)
from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR, run_ragas
from backend.evaluation.run_rgb_e2e_global_pool_eval import patch_graph_llms_with_deepseek, percentile


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "evaluation" / "lexrag" / "prepared"
    / "lexrag_conversations_80x5.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LexRAG multi-turn RAG.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--limit-conversations", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-reflections", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--case-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Run standard per-metric RAGAS after the five per-turn composite judge calls.",
    )
    parser.add_argument("--skip-ragas", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness,context_precision,answer_correctness",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    patch_graph_llms_with_deepseek()
    conversations = load_jsonl(args.input)[:args.limit_conversations]
    judge = None if args.skip_judge else build_judge()
    profile_snapshot = snapshot_profile(args.user_id)
    session_ids: List[str] = []
    conversation_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    output: Optional[Path] = None
    try:
        for conversation_index, conversation in enumerate(conversations, start=1):
            session_id = f"lexmt_{conversation['source_dialogue_id']}_{uuid.uuid4().hex[:8]}"[:64]
            session_ids.append(session_id)
            create_session(session_id, args.user_id, args.kb_id)
            turn_rows: List[Dict[str, Any]] = []
            for turn_index, turn in enumerate(conversation.get("turns") or [], start=1):
                print(
                    f"[{conversation_index}/{len(conversations)} "
                    f"turn {turn_index}/{len(conversation.get('turns') or [])}] "
                    f"started id={conversation['id']}",
                    flush=True,
                )
                row = await run_turn(
                    conversation,
                    turn,
                    turn_index,
                    turn_rows,
                    session_id,
                    args,
                    judge,
                )
                turn_rows.append(row)
                print(
                    f"[{conversation_index}/{len(conversations)} "
                    f"turn {turn_index}/{len(conversation.get('turns') or [])}] "
                    f"{'ok' if not row.get('error') else 'failed'} id={conversation['id']}",
                    flush=True,
                )
            conversation_rows.append({
                "id": conversation["id"],
                "category": conversation.get("category"),
                "session_id": session_id,
                "turns": turn_rows,
                "pass": all(turn_passed(row) for row in turn_rows),
            })
    finally:
        cleanup_sessions(session_ids)
        restore_profile(args.user_id, profile_snapshot)

    ragas_enabled = bool(args.with_ragas and not args.skip_ragas)
    checkpoint_report = build_report(
        args,
        conversations,
        conversation_rows,
        {},
        started,
        ragas_enabled=ragas_enabled,
        stage="graph_and_judge_complete",
    )
    output = save_report(checkpoint_report, args.report_dir, args.kb_id)
    print(f"[checkpoint] report={output}", flush=True)

    ragas_report: Dict[str, Any] = {}
    if ragas_enabled:
        ragas_rows = build_ragas_rows(conversation_rows)
        if ragas_rows:
            print(
                f"[ragas] started rows={len(ragas_rows)} metrics={args.ragas_metrics}",
                flush=True,
            )
            ragas_report = run_ragas(
                ragas_rows,
                [item.strip() for item in args.ragas_metrics.split(",") if item.strip()],
            )
            merge_ragas_rows(conversation_rows, ragas_report.get("rows") or [])
            print("[ragas] completed", flush=True)

    report = build_report(
        args,
        conversations,
        conversation_rows,
        ragas_report,
        started,
        ragas_enabled=ragas_enabled,
        stage="complete",
    )
    save_report(report, args.report_dir, args.kb_id, output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"report={output}", flush=True)


def build_report(
    args: argparse.Namespace,
    conversations: List[Dict[str, Any]],
    conversation_rows: List[Dict[str, Any]],
    ragas_report: Dict[str, Any],
    started: float,
    *,
    ragas_enabled: bool,
    stage: str,
) -> Dict[str, Any]:
    return {
        "dataset": "LexRAG",
        "mode": "multiturn_e2e_ragas",
        "stage": stage,
        "config": {
            "input": str(args.input),
            "kb_id": args.kb_id,
            "conversation_count": len(conversations),
            "top_k": args.top_k,
            "ragas_metrics": args.ragas_metrics,
            "ragas_enabled": ragas_enabled,
            "skip_judge": args.skip_judge,
            "evaluation_strategy": (
                "one_composite_judge_call_per_successful_turn_plus_optional_ragas"
            ),
        },
        "seconds": round(time.perf_counter() - started, 3),
        "summary": summarize(conversation_rows, ragas_report, args.top_k),
        "ragas": ragas_report,
        "conversations": conversation_rows,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def run_turn(
    conversation: Dict[str, Any],
    turn: Dict[str, Any],
    turn_index: int,
    previous_turns: List[Dict[str, Any]],
    session_id: str,
    args: argparse.Namespace,
    judge,
) -> Dict[str, Any]:
    from backend.graph.graph import run_agentic_rag

    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    query = turn["query"]
    pre_memory = read_memory(args.user_id, session_id, query)
    persist_message(session_id, "user", query, request_id=request_id)
    try:
        async def consume() -> Optional[Dict[str, Any]]:
            final_state = None
            async for state in run_agentic_rag(
                query=query,
                user_id=args.user_id,
                request_id=request_id,
                kb_id=args.kb_id,
                session_id=session_id,
                web_enabled=False,
                stream_events=False,
                top_k=args.top_k,
                max_reflections=args.max_reflections,
                max_steps=args.max_steps,
            ):
                if isinstance(state, dict):
                    final_state = state
            return final_state

        final_state = await asyncio.wait_for(consume(), timeout=args.case_timeout_seconds)
        if not final_state or final_state.get("error"):
            raise RuntimeError(str((final_state or {}).get("error") or "empty graph result"))
        answer = final_state.get("final_answer") or ""
        citations = final_state.get("citations") or []
        message_id = persist_message(session_id, "assistant", answer, citations, request_id=request_id)
        apply_memory_plan(final_state.get("memory_update_plan"), assistant_message_id=message_id)
        post_session = read_session(session_id)
        evidence = final_state.get("selected_evidence") or []
        retrieval = retrieval_metrics(
            evidence,
            turn.get("gold_doc_ids") or [],
            citations,
            args.top_k,
        )
        judged = {}
        if judge is not None:
            judged = await judge_turn(
                judge,
                conversation,
                turn,
                answer,
                citations,
                evidence,
                previous_turns,
            )
        return {
            "turn_index": turn_index,
            "query": query,
            "reference": turn.get("reference"),
            "requires_history": bool(turn.get("requires_history")),
            "gold_doc_ids": turn.get("gold_doc_ids") or [],
            "gold_contexts": turn.get("gold_contexts") or [],
            "keywords": turn.get("keywords") or [],
            "answer": answer,
            "contexts": unique_contexts(evidence),
            "selected_evidence": evidence,
            "citations": citations,
            "retrieval": retrieval,
            "query_rewritten": final_state.get("query_rewritten"),
            "route_type": final_state.get("route_type"),
            "route_reason": final_state.get("route_reason"),
            "evidence_grade": final_state.get("evidence_grade"),
            "verification": final_state.get("verification"),
            "latency_breakdown_ms": final_state.get("latency_breakdown_ms") or {},
            "memory_read_success": (
                True
                if not turn.get("requires_history")
                else bool(pre_memory.get("recent_conversations") or pre_memory.get("session_summary"))
            ),
            "memory_write_success": session_contains_query(post_session, query),
            "message_persistence_success": int(post_session.get("message_count") or 0) == turn_index * 2,
            "judge": judged,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "turn_index": turn_index,
            "query": query,
            "reference": turn.get("reference"),
            "gold_doc_ids": turn.get("gold_doc_ids") or [],
            "answer": "",
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def unique_contexts(evidence: List[Dict[str, Any]]) -> List[str]:
    contexts = []
    for item in evidence:
        text = item.get("support_snippet") or item.get("snippet") or ""
        if text and text not in contexts:
            contexts.append(text)
    return contexts


def retrieval_metrics(
    evidence: List[Dict[str, Any]],
    gold_source_doc_ids: List[str],
    citations: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, float]:
    gold = {project_doc_id(doc_id) for doc_id in gold_source_doc_ids}
    ranked = [item.get("doc_id") for item in evidence[:top_k] if item.get("doc_id")]
    hits = [index + 1 for index, doc_id in enumerate(ranked) if doc_id in gold]
    cited = {item.get("doc_id") for item in citations if item.get("doc_id")}
    return {
        f"gold_article_recall@{top_k}": round(len(set(ranked) & gold) / max(1, len(gold)), 6),
        f"gold_article_hit@{top_k}": 1.0 if hits else 0.0,
        f"gold_article_mrr@{top_k}": round(1.0 / min(hits), 6) if hits else 0.0,
        "citation_to_gold_article": round(len(cited & gold) / max(1, len(cited)), 6),
    }


def build_ragas_rows(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for conversation in conversations:
        history: List[str] = []
        for turn in conversation.get("turns") or []:
            if turn.get("error") or not turn.get("answer") or not turn.get("contexts"):
                continue
            rows.append({
                "id": f"{conversation['id']}:{turn['turn_index']}",
                "question": "\n".join(history[-8:] + [f"用户：{turn['query']}"]),
                "answer": turn["answer"],
                "contexts": turn["contexts"],
                "reference": turn.get("reference"),
            })
            history.extend([f"用户：{turn['query']}", f"助手：{turn['answer']}"])
    return rows


def merge_ragas_rows(conversations: List[Dict[str, Any]], ragas_rows: List[Dict[str, Any]]) -> None:
    index = 0
    for conversation in conversations:
        for turn in conversation.get("turns") or []:
            if turn.get("error") or not turn.get("answer") or not turn.get("contexts"):
                continue
            if index < len(ragas_rows):
                turn["ragas"] = {
                    key: value
                    for key, value in ragas_rows[index].items()
                    if isinstance(value, (int, float))
                }
            index += 1


def turn_passed(turn: Dict[str, Any]) -> bool:
    if turn.get("error"):
        return False
    judge = turn.get("judge") or {}
    return str(judge.get("verdict") or "").lower() == "pass" or float(judge.get("overall") or 0) >= 0.8


def summarize(
    conversations: List[Dict[str, Any]],
    ragas_report: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    turns = [turn for row in conversations for turn in row.get("turns") or []]
    successful = [turn for turn in turns if not turn.get("error")]
    latencies = [float(turn.get("latency_ms") or 0) for turn in turns]
    retrieval_keys = (
        f"gold_article_recall@{top_k}",
        f"gold_article_hit@{top_k}",
        f"gold_article_mrr@{top_k}",
        "citation_to_gold_article",
    )
    retrieval = {
        key: round(
            sum(float(turn.get("retrieval", {}).get(key) or 0) for turn in successful)
            / max(1, len(successful)),
            6,
        )
        for key in retrieval_keys
    }
    judge_success = [
        turn for turn in successful
        if isinstance((turn.get("judge") or {}).get("answer_correctness"), (int, float))
    ]
    judge_summary = {}
    for key in (
        "answer_correctness",
        "faithfulness",
        "citation_quality",
        "context_resolution",
        "history_consistency",
        "overall",
    ):
        values = [
            float(turn["judge"][key])
            for turn in judge_success
            if isinstance(turn["judge"].get(key), (int, float))
        ]
        if values:
            judge_summary[key] = round(sum(values) / len(values), 6)
    return {
        "conversation_count": len(conversations),
        "turn_count": len(turns),
        "success_rate": round(len(successful) / max(1, len(turns)), 6),
        "evaluation_judge_call_count": len(judge_success),
        "judge_success_rate": round(len(judge_success) / max(1, len(successful)), 6),
        "turn_pass_rate": round(sum(1 for turn in turns if turn_passed(turn)) / max(1, len(turns)), 6),
        "conversation_pass_rate": round(
            sum(1 for row in conversations if row.get("pass")) / max(1, len(conversations)),
            6,
        ),
        "memory_read_success_rate": round(
            sum(1 for turn in successful if not turn.get("requires_history") or turn.get("memory_read_success"))
            / max(1, len(successful)),
            6,
        ),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "retrieval": retrieval,
        "generation_judge": judge_summary,
        "ragas": ragas_report.get("summary") or {},
    }


def save_report(
    report: Dict[str, Any],
    report_dir: Path,
    kb_id: int,
    path: Optional[Path] = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = report_dir / f"lexrag_multiturn_eval_kb{kb_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    asyncio.run(main_async())
