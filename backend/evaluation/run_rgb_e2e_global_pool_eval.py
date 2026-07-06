# -*- coding: utf-8 -*-
"""Run RGB end-to-end global-pool evaluation through the real RAG graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.db.mysql.connection import SessionLocal, init_db
from backend.evaluation.import_rgb_global_pool import DEFAULT_INPUT, rgb_doc_id
from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR
from backend.evaluation.run_rgb_fixed_context_eval import DeepSeekDirectChat, run_deepseek_judge
from backend.models.database.knowledge import DocumentTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB global-pool retrieval-to-generation E2E.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--samples-per-type", type=int, default=None)
    parser.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="Comma-separated RGB question types to include.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--pipeline", choices=["retrieval", "direct", "graph"], default="graph")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--llm", choices=["deepseek", "project"], default="deepseek")
    parser.add_argument("--judge", choices=["deepseek", "none"], default="deepseek")
    parser.add_argument("--web-enabled", action="store_true")
    parser.add_argument("--max-reflections", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--case-timeout-seconds", type=int, default=120)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    init_db()
    question_types = {
        item.strip()
        for item in (args.question_types or "").split(",")
        if item.strip()
    }
    samples = load_samples(args.input, args.limit, args.samples_per_type, question_types)
    if not samples:
        raise RuntimeError("No RGB samples loaded.")

    if args.llm == "deepseek":
        patch_graph_llms_with_deepseek()

    doc_meta = load_doc_metadata(args.kb_id)
    if not doc_meta:
        raise RuntimeError(f"No imported RGB docs found for kb_id={args.kb_id}. Run import_rgb_global_pool.py first.")

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        row = await run_case(sample, args, doc_meta)
        rows.append(row)
        status = "ok" if not row.get("error") else "failed"
        print(f"[{index}/{len(samples)}] {status} id={row.get('id')}")

    judge_report: Dict[str, Any] = {}
    if args.judge == "deepseek":
        judge_rows = [row for row in rows if not row.get("error") and row.get("answer")]
        judge_report = await run_deepseek_judge(judge_rows)

    report = {
        "dataset": "RGB",
        "mode": "e2e_global_pool",
        "kb_id": args.kb_id,
        "config": {
            "input": str(args.input),
            "limit": args.limit,
            "samples_per_type": args.samples_per_type,
            "question_types": sorted(question_types),
            "top_k": args.top_k,
            "pipeline": args.pipeline,
            "skip_reranker": args.skip_reranker,
            "llm": args.llm,
            "judge": args.judge,
            "user_id": args.user_id,
            "web_enabled": args.web_enabled,
            "max_reflections": args.max_reflections,
            "max_steps": args.max_steps,
            "case_timeout_seconds": args.case_timeout_seconds,
        },
        "seconds": round(time.perf_counter() - started, 3),
        "sample_count": len(rows),
        "success_count": sum(1 for row in rows if not row.get("error")),
        "summary": summarize(rows, judge_report, args.top_k),
        "deepseek_judge": judge_report,
        "samples": rows,
    }
    print_summary(report)
    output_path = save_report(report, args.report_dir)
    print(f"report={output_path}")


def load_samples(
    path: Path,
    limit: Optional[int],
    samples_per_type: Optional[int] = None,
    question_types: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if question_types:
        rows = [
            row for row in rows
            if str(row.get("question_type") or "unknown") in question_types
        ]
    if samples_per_type is not None:
        selected: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for row in rows:
            question_type = str(row.get("question_type") or "unknown")
            if counts.get(question_type, 0) >= max(0, samples_per_type):
                continue
            selected.append(row)
            counts[question_type] = counts.get(question_type, 0) + 1
        return selected
    return rows[:limit] if limit is not None else rows


async def run_case(sample: Dict[str, Any], args: argparse.Namespace, doc_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    expected = build_expected_doc_labels(sample)
    try:
        if args.pipeline == "retrieval":
            return await run_case_retrieval(sample, args, doc_meta, expected, started)
        if args.pipeline == "direct":
            return await run_case_direct(sample, args, doc_meta, expected, started)
        return await run_case_graph(sample, args, doc_meta, expected, started)
    except Exception as exc:
        return {
            "id": sample.get("id"),
            "question": sample.get("question"),
            "answer": "",
            "reference": sample.get("ground_truth"),
            "original_reference": sample.get("original_ground_truth"),
            "contexts": [],
            "question_type": sample.get("question_type"),
            "expected_behavior": sample.get("expected_behavior"),
            "expected_doc_labels": expected,
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


async def run_case_graph(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    doc_meta: Dict[str, Dict[str, Any]],
    expected: Dict[str, str],
    started: float,
) -> Dict[str, Any]:
    from backend.graph.graph import run_agentic_rag

    try:
        async def consume_graph() -> Optional[Dict[str, Any]]:
            result: Optional[Dict[str, Any]] = None
            async for state in run_agentic_rag(
                query=sample["question"],
                user_id=args.user_id,
                kb_id=args.kb_id,
                session_id=None,
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

        retrieved_docs = collect_docs(final_state, "retrieved_docs")
        reranked_docs = collect_docs(final_state, "reranked_docs")
        selected_evidence = final_state.get("selected_evidence") or []
        selected_docs = evidence_to_docs(selected_evidence)
        selected_contexts = [
            item.get("support_snippet") or item.get("snippet") or ""
            for item in selected_evidence
            if item.get("support_snippet") or item.get("snippet")
        ]

        retrieval_metrics = {
            "retrieved": compute_retrieval_metrics(retrieved_docs, expected, doc_meta, args.top_k),
            "reranked": compute_retrieval_metrics(reranked_docs, expected, doc_meta, args.top_k),
            "selected": compute_retrieval_metrics(selected_docs, expected, doc_meta, args.top_k),
        }
        return {
            "id": sample["id"],
            "question": sample["question"],
            "answer": final_state.get("final_answer") or "",
            "reference": sample.get("ground_truth"),
            "original_reference": sample.get("original_ground_truth"),
            "contexts": selected_contexts,
            "context_labels": [
                label_for_doc(item.get("doc_id"), expected, doc_meta)
                for item in selected_evidence
            ],
            "question_type": sample.get("question_type"),
            "expected_behavior": sample.get("expected_behavior"),
            "expected_doc_labels": expected,
            "retrieval_metrics": retrieval_metrics,
            "retrieved_doc_ids": [doc.get("doc_id") for doc in retrieved_docs[: args.top_k] if doc.get("doc_id")],
            "reranked_doc_ids": [doc.get("doc_id") for doc in reranked_docs[: args.top_k] if doc.get("doc_id")],
            "selected_doc_ids": [doc.get("doc_id") for doc in selected_docs[: args.top_k] if doc.get("doc_id")],
            "selected_evidence": selected_evidence,
            "citations": final_state.get("citations") or [],
            "confidence": final_state.get("confidence", 0.0),
            "verification": final_state.get("verification"),
            "evidence_grade": final_state.get("evidence_grade"),
            "used_web_search": bool(final_state.get("used_web_search")),
            "event_names": [
                event.get("event")
                for event in final_state.get("events") or []
                if isinstance(event, dict) and event.get("event")
            ],
            "latency_breakdown_ms": final_state.get("latency_breakdown_ms") or {},
            "error": final_state.get("error"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "id": sample.get("id"),
            "question": sample.get("question"),
            "answer": "",
            "reference": sample.get("ground_truth"),
            "original_reference": sample.get("original_ground_truth"),
            "contexts": [],
            "question_type": sample.get("question_type"),
            "expected_behavior": sample.get("expected_behavior"),
            "expected_doc_labels": expected,
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


async def run_case_direct(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    doc_meta: Dict[str, Dict[str, Any]],
    expected: Dict[str, str],
    started: float,
) -> Dict[str, Any]:
    from backend.graph.nodes import generation_nodes
    from backend.services.retrieval.hybrid import hybrid_retriever
    from backend.services.retrieval.reranker import reranker

    query = sample["question"]
    retrieved_docs = await hybrid_retriever.retrieve_async(
        query=query,
        user_id=args.user_id,
        kb_id=args.kb_id,
        query_variants=[query],
        top_k=max(args.top_k * 3, 12),
    )
    if args.skip_reranker:
        reranked_docs = retrieved_docs[: max(args.top_k * 2, 8)]
    else:
        try:
            reranked_docs = reranker.rerank(
                query=query,
                documents=[doc.copy() for doc in retrieved_docs],
                top_k=max(args.top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )
        except Exception:
            reranked_docs = retrieved_docs[: max(args.top_k * 2, 8)]

    selected_evidence = build_selected_evidence(reranked_docs, args.top_k)
    result = await generation_nodes._generate_sub_answer(
        sub_question=query,
        route_type="knowledge_base",
        selected_evidence=selected_evidence,
        preferred_language="zh-CN",
        interaction_style="detailed",
        session_summary="",
        prompt_context=build_prompt_context(sample),
        working_memory=[],
        long_term_facts=[],
    )
    selected_docs = evidence_to_docs(selected_evidence)
    contexts = [
        item.get("support_snippet") or item.get("snippet") or ""
        for item in selected_evidence
        if item.get("support_snippet") or item.get("snippet")
    ]
    retrieval_metrics = {
        "retrieved": compute_retrieval_metrics(retrieved_docs, expected, doc_meta, args.top_k),
        "reranked": compute_retrieval_metrics(reranked_docs, expected, doc_meta, args.top_k),
        "selected": compute_retrieval_metrics(selected_docs, expected, doc_meta, args.top_k),
    }
    return {
        "id": sample["id"],
        "question": query,
        "answer": result.get("answer") or "",
        "reference": sample.get("ground_truth"),
        "original_reference": sample.get("original_ground_truth"),
        "contexts": contexts,
        "context_labels": [
            label_for_doc(item.get("doc_id"), expected, doc_meta)
            for item in selected_evidence
        ],
        "question_type": sample.get("question_type"),
        "expected_behavior": sample.get("expected_behavior"),
        "expected_doc_labels": expected,
        "retrieval_metrics": retrieval_metrics,
        "retrieved_doc_ids": [doc.get("doc_id") for doc in retrieved_docs[: args.top_k] if doc.get("doc_id")],
        "reranked_doc_ids": [doc.get("doc_id") for doc in reranked_docs[: args.top_k] if doc.get("doc_id")],
        "selected_doc_ids": [doc.get("doc_id") for doc in selected_docs[: args.top_k] if doc.get("doc_id")],
        "selected_evidence": selected_evidence,
        "citations": result.get("citations") or [],
        "confidence": result.get("confidence", 0.0),
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


async def run_case_retrieval(
    sample: Dict[str, Any],
    args: argparse.Namespace,
    doc_meta: Dict[str, Dict[str, Any]],
    expected: Dict[str, str],
    started: float,
) -> Dict[str, Any]:
    from backend.services.retrieval.hybrid import hybrid_retriever
    from backend.services.retrieval.reranker import reranker

    query = sample["question"]
    retrieved_docs = await hybrid_retriever.retrieve_async(
        query=query,
        user_id=args.user_id,
        kb_id=args.kb_id,
        query_variants=[query],
        top_k=max(args.top_k * 3, 12),
    )
    if args.skip_reranker:
        reranked_docs = retrieved_docs[: max(args.top_k * 2, 8)]
    else:
        try:
            reranked_docs = reranker.rerank(
                query=query,
                documents=[doc.copy() for doc in retrieved_docs],
                top_k=max(args.top_k * 2, 8),
                max_per_doc=3,
                max_per_parent=1,
            )
        except Exception:
            reranked_docs = retrieved_docs[: max(args.top_k * 2, 8)]

    selected_evidence = build_selected_evidence(reranked_docs, args.top_k)
    selected_docs = evidence_to_docs(selected_evidence)
    contexts = [
        item.get("support_snippet") or item.get("snippet") or ""
        for item in selected_evidence
        if item.get("support_snippet") or item.get("snippet")
    ]
    return {
        "id": sample["id"],
        "question": query,
        "answer": "",
        "reference": sample.get("ground_truth"),
        "original_reference": sample.get("original_ground_truth"),
        "contexts": contexts,
        "context_labels": [
            label_for_doc(item.get("doc_id"), expected, doc_meta)
            for item in selected_evidence
        ],
        "question_type": sample.get("question_type"),
        "expected_behavior": sample.get("expected_behavior"),
        "expected_doc_labels": expected,
        "retrieval_metrics": {
            "retrieved": compute_retrieval_metrics(retrieved_docs, expected, doc_meta, args.top_k),
            "reranked": compute_retrieval_metrics(reranked_docs, expected, doc_meta, args.top_k),
            "selected": compute_retrieval_metrics(selected_docs, expected, doc_meta, args.top_k),
        },
        "retrieved_doc_ids": [doc.get("doc_id") for doc in retrieved_docs[: args.top_k] if doc.get("doc_id")],
        "reranked_doc_ids": [doc.get("doc_id") for doc in reranked_docs[: args.top_k] if doc.get("doc_id")],
        "selected_doc_ids": [doc.get("doc_id") for doc in selected_docs[: args.top_k] if doc.get("doc_id")],
        "selected_evidence": selected_evidence,
        "citations": [],
        "confidence": 0.0,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def build_selected_evidence(docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for index, doc in enumerate(dedupe_docs(docs)[:top_k], start=1):
        parent_content = doc.get("parent_content") or ""
        child_content = doc.get("content") or ""
        snippet = parent_content[:1200] if parent_content else child_content[:600]
        support_snippet = child_content[:400] or snippet[:400]
        score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))
        evidence.append({
            "evidence_id": f"E{index}",
            "source_type": "knowledge_base",
            "doc_id": doc.get("doc_id"),
            "chunk_id": doc.get("chunk_id"),
            "parent_id": doc.get("parent_id"),
            "title": doc.get("parent_title") or doc.get("title") or "",
            "page_no": doc.get("parent_page_no") or doc.get("page_no"),
            "section_path": doc.get("parent_section_path") or doc.get("section_path"),
            "snippet": snippet,
            "support_snippet": support_snippet,
            "score": round(float(score or 0.0), 6),
            "match_query": doc.get("match_query"),
            "retrieval_type": doc.get("retrieval_type"),
        })
    return evidence


def build_prompt_context(sample: Dict[str, Any]) -> str:
    return (
        "请严格依据检索证据回答并标注关键结论的证据编号；"
        "证据不足时明确拒答，证据冲突且无法判断可靠性时说明冲突，不按数量多数猜测。"
    )


def patch_graph_llms_with_deepseek() -> None:
    from backend.core.config import settings
    from backend.graph.nodes import generation_nodes, query_nodes, reflection_nodes, retrieval_nodes

    flash_llm = DeepSeekDirectChat(
        model=settings.DEEPSEEK_FLASH_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=settings.RAGAS_LLM_MAX_RETRIES,
    )
    pro_llm = DeepSeekDirectChat(
        model=settings.DEEPSEEK_PRO_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=settings.RAGAS_LLM_MAX_RETRIES,
    )
    query_nodes.llm = flash_llm
    retrieval_nodes.llm = flash_llm
    reflection_nodes.llm = pro_llm
    generation_nodes.generation_llm = pro_llm
    generation_nodes.verification_llm = flash_llm


def build_expected_doc_labels(sample: Dict[str, Any]) -> Dict[str, str]:
    contexts = sample.get("contexts") or []
    labels = sample.get("context_labels") or infer_context_labels(sample)
    expected: Dict[str, str] = {}
    for index, text in enumerate(contexts):
        label = labels[index] if index < len(labels) else "unknown"
        expected[rgb_doc_id(sample["id"], index, text)] = label
    return expected


def infer_context_labels(sample: Dict[str, Any]) -> List[str]:
    contexts = sample.get("contexts") or []
    positive_count = int(sample.get("positive_context_count") or 0)
    labels = ["positive"] * positive_count
    labels.extend(["negative"] * max(0, len(contexts) - positive_count))
    return labels[:len(contexts)]


def collect_docs(state: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    docs = [doc for doc in state.get(key) or [] if isinstance(doc, dict)]
    for plan in state.get("sub_query_plans") or []:
        docs.extend(doc for doc in plan.get(key) or [] if isinstance(doc, dict))
    return dedupe_docs(docs)


def evidence_to_docs(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": item.get("doc_id"),
            "chunk_id": item.get("chunk_id"),
            "parent_id": item.get("parent_id"),
            "score": item.get("score", 0.0),
        }
        for item in evidence
        if item.get("doc_id")
    ]


def dedupe_docs(docs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    results = []
    for doc in docs:
        key = doc.get("doc_id") or doc.get("chunk_id")
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(doc)
    return results


def load_doc_metadata(kb_id: int) -> Dict[str, Dict[str, Any]]:
    db = SessionLocal()
    try:
        rows = db.query(DocumentTable.doc_id, DocumentTable.doc_metadata).filter(DocumentTable.kb_id == kb_id).all()
        return {doc_id: (metadata or {}) for doc_id, metadata in rows}
    finally:
        db.close()


def compute_retrieval_metrics(
    docs: List[Dict[str, Any]],
    expected_labels: Dict[str, str],
    global_doc_meta: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    top_docs = docs[:top_k]
    positive_ids = {doc_id for doc_id, label in expected_labels.items() if label == "positive"}
    hit_positive = {doc.get("doc_id") for doc in top_docs if doc.get("doc_id") in positive_ids}
    labels = [label_for_doc(doc.get("doc_id"), expected_labels, global_doc_meta) for doc in top_docs]
    wrong_count = sum(1 for label in labels if label == "positive_wrong")
    negative_count = sum(1 for label in labels if label in {"negative", "cross_sample", "unknown"})
    distractor_count = sum(
        1
        for doc in top_docs
        if (global_doc_meta.get(doc.get("doc_id")) or {}).get("evidence_label") == "distractor"
    )
    mrr = 0.0
    for rank, doc in enumerate(top_docs, start=1):
        if doc.get("doc_id") in positive_ids:
            mrr = 1.0 / rank
            break
    return {
        f"positive_recall@{top_k}": round(len(hit_positive) / len(positive_ids), 6) if positive_ids else None,
        f"positive_hit@{top_k}": (1.0 if hit_positive else 0.0) if positive_ids else None,
        f"positive_mrr@{top_k}": round(mrr, 6) if positive_ids else None,
        f"wrong_evidence_rate@{top_k}": round(wrong_count / max(1, len(top_docs)), 6),
        f"negative_evidence_rate@{top_k}": round(negative_count / max(1, len(top_docs)), 6),
        f"distractor_evidence_rate@{top_k}": round(distractor_count / max(1, len(top_docs)), 6),
        "result_count": float(len(top_docs)),
    }


def label_for_doc(doc_id: Optional[str], expected_labels: Dict[str, str], global_doc_meta: Dict[str, Dict[str, Any]]) -> str:
    if not doc_id:
        return "unknown"
    if doc_id in expected_labels:
        return expected_labels[doc_id]
    metadata = global_doc_meta.get(doc_id) or {}
    if metadata.get("dataset") == "RGB":
        return "cross_sample"
    return str(metadata.get("evidence_label") or "unknown")


def summarize(rows: List[Dict[str, Any]], judge_report: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    latencies = [float(row.get("latency_ms") or 0) for row in rows]
    summary: Dict[str, Any] = {
        "sample_count": len(rows),
        "success_count": len(successful),
        "success_rate": round(len(successful) / max(1, len(rows)), 6),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "web_usage_rate": round(
            sum(1 for row in successful if row.get("used_web_search")) / max(1, len(successful)),
            6,
        ),
    }

    selected_items = [
        row.get("retrieval_metrics", {}).get("selected", {})
        for row in successful
        if row.get("retrieval_metrics", {}).get("selected")
    ]
    selected_summary = average_numeric_dicts(selected_items) if selected_items else {}
    positive_selected_items = [
        row.get("retrieval_metrics", {}).get("selected", {})
        for row in successful
        if row.get("retrieval_metrics", {}).get("selected")
        and has_expected_positive(row)
    ]
    positive_selected_summary = average_numeric_dicts(positive_selected_items) if positive_selected_items else {}
    retrieval_keys = (
        f"positive_recall@{top_k}",
        f"positive_mrr@{top_k}",
        f"negative_evidence_rate@{top_k}",
        f"wrong_evidence_rate@{top_k}",
        f"distractor_evidence_rate@{top_k}",
    )
    summary["retrieval"] = {
        retrieval_keys[0]: positive_selected_summary.get(retrieval_keys[0]),
        retrieval_keys[1]: positive_selected_summary.get(retrieval_keys[1]),
        retrieval_keys[2]: selected_summary.get(retrieval_keys[2], 0.0),
        retrieval_keys[3]: selected_summary.get(retrieval_keys[3], 0.0),
        retrieval_keys[4]: selected_summary.get(retrieval_keys[4], 0.0),
    }
    summary["retrieval"].update(citation_source_metrics(successful))
    summary["latency_breakdown_ms"] = average_numeric_dicts([
        row.get("latency_breakdown_ms") or {}
        for row in successful
        if row.get("latency_breakdown_ms")
    ])

    generation_keys = (
        "answer_correctness",
        "faithfulness",
        "citation_quality",
        "refusal_score",
        "counterfactual_score",
    )
    judge_summary = judge_report.get("summary") or {}
    summary["generation"] = {
        key: judge_summary[key]
        for key in generation_keys
        if isinstance(judge_summary.get(key), (int, float))
    }
    for key in ("valid_sample_count", "invalid_sample_count", "evaluation_valid_rate"):
        if isinstance(judge_summary.get(key), (int, float)):
            summary["generation"][key] = judge_summary[key]
    slot_rates = [
        float(row.get("evidence_grade", {}).get("slot_coverage_rate"))
        for row in successful
        if isinstance(row.get("evidence_grade", {}).get("slot_coverage_rate"), (int, float))
    ]
    if slot_rates:
        summary["generation"]["slot_coverage_rate"] = round(sum(slot_rates) / len(slot_rates), 6)
    summary["by_question_type"] = summarize_by_question_type(rows, top_k)
    return summary


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * min(max(quantile, 0.0), 1.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def citation_to_positive_rate(rows: List[Dict[str, Any]]) -> float:
    positive_count = 0
    citation_count = 0
    for row in rows:
        expected_labels = row.get("expected_doc_labels") or {}
        seen = set()
        for citation in row.get("citations") or []:
            doc_id = citation.get("doc_id")
            key = citation.get("evidence_id") or doc_id
            if not key or key in seen:
                continue
            seen.add(key)
            citation_count += 1
            if expected_labels.get(doc_id) == "positive":
                positive_count += 1
    return round(positive_count / max(1, citation_count), 6)


def citation_source_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = {
        "current_positive": 0,
        "cross_sample_supporting": 0,
        "cross_sample_irrelevant": 0,
        "negative": 0,
        "positive_wrong": 0,
        "web_supporting": 0,
    }
    total = 0
    for row in rows:
        labels = row.get("expected_doc_labels") or {}
        judge = row.get("deepseek_judge") or {}
        citation_quality = float(judge.get("citation_quality") or 0.0)
        seen = set()
        for citation in row.get("citations") or []:
            key = citation.get("evidence_id") or citation.get("doc_id")
            if not key or key in seen:
                continue
            seen.add(key)
            total += 1
            if citation.get("source_type") == "web":
                counts["web_supporting"] += 1
                continue
            label = labels.get(citation.get("doc_id")) or "cross_sample"
            if label == "positive":
                counts["current_positive"] += 1
            elif label == "positive_wrong":
                counts["positive_wrong"] += 1
            elif label == "negative":
                counts["negative"] += 1
            elif citation_quality >= 0.8:
                counts["cross_sample_supporting"] += 1
            else:
                counts["cross_sample_irrelevant"] += 1
    supporting = counts["current_positive"] + counts["cross_sample_supporting"] + counts["web_supporting"]
    irrelevant = counts["cross_sample_irrelevant"] + counts["negative"]
    return {
        "citation_support_rate": round(supporting / max(1, total), 6),
        "citation_current_positive_rate": round(counts["current_positive"] / max(1, total), 6),
        "citation_cross_sample_supporting_rate": round(counts["cross_sample_supporting"] / max(1, total), 6),
        "citation_wrong_rate": round(counts["positive_wrong"] / max(1, total), 6),
        "citation_irrelevant_rate": round(irrelevant / max(1, total), 6),
    }


def has_expected_positive(row: Dict[str, Any]) -> bool:
    return any(label == "positive" for label in (row.get("expected_doc_labels") or {}).values())


def summarize_by_question_type(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    question_types = sorted({str(row.get("question_type") or "unknown") for row in rows})
    retrieval_keys = (
        f"positive_recall@{top_k}",
        f"positive_mrr@{top_k}",
        f"negative_evidence_rate@{top_k}",
        f"wrong_evidence_rate@{top_k}",
        f"distractor_evidence_rate@{top_k}",
    )
    generation_keys = ("answer_correctness", "faithfulness", "citation_quality")

    for question_type in question_types:
        type_rows = [row for row in rows if str(row.get("question_type") or "unknown") == question_type]
        successful = [row for row in type_rows if not row.get("error")]
        selected_items = [
            row.get("retrieval_metrics", {}).get("selected", {})
            for row in successful
            if row.get("retrieval_metrics", {}).get("selected")
        ]
        retrieval_summary = average_numeric_dicts(selected_items) if selected_items else {}
        positive_selected_items = [
            row.get("retrieval_metrics", {}).get("selected", {})
            for row in successful
            if row.get("retrieval_metrics", {}).get("selected")
            and has_expected_positive(row)
        ]
        positive_retrieval_summary = average_numeric_dicts(positive_selected_items) if positive_selected_items else {}
        all_judged_items = [
            row.get("deepseek_judge", {})
            for row in successful
            if row.get("deepseek_judge")
        ]
        judged_items = [
            judged for judged in all_judged_items
            if judged.get("evaluation_valid") is not False
        ]
        judged_summary = average_numeric_dicts(judged_items) if judged_items else {}
        invalid_count = len(all_judged_items) - len(judged_items)
        type_summary: Dict[str, Any] = {
            "count": len(type_rows),
            "valid_count": len(judged_items),
            "invalid_count": invalid_count,
            "pass_rate": (
                round(sum(1 for judged in judged_items if judge_passed(judged)) / len(judged_items), 6)
                if judged_items
                else None
            ),
        }
        for key in generation_keys:
            if isinstance(judged_summary.get(key), (int, float)):
                type_summary[key] = judged_summary[key]
        type_summary[retrieval_keys[0]] = positive_retrieval_summary.get(retrieval_keys[0])
        type_summary[retrieval_keys[1]] = positive_retrieval_summary.get(retrieval_keys[1])
        type_summary[retrieval_keys[2]] = retrieval_summary.get(retrieval_keys[2], 0.0)
        type_summary[retrieval_keys[3]] = retrieval_summary.get(retrieval_keys[3], 0.0)
        type_summary[retrieval_keys[4]] = retrieval_summary.get(retrieval_keys[4], 0.0)
        if question_type == "negative_rejection" and isinstance(judged_summary.get("refusal_score"), (int, float)):
            type_summary["refusal_score"] = judged_summary["refusal_score"]
        if question_type == "counterfactual_robustness" and isinstance(judged_summary.get("counterfactual_score"), (int, float)):
            type_summary["counterfactual_score"] = judged_summary["counterfactual_score"]
        slot_rates = [
            float(row.get("evidence_grade", {}).get("slot_coverage_rate"))
            for row in successful
            if isinstance(row.get("evidence_grade", {}).get("slot_coverage_rate"), (int, float))
        ]
        if slot_rates:
            type_summary["slot_coverage_rate"] = round(sum(slot_rates) / len(slot_rates), 6)
        result[question_type] = type_summary
    return result


def judge_passed(judged: Dict[str, Any]) -> bool:
    verdict = str(judged.get("verdict") or "").strip().lower()
    if verdict in {"pass", "passed", "correct", "ok"}:
        return True
    if verdict in {"fail", "failed", "incorrect", "wrong"}:
        return False
    overall = judged.get("overall")
    return isinstance(overall, (int, float)) and float(overall) >= 0.8


def average_numeric_dicts(items: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = sorted({key for item in items for key, value in item.items() if isinstance(value, (int, float))})
    result: Dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
        if values:
            result[key] = round(sum(values) / len(values), 6)
    return result


def print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print("summary")
    print(f"- sample_count: {summary['sample_count']}")
    print(f"- success_rate: {summary['success_rate']:.4f}")
    print(f"- avg_latency_ms: {summary['avg_latency_ms']:.1f}")
    print(f"- p95_latency_ms: {summary['p95_latency_ms']:.1f}")
    retrieval = summary.get("retrieval")
    if retrieval:
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in retrieval.items())
        print(f"- retrieval: {formatted}")
    generation = summary.get("generation")
    if generation:
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in generation.items() if isinstance(value, (int, float)))
        print(f"- generation: {formatted}")
    for question_type, metrics in summary.get("by_question_type", {}).items():
        pass_rate = metrics.get("pass_rate")
        pass_rate_text = f"{pass_rate:.4f}" if isinstance(pass_rate, (int, float)) else "n/a"
        print(
            f"- {question_type}: count={metrics['count']}, "
            f"valid={metrics.get('valid_count', 0)}, invalid={metrics.get('invalid_count', 0)}, "
            f"pass_rate={pass_rate_text}"
        )


def save_report(report: Dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"rgb_e2e_global_pool_eval_kb{report['kb_id']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    asyncio.run(main_async())
