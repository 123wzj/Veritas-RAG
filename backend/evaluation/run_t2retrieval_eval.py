# -*- coding: utf-8 -*-
"""Evaluate the project's real RAG retrieval pipeline on mteb/T2Retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db.mysql.connection import SessionLocal, init_db
from evaluation.t2retrieval import (
    DATASET_REPO_ID,
    DEFAULT_DATASET_DIR,
    DEFAULT_REPORT_DIR,
    download_dataset,
    load_dataset,
    select_queries_for_eval,
)
from models.database.knowledge import DocumentTable
from services.retrieval.hybrid import hybrid_retriever
from services.retrieval.reranker import reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an imported T2Retrieval KB.")
    parser.add_argument("--download", action="store_true", help="Download dataset files before evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--limit-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--max-per-doc", type=int, default=3)
    parser.add_argument("--max-per-parent", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-reranker", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    init_db()

    if args.download:
        download_dataset(args.dataset_dir)

    _, queries, qrels = load_dataset(args.dataset_dir)
    available_doc_ids = load_imported_doc_ids(args.kb_id)
    if not available_doc_ids:
        raise RuntimeError(
            f"No imported documents found for kb_id={args.kb_id}. "
            "Run import_t2retrieval_dataset.py first."
        )

    eval_queries = select_queries_for_eval(
        queries=queries,
        qrels=qrels,
        available_doc_ids=available_doc_ids,
        limit_queries=args.limit_queries,
        seed=args.seed,
    )
    if not eval_queries:
        raise RuntimeError("No usable queries found for this imported KB.")

    print({
        "dataset": DATASET_REPO_ID,
        "kb_id": args.kb_id,
        "imported_docs": len(available_doc_ids),
        "eval_queries": len(eval_queries),
        "top_k": args.top_k,
    })

    start = time.perf_counter()
    per_query: List[Dict[str, Any]] = []
    metric_buckets: Dict[str, List[Dict[str, float]]] = {
        "dense": [],
        "bm25": [],
        "rrf": [],
        "rerank": [],
    }

    for index, (query_id, query) in enumerate(eval_queries, start=1):
        relevant_docs = {
            doc_id: score
            for doc_id, score in qrels.get(query_id, {}).items()
            if doc_id in available_doc_ids
        }
        if not relevant_docs:
            continue

        dense_results, bm25_results = await asyncio.gather(
            hybrid_retriever._dense_retrieve_async(
                query=query,
                user_id=args.user_id,
                kb_id=args.kb_id,
                modality=None,
                limit=args.dense_top_k,
            ),
            hybrid_retriever._sparse_retrieve_async(
                query=query,
                kb_id=args.kb_id,
                modality=None,
                limit=args.bm25_top_k,
            ),
        )
        rrf_results = hybrid_retriever._fuse_multi_query_results(
            dense_groups=[dense_results],
            sparse_groups=[bm25_results],
        )
        rrf_results = hybrid_retriever._parent_backfill_sync(
            rrf_results[: max(args.rerank_candidates, args.top_k)],
            args.kb_id,
        )

        result_sets = {
            "dense": [item.get("doc_id") for item in dense_results],
            "bm25": [item.get("doc_id") for item in bm25_results],
            "rrf": [item.get("doc_id") for item in rrf_results],
        }
        if not args.skip_reranker:
            reranked = reranker.rerank(
                query=query,
                documents=rrf_results[: args.rerank_candidates],
                top_k=args.top_k,
                max_per_doc=args.max_per_doc,
                max_per_parent=args.max_per_parent,
            )
            result_sets["rerank"] = [item.get("doc_id") for item in reranked]

        query_report = {
            "query_id": query_id,
            "query": query,
            "gold_doc_ids": list(relevant_docs),
            "results": {},
        }
        for name, ranked_ids in result_sets.items():
            ranked_ids = [str(doc_id) for doc_id in ranked_ids if doc_id]
            metrics = compute_metrics(ranked_ids, relevant_docs, top_k=args.top_k)
            metric_buckets.setdefault(name, []).append(metrics)
            query_report["results"][name] = {
                "top_doc_ids": ranked_ids[: args.top_k],
                "metrics": metrics,
            }

        per_query.append(query_report)
        print(f"[{index}/{len(eval_queries)}] {query_id}")

    report = {
        "dataset": DATASET_REPO_ID,
        "kb_id": args.kb_id,
        "config": {
            "limit_queries": args.limit_queries,
            "top_k": args.top_k,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "rerank_candidates": args.rerank_candidates,
            "max_per_doc": args.max_per_doc,
            "max_per_parent": args.max_per_parent,
            "skip_reranker": args.skip_reranker,
            "seed": args.seed,
        },
        "seconds": round(time.perf_counter() - start, 3),
        "summary": {
            name: average_metrics(values)
            for name, values in metric_buckets.items()
            if values
        },
        "per_query": per_query,
    }
    print_summary(report)
    output_path = save_report(report, args.report_dir)
    print(f"report={output_path}")


def load_imported_doc_ids(kb_id: int) -> set[str]:
    db = SessionLocal()
    try:
        return {
            row.doc_id
            for row in db.query(DocumentTable.doc_id)
            .filter(DocumentTable.kb_id == kb_id)
            .all()
        }
    finally:
        db.close()


def compute_metrics(
    ranked_doc_ids: List[str],
    relevant_docs: Dict[str, float],
    top_k: int,
) -> Dict[str, float]:
    ranked_top = ranked_doc_ids[:top_k]
    relevant_set = set(relevant_docs)
    hits = [doc_id for doc_id in ranked_top if doc_id in relevant_set]

    recall = len(set(hits)) / max(len(relevant_set), 1)
    hit = 1.0 if hits else 0.0
    mrr = 0.0
    for rank, doc_id in enumerate(ranked_top, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    seen = set()
    for rank, doc_id in enumerate(ranked_top, start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rel = relevant_docs.get(doc_id, 0.0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1)

    ideal_rels = sorted(relevant_docs.values(), reverse=True)[:top_k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {
        f"hit@{top_k}": hit,
        f"recall@{top_k}": recall,
        f"mrr@{top_k}": mrr,
        f"ndcg@{top_k}": ndcg,
    }


def average_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {
        key: round(sum(item.get(key, 0.0) for item in items) / len(items), 6)
        for key in keys
    }


def print_summary(report: Dict[str, Any]) -> None:
    print("\nsummary")
    for name, metrics in report["summary"].items():
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"- {name}: {formatted}")


def save_report(report: Dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"t2retrieval_eval_kb{report['kb_id']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    asyncio.run(main_async())
