# -*- coding: utf-8 -*-
"""Audit an imported RGB evaluation KB across MySQL and Chroma."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal
from backend.evaluation.import_rgb_global_pool import build_docs, load_samples
from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR
from backend.models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBaseTable


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rgb"
    / "e2e_global_pool"
    / "rgb_zh_e2e_kb_pool.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB evaluation KB integrity.")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def main() -> None:
    args = parse_args()
    expected_docs = build_docs(load_samples(args.input, None))
    expected_doc_ids = {doc.doc_id for doc in expected_docs}

    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBaseTable).filter(KnowledgeBaseTable.id == args.kb_id).first()
        if kb is None:
            raise RuntimeError(f"Knowledge base {args.kb_id} does not exist.")
        documents = db.query(DocumentTable).filter(DocumentTable.kb_id == args.kb_id).all()
        chunks = db.query(ChunkTable).filter(ChunkTable.kb_id == args.kb_id).all()
    finally:
        db.close()

    document_ids = {doc.doc_id for doc in documents}
    parent_chunks = [chunk for chunk in chunks if chunk.is_parent]
    child_chunks = [chunk for chunk in chunks if not chunk.is_parent]
    parent_ids = {chunk.chunk_id for chunk in parent_chunks}
    child_ids = {chunk.chunk_id for chunk in child_chunks}
    parent_counts = Counter(chunk.doc_id for chunk in parent_chunks)
    child_counts = Counter(chunk.doc_id for chunk in child_chunks)

    vector_result = chroma_client.get_collection().get(
        where={"kb_id": args.kb_id},
        include=["metadatas"],
    )
    vector_ids = set(vector_result.get("ids") or [])
    vector_metadatas: List[Dict[str, Any]] = vector_result.get("metadatas") or []

    missing_expected_docs = sorted(expected_doc_ids - document_ids)
    unexpected_docs = sorted(document_ids - expected_doc_ids)
    documents_without_parent = sorted(doc_id for doc_id in document_ids if parent_counts[doc_id] == 0)
    documents_without_child = sorted(doc_id for doc_id in document_ids if child_counts[doc_id] == 0)
    invalid_parent_links = sorted(
        chunk.chunk_id
        for chunk in child_chunks
        if not chunk.parent_id or chunk.parent_id not in parent_ids
    )
    chunks_without_document = sorted(chunk.chunk_id for chunk in chunks if chunk.doc_id not in document_ids)
    missing_vectors = sorted(child_ids - vector_ids)
    orphan_vectors = sorted(vector_ids - child_ids)
    invalid_vector_metadata = sum(
        1
        for metadata in vector_metadatas
        if int(metadata.get("kb_id", -1)) != args.kb_id
        or not metadata.get("doc_id")
        or metadata.get("doc_id") not in document_ids
    )
    incomplete_documents = sorted(
        doc.doc_id
        for doc in documents
        if enum_value(doc.status).lower() != "completed"
    )

    type_counts = Counter()
    label_counts = Counter()
    for doc in documents:
        metadata = doc.doc_metadata or {}
        type_counts[str(metadata.get("question_type") or "unknown")] += 1
        label_counts[str(metadata.get("evidence_label") or "unknown")] += 1

    failures = {
        "missing_expected_docs": len(missing_expected_docs),
        "unexpected_docs": len(unexpected_docs),
        "documents_without_parent": len(documents_without_parent),
        "documents_without_child": len(documents_without_child),
        "invalid_parent_links": len(invalid_parent_links),
        "chunks_without_document": len(chunks_without_document),
        "missing_vectors": len(missing_vectors),
        "orphan_vectors": len(orphan_vectors),
        "invalid_vector_metadata": invalid_vector_metadata,
        "incomplete_documents": len(incomplete_documents),
    }
    passed = all(value == 0 for value in failures.values())
    report = {
        "dataset": "RGB",
        "mode": "kb_integrity_audit",
        "kb_id": args.kb_id,
        "kb_name": kb.name,
        "input": str(args.input),
        "passed": passed,
        "counts": {
            "expected_documents": len(expected_doc_ids),
            "mysql_documents": len(documents),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "chroma_vectors": len(vector_ids),
            "question_types": dict(sorted(type_counts.items())),
            "evidence_labels": dict(sorted(label_counts.items())),
        },
        "failures": failures,
        "details": {
            "missing_expected_docs": missing_expected_docs[:50],
            "unexpected_docs": unexpected_docs[:50],
            "documents_without_parent": documents_without_parent[:50],
            "documents_without_child": documents_without_child[:50],
            "invalid_parent_links": invalid_parent_links[:50],
            "chunks_without_document": chunks_without_document[:50],
            "missing_vectors": missing_vectors[:50],
            "orphan_vectors": orphan_vectors[:50],
            "incomplete_documents": incomplete_documents[:50],
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / f"rgb_kb_integrity_audit_kb{args.kb_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "passed": passed,
        "counts": report["counts"],
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
