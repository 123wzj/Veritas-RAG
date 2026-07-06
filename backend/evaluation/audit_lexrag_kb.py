# -*- coding: utf-8 -*-
"""Audit LexRAG source labels, MySQL rows, and Chroma vectors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal
from backend.evaluation.import_lexrag_kb import DEFAULT_INPUT, load_jsonl, project_doc_id
from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR
from backend.models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBaseTable


DEFAULT_CONVERSATIONS = (
    PROJECT_ROOT / "data" / "evaluation" / "lexrag" / "prepared"
    / "lexrag_conversations_80x5.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LexRAG KB integrity.")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--documents", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def main() -> None:
    args = parse_args()
    source_documents = load_jsonl(args.documents)
    conversations = load_jsonl(args.conversations)
    source_ids = {row["doc_id"] for row in source_documents}
    expected_ids = {project_doc_id(doc_id) for doc_id in source_ids}
    gold_source_ids = {
        doc_id
        for conversation in conversations
        for turn in conversation.get("turns") or []
        for doc_id in turn.get("gold_doc_ids") or []
    }

    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBaseTable).filter(KnowledgeBaseTable.id == args.kb_id).first()
        if kb is None:
            raise RuntimeError(f"Knowledge base {args.kb_id} does not exist.")
        documents = db.query(DocumentTable).filter(DocumentTable.kb_id == args.kb_id).all()
        chunks = db.query(ChunkTable).filter(ChunkTable.kb_id == args.kb_id).all()
    finally:
        db.close()

    document_ids = {row.doc_id for row in documents}
    parent_chunks = [row for row in chunks if row.is_parent]
    child_chunks = [row for row in chunks if not row.is_parent]
    parent_ids = {row.chunk_id for row in parent_chunks}
    child_ids = {row.chunk_id for row in child_chunks}
    parent_counts = Counter(row.doc_id for row in parent_chunks)
    child_counts = Counter(row.doc_id for row in child_chunks)
    vectors = chroma_client.get_collection().get(
        where={"kb_id": args.kb_id},
        include=["metadatas"],
    )
    vector_ids = set(vectors.get("ids") or [])
    vector_metadata: List[Dict[str, Any]] = vectors.get("metadatas") or []

    details = {
        "mapping_collisions": len(source_ids) - len(expected_ids),
        "missing_expected_documents": sorted(expected_ids - document_ids),
        "unexpected_documents": sorted(document_ids - expected_ids),
        "missing_gold_documents": sorted(
            project_doc_id(doc_id) for doc_id in gold_source_ids if project_doc_id(doc_id) not in document_ids
        ),
        "documents_without_parent": sorted(doc_id for doc_id in document_ids if not parent_counts[doc_id]),
        "documents_without_child": sorted(doc_id for doc_id in document_ids if not child_counts[doc_id]),
        "invalid_parent_links": sorted(
            row.chunk_id for row in child_chunks if not row.parent_id or row.parent_id not in parent_ids
        ),
        "chunks_without_document": sorted(row.chunk_id for row in chunks if row.doc_id not in document_ids),
        "missing_vectors": sorted(child_ids - vector_ids),
        "orphan_vectors": sorted(vector_ids - child_ids),
        "incomplete_documents": sorted(
            row.doc_id for row in documents if enum_value(row.status).lower() != "completed"
        ),
    }
    invalid_vector_metadata = sum(
        1
        for metadata in vector_metadata
        if int(metadata.get("kb_id", -1)) != args.kb_id
        or metadata.get("doc_id") not in document_ids
    )
    failures = {
        key: value if isinstance(value, int) else len(value)
        for key, value in details.items()
    }
    failures["invalid_vector_metadata"] = invalid_vector_metadata
    passed = all(value == 0 for value in failures.values())
    report = {
        "dataset": "LexRAG",
        "mode": "kb_integrity_audit",
        "kb_id": args.kb_id,
        "kb_name": kb.name,
        "passed": passed,
        "counts": {
            "source_documents": len(source_ids),
            "mapped_document_ids": len(expected_ids),
            "gold_source_documents": len(gold_source_ids),
            "mysql_documents": len(documents),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "chroma_vectors": len(vector_ids),
        },
        "failures": failures,
        "details": {
            key: value[:50] if isinstance(value, list) else value
            for key, value in details.items()
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / (
        f"lexrag_kb_integrity_audit_kb{args.kb_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
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
