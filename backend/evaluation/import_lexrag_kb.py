# -*- coding: utf-8 -*-
"""Import the complete LexRAG law library into an isolated evaluation KB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal, init_db
from backend.embeddings.embeddings import get_embeddings
from backend.embeddings.sparse import get_sparse_embedding
from backend.evaluation.import_rgb_global_pool import delete_eval_kb
from backend.models.database.knowledge import (
    ChunkTable,
    DocumentTable,
    KnowledgeBasePermissionTable,
    KnowledgeBaseTable,
)
from backend.models.schemas.knowledge import DocumentStatus, ModalityType
from backend.services.ingestion.chunker import document_chunker
from backend.services.retrieval.hybrid import hybrid_retriever


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "evaluation" / "lexrag" / "prepared" / "lexrag_law_library.jsonl"
)
DEFAULT_KB_NAME = "public_eval_lexrag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the LexRAG law library.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-name", default=DEFAULT_KB_NAME)
    parser.add_argument("--clean-existing", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    documents = load_jsonl(args.input)
    db = SessionLocal()
    try:
        kb = (
            db.query(KnowledgeBaseTable)
            .filter(KnowledgeBaseTable.user_id == args.user_id, KnowledgeBaseTable.name == args.kb_name)
            .first()
        )
        if kb and args.clean_existing:
            delete_eval_kb(db, kb.id)
            kb = None
        if kb is None:
            kb = KnowledgeBaseTable(
                user_id=args.user_id,
                name=args.kb_name,
                description="LexRAG complete law library for Chinese multi-turn RAG evaluation.",
                acl_tags={"evaluation": True, "dataset": "LexRAG", "task": "multiturn_rag"},
                visibility="private",
            )
            db.add(kb)
            db.commit()
            db.refresh(kb)
            db.add(KnowledgeBasePermissionTable(
                kb_id=kb.id,
                user_id=args.user_id,
                permission_type="owner",
                granted_by=args.user_id,
            ))
            db.commit()
        elif db.query(DocumentTable).filter(DocumentTable.kb_id == kb.id).count():
            raise RuntimeError("LexRAG KB already contains documents; use --clean-existing.")

        started = time.perf_counter()
        counts = import_documents(
            db,
            kb_id=kb.id,
            user_id=args.user_id,
            documents=documents,
            embed_batch_size=args.embed_batch_size,
        )
        hybrid_retriever.invalidate_sparse_cache(kb_id=kb.id)
        print(json.dumps({
            "dataset": "LexRAG",
            "kb_id": kb.id,
            "kb_name": kb.name,
            "source_documents": len(documents),
            **counts,
            "seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False, indent=2))
    finally:
        db.close()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def project_doc_id(source_doc_id: str) -> str:
    digest = hashlib.sha256(source_doc_id.encode("utf-8")).hexdigest()[:40]
    return f"lex_{digest}"


def import_documents(
    db,
    *,
    kb_id: int,
    user_id: int,
    documents: List[Dict[str, Any]],
    embed_batch_size: int,
) -> Dict[str, int]:
    embeddings = get_embeddings()
    sparse = get_sparse_embedding()
    collection = chroma_client.get_collection()
    pending_documents: List[DocumentTable] = []
    pending_chunks: List[ChunkTable] = []
    pending_children: List[ChunkTable] = []
    parent_count = 0
    child_count = 0
    imported_count = 0

    def flush() -> None:
        nonlocal pending_documents, pending_chunks, pending_children
        if not pending_children:
            return
        texts = [row.content or "" for row in pending_children]
        sparse_vectors = sparse.embed_documents(texts)
        dense_vectors = embeddings.embed_documents(texts)
        for row, vector in zip(pending_children, sparse_vectors):
            row.sparse_vector = vector
        db.add_all(pending_documents)
        db.add_all(pending_chunks)
        db.commit()
        collection.upsert(
            ids=[row.chunk_id for row in pending_children],
            embeddings=dense_vectors,
            metadatas=[vector_metadata(row, user_id, kb_id) for row in pending_children],
        )
        pending_documents = []
        pending_chunks = []
        pending_children = []

    for index, source in enumerate(documents, start=1):
        doc_id = project_doc_id(source["doc_id"])
        text = f"{source['title']}\n{source.get('text') or ''}".strip()
        parent_chunks, child_chunks = document_chunker.chunk_document({
            "type": "txt",
            "content": text,
            "metadata": {"title": source["title"]},
        }, doc_id)
        if not child_chunks:
            continue
        pending_documents.append(DocumentTable(
            kb_id=kb_id,
            doc_id=doc_id,
            filename=f"{doc_id}.txt",
            file_path=f"lexrag://law/{source.get('law_id')}",
            file_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            file_size=len(text.encode("utf-8")),
            status=DocumentStatus.COMPLETED,
            modality=ModalityType.TEXT,
            language="zh",
            total_chunks=len(child_chunks),
            doc_metadata={
                "dataset": "LexRAG",
                "evaluation": True,
                "source_doc_id": source["doc_id"],
                "law_id": source.get("law_id"),
                "title": source["title"],
            },
        ))
        for chunk in parent_chunks + child_chunks:
            row = ChunkTable(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                is_parent=chunk.parent_id is None,
                modality=chunk.modality,
                language="zh",
                title=source["title"],
                section_path=chunk.section_path,
                page_no=chunk.page_no,
                content=chunk.content,
                token_count=chunk.token_count,
                source_uri=f"lexrag://law/{source.get('law_id')}",
            )
            pending_chunks.append(row)
            if not row.is_parent:
                pending_children.append(row)
        parent_count += len(parent_chunks)
        child_count += len(child_chunks)
        imported_count += 1
        if len(pending_children) >= embed_batch_size:
            flush()
        if index % 1000 == 0:
            print(f"prepared_documents={index}/{len(documents)}")
    flush()
    return {
        "imported_documents": imported_count,
        "parent_chunks": parent_count,
        "child_chunks": child_count,
    }


def vector_metadata(row: ChunkTable, user_id: int, kb_id: int) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "kb_id": kb_id,
        "doc_id": row.doc_id,
        "parent_id": row.parent_id or "",
        "chunk_id": row.chunk_id,
        "is_parent": False,
        "modality": "text",
        "language": "zh",
        "title": row.title or "",
        "section_path": row.section_path or "",
        "page_no": row.page_no or 0,
        "token_count": row.token_count or 0,
        "lexical_terms": " ".join(
            key for key in (row.sparse_vector or {}).keys() if not key.startswith("__")
        )[:1000],
    }


if __name__ == "__main__":
    main()
