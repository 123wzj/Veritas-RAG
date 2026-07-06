# -*- coding: utf-8 -*-
"""Import mteb/T2Retrieval into the project's real RAG stores for testing."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal, init_db
from backend.embeddings.embeddings import get_embeddings
from backend.embeddings.sparse import get_sparse_embedding
from backend.evaluation.t2retrieval import (
    DATASET_REPO_ID,
    DEFAULT_DATASET_DIR,
    CorpusDoc,
    download_dataset,
    load_dataset,
    select_docs_for_eval,
)
from backend.models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBasePermissionTable, KnowledgeBaseTable
from backend.models.schemas.knowledge import DocumentStatus, ModalityType
from backend.services.ingestion.chunker import document_chunker
from backend.services.retrieval.hybrid import hybrid_retriever


DEFAULT_KB_NAME = "public_eval_t2retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import T2Retrieval into the real RAG database.")
    parser.add_argument("--download", action="store_true", help="Download dataset files before import.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-name", type=str, default=DEFAULT_KB_NAME)
    parser.add_argument("--limit-corpus", type=int, default=1000)
    parser.add_argument("--limit-gold-queries", type=int, default=50)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-existing", action="store_true", help="Delete an existing eval KB with the same name first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    if args.download:
        download_dataset(args.dataset_dir)

    corpus, queries, qrels = load_dataset(args.dataset_dir)
    selected_docs = select_docs_for_eval(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        limit_corpus=args.limit_corpus,
        limit_queries=args.limit_gold_queries,
        seed=args.seed,
    )

    db = SessionLocal()
    try:
        existing_kb = (
            db.query(KnowledgeBaseTable)
            .filter(
                KnowledgeBaseTable.user_id == args.user_id,
                KnowledgeBaseTable.name == args.kb_name,
            )
            .first()
        )
        if existing_kb and args.clean_existing:
            delete_eval_kb(db, existing_kb.id)
            existing_kb = None

        kb = existing_kb or KnowledgeBaseTable(
            user_id=args.user_id,
            name=args.kb_name,
            description=f"Public evaluation KB imported from {DATASET_REPO_ID}.",
            acl_tags={"evaluation": True, "dataset": DATASET_REPO_ID},
            visibility="private",
        )
        if not existing_kb:
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

        start = time.perf_counter()
        imported = import_docs(
            db=db,
            kb_id=kb.id,
            user_id=args.user_id,
            docs=selected_docs,
            embed_batch_size=args.embed_batch_size,
        )
        hybrid_retriever.invalidate_sparse_cache(kb_id=kb.id)
        elapsed = round(time.perf_counter() - start, 3)

        print({
            "dataset": DATASET_REPO_ID,
            "kb_id": kb.id,
            "kb_name": kb.name,
            "selected_docs": len(selected_docs),
            "imported_or_updated_docs": imported["docs"],
            "parent_chunks": imported["parents"],
            "child_chunks": imported["children"],
            "seconds": elapsed,
        })
    finally:
        db.close()


def import_docs(
    db,
    kb_id: int,
    user_id: int,
    docs: Dict[str, CorpusDoc],
    embed_batch_size: int,
) -> Dict[str, int]:
    embeddings = get_embeddings()
    sparse_embedding = get_sparse_embedding()
    collection = chroma_client.get_collection()

    doc_count = 0
    parent_count = 0
    child_count = 0
    pending_children: List[ChunkTable] = []

    for doc in docs.values():
        parent_chunks, child_chunks = build_chunks(doc)
        child_texts = [chunk.content or "" for chunk in child_chunks]
        sparse_vectors = sparse_embedding.embed_documents(child_texts)

        upsert_document(db, kb_id=kb_id, doc=doc, parent_count=len(parent_chunks), child_count=len(child_chunks))
        sparse_by_chunk_id = {
            chunk.chunk_id: sparse_vectors[index]
            for index, chunk in enumerate(child_chunks)
            if index < len(sparse_vectors)
        }
        for chunk in parent_chunks + child_chunks:
            row = upsert_chunk(db, kb_id=kb_id, chunk=chunk)
            if not row.is_parent:
                row.sparse_vector = sparse_by_chunk_id.get(row.chunk_id, {})
                pending_children.append(row)
        db.commit()

        parent_count += len(parent_chunks)
        child_count += len(child_chunks)
        doc_count += 1

        if len(pending_children) >= embed_batch_size:
            upsert_child_vectors(collection, embeddings, user_id, kb_id, pending_children)
            pending_children.clear()

    if pending_children:
        upsert_child_vectors(collection, embeddings, user_id, kb_id, pending_children)

    return {"docs": doc_count, "parents": parent_count, "children": child_count}


def build_chunks(doc: CorpusDoc):
    parsed = {
        "type": "txt",
        "content": doc.text,
        "metadata": {"title": doc.title or doc.doc_id},
    }
    return document_chunker.chunk_document(parsed, doc.doc_id)


def upsert_document(db, kb_id: int, doc: CorpusDoc, parent_count: int, child_count: int) -> None:
    file_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    row = db.query(DocumentTable).filter(DocumentTable.doc_id == doc.doc_id).first()
    if row is None:
        row = DocumentTable(
            kb_id=kb_id,
            doc_id=doc.doc_id,
            filename=f"{doc.doc_id}.txt",
            file_path=f"public://{DATASET_REPO_ID}/{doc.doc_id}",
            file_hash=file_hash,
            file_size=len(doc.text.encode("utf-8")),
            status=DocumentStatus.COMPLETED,
            modality=ModalityType.TEXT,
            language="zh",
            total_chunks=child_count,
            doc_metadata={
                "dataset": DATASET_REPO_ID,
                "eval": True,
                "parent_chunks": parent_count,
                "title": doc.title,
            },
        )
        db.add(row)
        return

    row.kb_id = kb_id
    row.filename = f"{doc.doc_id}.txt"
    row.file_path = f"public://{DATASET_REPO_ID}/{doc.doc_id}"
    row.file_hash = file_hash
    row.file_size = len(doc.text.encode("utf-8"))
    row.status = DocumentStatus.COMPLETED
    row.total_chunks = child_count
    row.doc_metadata = {
        "dataset": DATASET_REPO_ID,
        "eval": True,
        "parent_chunks": parent_count,
        "title": doc.title,
    }


def upsert_chunk(db, kb_id: int, chunk) -> ChunkTable:
    row = db.query(ChunkTable).filter(ChunkTable.chunk_id == chunk.chunk_id).first()
    if row is None:
        row = ChunkTable(kb_id=kb_id, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)
        db.add(row)

    row.kb_id = kb_id
    row.doc_id = chunk.doc_id
    row.parent_id = chunk.parent_id
    row.is_parent = chunk.parent_id is None
    row.modality = chunk.modality
    row.language = chunk.language
    row.title = chunk.title
    row.section_path = chunk.section_path
    row.page_no = chunk.page_no
    row.content = chunk.content
    row.token_count = chunk.token_count
    row.source_uri = f"public://{DATASET_REPO_ID}/{chunk.doc_id}"
    return row


def upsert_child_vectors(collection, embeddings, user_id: int, kb_id: int, child_rows: List[ChunkTable]) -> None:
    texts = [row.content or "" for row in child_rows]
    vectors = embeddings.embed_documents(texts)
    metadatas = []
    for row in child_rows:
        sparse_vector = row.sparse_vector or {}
        lexical_terms = [token for token in sparse_vector.keys() if not token.startswith("__")]
        metadatas.append({
            "user_id": user_id,
            "kb_id": int(kb_id),
            "doc_id": row.doc_id,
            "parent_id": row.parent_id or "",
            "chunk_id": row.chunk_id,
            "is_parent": False,
            "modality": row.modality.value if hasattr(row.modality, "value") else str(row.modality),
            "language": row.language or "zh",
            "title": row.title or "",
            "section_path": row.section_path or "",
            "page_no": row.page_no or 0,
            "token_count": row.token_count or 0,
            "lexical_terms": " ".join(lexical_terms[:24]),
        })

    collection.upsert(
        ids=[row.chunk_id for row in child_rows],
        embeddings=vectors,
        metadatas=metadatas,
    )


def delete_eval_kb(db, kb_id: int) -> None:
    chunk_ids = [
        row.chunk_id
        for row in db.query(ChunkTable.chunk_id).filter(ChunkTable.kb_id == kb_id).all()
    ]
    if chunk_ids:
        collection = chroma_client.get_collection()
        for start in range(0, len(chunk_ids), 500):
            collection.delete(ids=chunk_ids[start : start + 500])

    db.query(ChunkTable).filter(ChunkTable.kb_id == kb_id).delete(synchronize_session=False)
    db.query(DocumentTable).filter(DocumentTable.kb_id == kb_id).delete(synchronize_session=False)
    db.query(KnowledgeBasePermissionTable).filter(KnowledgeBasePermissionTable.kb_id == kb_id).delete(synchronize_session=False)
    db.query(KnowledgeBaseTable).filter(KnowledgeBaseTable.id == kb_id).delete(synchronize_session=False)
    db.commit()


if __name__ == "__main__":
    main()
