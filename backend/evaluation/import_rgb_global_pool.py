# -*- coding: utf-8 -*-
"""Import RGB fixed samples into one global-pool evaluation KB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal, init_db
from backend.embeddings.embeddings import get_embeddings
from backend.embeddings.sparse import get_sparse_embedding
from backend.models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBasePermissionTable, KnowledgeBaseTable
from backend.models.schemas.knowledge import DocumentStatus, ModalityType
from backend.services.ingestion.chunker import document_chunker
from backend.services.retrieval.hybrid import hybrid_retriever


DATASET_ID = "RGB"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "rgb" / "rgb_zh_fixed_samples.jsonl"
DEFAULT_KB_NAME = "public_eval_rgb_global_pool"


@dataclass
class RGBDoc:
    doc_id: str
    sample_id: str
    question: str
    question_type: str
    evidence_label: str
    context_index: int
    text: str
    title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import RGB fixed-context samples into a global-pool KB.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-name", type=str, default=DEFAULT_KB_NAME)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="Comma-separated RGB question types to import.",
    )
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--clean-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    question_types = {
        item.strip()
        for item in (args.question_types or "").split(",")
        if item.strip()
    }
    samples = load_samples(args.input, args.limit_samples, question_types)
    docs = build_docs(samples)

    db = SessionLocal()
    try:
        existing_kb = (
            db.query(KnowledgeBaseTable)
            .filter(KnowledgeBaseTable.user_id == args.user_id, KnowledgeBaseTable.name == args.kb_name)
            .first()
        )
        if existing_kb and args.clean_existing:
            delete_eval_kb(db, existing_kb.id)
            existing_kb = None

        kb = existing_kb or KnowledgeBaseTable(
            user_id=args.user_id,
            name=args.kb_name,
            description=f"RGB global-pool E2E evaluation KB imported from {args.input}.",
            acl_tags={"evaluation": True, "dataset": DATASET_ID, "pool": "global", "task": "rag_e2e"},
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

        started = time.perf_counter()
        imported = import_docs(
            db=db,
            kb_id=kb.id,
            user_id=args.user_id,
            docs=docs,
            embed_batch_size=args.embed_batch_size,
        )
        hybrid_retriever.invalidate_sparse_cache(kb_id=kb.id)
        print(json.dumps({
            "dataset": DATASET_ID,
            "input": str(args.input),
            "kb_id": kb.id,
            "kb_name": kb.name,
            "samples": len(samples),
            "question_types": sorted(question_types),
            "evidence_docs": len(docs),
            "imported_or_updated_docs": imported["docs"],
            "parent_chunks": imported["parents"],
            "child_chunks": imported["children"],
            "seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False, indent=2))
    finally:
        db.close()


def load_samples(
    path: Path,
    limit: int | None,
    question_types: set[str] | None = None,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if question_types:
        rows = [
            row for row in rows
            if str(row.get("question_type") or "unknown") in question_types
        ]
    return rows[:limit] if limit is not None else rows


def build_docs(samples: List[Dict[str, Any]]) -> List[RGBDoc]:
    docs: List[RGBDoc] = []
    for sample in samples:
        contexts = sample.get("contexts") or []
        labels = sample.get("context_labels") or infer_context_labels(sample)
        for index, text in enumerate(contexts):
            label = labels[index] if index < len(labels) else "unknown"
            doc_id = rgb_doc_id(sample["id"], index, text)
            docs.append(RGBDoc(
                doc_id=doc_id,
                sample_id=sample["id"],
                question=sample.get("question") or "",
                question_type=sample.get("question_type") or "",
                evidence_label=label,
                context_index=index,
                text=text,
                title=f"RGB {sample.get('question_type')} {sample['id']} {label} #{index + 1}",
            ))
    return docs


def infer_context_labels(sample: Dict[str, Any]) -> List[str]:
    contexts = sample.get("contexts") or []
    positive_count = int(sample.get("positive_context_count") or 0)
    if sample.get("question_type") == "counterfactual_robustness":
        wrong_count = max(0, len(contexts) - positive_count)
        labels = ["positive_wrong"] * wrong_count
        labels.extend(["positive"] * positive_count)
        return labels[:len(contexts)]
    labels = ["positive"] * positive_count
    labels.extend(["negative"] * max(0, len(contexts) - positive_count))
    return labels[:len(contexts)]


def rgb_doc_id(sample_id: str, context_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{sample_id}:{context_index}:{text}".encode("utf-8")).hexdigest()[:20]
    return f"rgbgp_{digest}"


def import_docs(db, kb_id: int, user_id: int, docs: List[RGBDoc], embed_batch_size: int) -> Dict[str, int]:
    embeddings = get_embeddings()
    sparse_embedding = get_sparse_embedding()
    collection = chroma_client.get_collection()

    doc_count = 0
    parent_count = 0
    child_count = 0
    pending_children: List[ChunkTable] = []

    for doc in docs:
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
            row = upsert_chunk(db, kb_id=kb_id, chunk=chunk, doc=doc)
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


def build_chunks(doc: RGBDoc):
    parsed = {
        "type": "txt",
        "content": doc.text,
        "metadata": {"title": doc.title},
    }
    parent_chunks, child_chunks = document_chunker.chunk_document(parsed, doc.doc_id)
    if not child_chunks:
        for parent in parent_chunks:
            document_chunker._create_chunk_from_text(
                parent.content,
                doc.doc_id,
                parent.chunk_id,
                parent.title,
                parent.section_path,
                parent.page_no,
                child_chunks,
                is_parent=False,
            )
    return parent_chunks, child_chunks


def upsert_document(db, kb_id: int, doc: RGBDoc, parent_count: int, child_count: int) -> None:
    file_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    metadata = {
        "dataset": DATASET_ID,
        "eval": True,
        "task": "rag_e2e",
        "pool": "global",
        "sample_id": doc.sample_id,
        "question_type": doc.question_type,
        "evidence_label": doc.evidence_label,
        "context_index": doc.context_index,
        "parent_chunks": parent_count,
        "title": doc.title,
    }
    row = db.query(DocumentTable).filter(DocumentTable.doc_id == doc.doc_id).first()
    if row is None:
        db.add(DocumentTable(
            kb_id=kb_id,
            doc_id=doc.doc_id,
            filename=f"{doc.doc_id}.txt",
            file_path=f"rgb://global/{doc.sample_id}/{doc.context_index}",
            file_hash=file_hash,
            file_size=len(doc.text.encode("utf-8")),
            status=DocumentStatus.COMPLETED,
            modality=ModalityType.TEXT,
            language="zh",
            total_chunks=child_count,
            doc_metadata=metadata,
        ))
        return

    row.kb_id = kb_id
    row.filename = f"{doc.doc_id}.txt"
    row.file_path = f"rgb://global/{doc.sample_id}/{doc.context_index}"
    row.file_hash = file_hash
    row.file_size = len(doc.text.encode("utf-8"))
    row.status = DocumentStatus.COMPLETED
    row.modality = ModalityType.TEXT
    row.language = "zh"
    row.total_chunks = child_count
    row.doc_metadata = metadata


def upsert_chunk(db, kb_id: int, chunk, doc: RGBDoc) -> ChunkTable:
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
    row.title = chunk.title or doc.title
    row.section_path = chunk.section_path
    row.page_no = chunk.page_no
    row.content = chunk.content
    row.token_count = chunk.token_count
    row.source_uri = f"rgb://global/{doc.sample_id}/{doc.context_index}/{doc.evidence_label}"
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

    collection.upsert(ids=[row.chunk_id for row in child_rows], embeddings=vectors, metadatas=metadatas)


def delete_eval_kb(db, kb_id: int) -> None:
    chunk_ids = [row.chunk_id for row in db.query(ChunkTable.chunk_id).filter(ChunkTable.kb_id == kb_id).all()]
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
