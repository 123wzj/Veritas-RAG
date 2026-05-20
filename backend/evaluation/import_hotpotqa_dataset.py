# -*- coding: utf-8 -*-
"""Import HotpotQA into the project's real RAG stores for generation evaluation.

HotpotQA provides question, answer, context paragraphs, and supporting facts.
This makes it more suitable than pure retrieval qrels for end-to-end answer
generation evaluation with RAGAS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from db.chroma.connection import chroma_client
from db.mysql.connection import SessionLocal, init_db
from embeddings.embeddings import get_embeddings
from embeddings.sparse import get_sparse_embedding
from models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBasePermissionTable, KnowledgeBaseTable
from models.schemas.knowledge import DocumentStatus, ModalityType
from services.ingestion.chunker import document_chunker
from services.retrieval.hybrid import hybrid_retriever


DATASET_REPO_ID = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG_NAME = "distractor"
DEFAULT_SPLIT = "validation"
DEFAULT_KB_NAME = "public_eval_hotpotqa_generation"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "evaluation" / "hotpotqa_ragas_samples.jsonl"


@dataclass
class HotpotDoc:
    doc_id: str
    title: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import HotpotQA for RAGAS answer-generation evaluation.")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--kb-name", type=str, default=DEFAULT_KB_NAME)
    parser.add_argument("--dataset", type=str, default=DATASET_REPO_ID)
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--limit-examples", type=int, default=20)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--clean-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    examples = load_hotpotqa_examples(
        dataset=args.dataset,
        config_name=args.config_name,
        split=args.split,
        limit_examples=args.limit_examples,
    )
    docs = build_docs(examples)

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
            description=f"HotpotQA generation evaluation KB imported from {args.dataset}.",
            acl_tags={"evaluation": True, "dataset": args.dataset, "task": "answer_generation"},
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
            docs=list(docs.values()),
            embed_batch_size=args.embed_batch_size,
            dataset=args.dataset,
        )
        hybrid_retriever.invalidate_sparse_cache(kb_id=kb.id)
        samples = write_ragas_samples(examples, docs, args.sample_output)
        elapsed = round(time.perf_counter() - start, 3)

        print({
            "dataset": args.dataset,
            "config": args.config_name,
            "split": args.split,
            "kb_id": kb.id,
            "kb_name": kb.name,
            "examples": len(examples),
            "documents": len(docs),
            "imported_or_updated_docs": imported["docs"],
            "parent_chunks": imported["parents"],
            "child_chunks": imported["children"],
            "sample_output": str(args.sample_output),
            "sample_count": samples,
            "seconds": elapsed,
        })
    finally:
        db.close()


def load_hotpotqa_examples(
    dataset: str,
    config_name: str,
    split: str,
    limit_examples: int,
) -> List[Dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "data" / "hf_cache" / "hub"))

    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = False
    except Exception:
        pass

    from datasets import load_dataset

    loaded = load_dataset(dataset, config_name, split=split)
    examples: List[Dict[str, Any]] = []
    for item in loaded:
        normalized = normalize_example(item)
        if normalized:
            examples.append(normalized)
        if len(examples) >= limit_examples:
            break
    return examples


def normalize_example(item: Dict[str, Any]) -> Dict[str, Any] | None:
    question = item.get("question")
    answer = item.get("answer")
    context = item.get("context") or {}
    titles = context.get("title") or []
    sentences_groups = context.get("sentences") or []
    if not question or not answer or not titles or not sentences_groups:
        return None

    supporting = item.get("supporting_facts") or {}
    supporting_titles = set(supporting.get("title") or [])
    contexts = []
    for title, sentences in zip(titles, sentences_groups):
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        if not text:
            continue
        doc_id = hotpot_doc_id(title)
        contexts.append({
            "doc_id": doc_id,
            "title": str(title),
            "text": text,
            "supporting": str(title) in supporting_titles,
        })

    if not contexts:
        return None

    return {
        "id": str(item.get("id") or item.get("_id") or stable_hash(question)),
        "question": str(question),
        "answer": str(answer),
        "contexts": contexts,
        "supporting_doc_ids": [ctx["doc_id"] for ctx in contexts if ctx["supporting"]],
    }


def hotpot_doc_id(title: str) -> str:
    return f"hp_{stable_hash(title)[:20]}"


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def build_docs(examples: Sequence[Dict[str, Any]]) -> Dict[str, HotpotDoc]:
    docs: Dict[str, HotpotDoc] = {}
    for example in examples:
        for ctx in example["contexts"]:
            doc_id = ctx["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = HotpotDoc(
                    doc_id=doc_id,
                    title=ctx["title"],
                    text=f"{ctx['title']}\n\n{ctx['text']}",
                )
    return docs


def import_docs(
    db,
    kb_id: int,
    user_id: int,
    docs: List[HotpotDoc],
    embed_batch_size: int,
    dataset: str,
) -> Dict[str, int]:
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

        upsert_document(db, kb_id=kb_id, doc=doc, parent_count=len(parent_chunks), child_count=len(child_chunks), dataset=dataset)
        sparse_by_chunk_id = {
            chunk.chunk_id: sparse_vectors[index]
            for index, chunk in enumerate(child_chunks)
            if index < len(sparse_vectors)
        }
        for chunk in parent_chunks + child_chunks:
            row = upsert_chunk(db, kb_id=kb_id, chunk=chunk, dataset=dataset)
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


def build_chunks(doc: HotpotDoc):
    parsed = {
        "type": "txt",
        "content": doc.text,
        "metadata": {"title": doc.title},
    }
    return document_chunker.chunk_document(parsed, doc.doc_id)


def upsert_document(db, kb_id: int, doc: HotpotDoc, parent_count: int, child_count: int, dataset: str) -> None:
    file_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    row = db.query(DocumentTable).filter(DocumentTable.doc_id == doc.doc_id).first()
    if row is None:
        row = DocumentTable(
            kb_id=kb_id,
            doc_id=doc.doc_id,
            filename=f"{doc.doc_id}.txt",
            file_path=f"public://{dataset}/{doc.doc_id}",
            file_hash=file_hash,
            file_size=len(doc.text.encode("utf-8")),
            status=DocumentStatus.COMPLETED,
            modality=ModalityType.TEXT,
            language="en",
            total_chunks=child_count,
            doc_metadata={
                "dataset": dataset,
                "eval": True,
                "task": "answer_generation",
                "parent_chunks": parent_count,
                "title": doc.title,
            },
        )
        db.add(row)
        return

    row.kb_id = kb_id
    row.filename = f"{doc.doc_id}.txt"
    row.file_path = f"public://{dataset}/{doc.doc_id}"
    row.file_hash = file_hash
    row.file_size = len(doc.text.encode("utf-8"))
    row.status = DocumentStatus.COMPLETED
    row.modality = ModalityType.TEXT
    row.language = "en"
    row.total_chunks = child_count
    row.doc_metadata = {
        "dataset": dataset,
        "eval": True,
        "task": "answer_generation",
        "parent_chunks": parent_count,
        "title": doc.title,
    }


def upsert_chunk(db, kb_id: int, chunk, dataset: str) -> ChunkTable:
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
    row.source_uri = f"public://{dataset}/{chunk.doc_id}"
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
            "language": row.language or "en",
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


def write_ragas_samples(examples: Sequence[Dict[str, Any]], docs: Dict[str, HotpotDoc], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            gold_doc_ids = example["supporting_doc_ids"] or [ctx["doc_id"] for ctx in example["contexts"][:2]]
            reference_context = "\n\n".join(docs[doc_id].text for doc_id in gold_doc_ids if doc_id in docs)
            row = {
                "id": f"hotpot_{example['id']}",
                "question": example["question"],
                "ground_truth": example["answer"],
                "reference_context": reference_context,
                "gold_doc_ids": gold_doc_ids,
                "dataset": DATASET_REPO_ID,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


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
