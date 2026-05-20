# -*- coding: utf-8 -*-
"""Clean an evaluation knowledge base from MySQL and Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db.chroma.connection import chroma_client
from db.mysql.connection import SessionLocal, init_db
from models.database.knowledge import ChunkTable, DocumentTable, KnowledgeBasePermissionTable, KnowledgeBaseTable
from services.retrieval.hybrid import hybrid_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete an evaluation KB and its vectors.")
    parser.add_argument("--kb-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBaseTable).filter(KnowledgeBaseTable.id == args.kb_id).first()
        if not kb:
            print({"deleted": False, "reason": "kb_not_found", "kb_id": args.kb_id})
            return

        kb_name = kb.name
        chunk_ids = [
            row.chunk_id
            for row in db.query(ChunkTable.chunk_id).filter(ChunkTable.kb_id == args.kb_id).all()
        ]
        if chunk_ids:
            collection = chroma_client.get_collection()
            for start in range(0, len(chunk_ids), 500):
                collection.delete(ids=chunk_ids[start : start + 500])

        doc_count = db.query(DocumentTable).filter(DocumentTable.kb_id == args.kb_id).count()
        chunk_count = db.query(ChunkTable).filter(ChunkTable.kb_id == args.kb_id).count()
        db.query(ChunkTable).filter(ChunkTable.kb_id == args.kb_id).delete(synchronize_session=False)
        db.query(DocumentTable).filter(DocumentTable.kb_id == args.kb_id).delete(synchronize_session=False)
        db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == args.kb_id
        ).delete(synchronize_session=False)
        db.query(KnowledgeBaseTable).filter(KnowledgeBaseTable.id == args.kb_id).delete(synchronize_session=False)
        db.commit()
        hybrid_retriever.invalidate_sparse_cache(kb_id=args.kb_id)
        print({
            "deleted": True,
            "kb_id": args.kb_id,
            "kb_name": kb_name,
            "documents": doc_count,
            "chunks": chunk_count,
            "vectors": len(chunk_ids),
        })
    finally:
        db.close()


if __name__ == "__main__":
    main()
