# -*- coding: utf-8 -*-
"""
文档入库服务。

当前目标结构：
- Chroma: 仅保存子块的 dense 向量和检索元数据
- MySQL: 保存文档、父块、子块正文及结构化元数据
"""

import os
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from backend.services.ingestion.parser import document_parser
from backend.services.ingestion.chunker import document_chunker, Chunk
from backend.embeddings.embeddings import get_embeddings
from backend.embeddings.sparse import get_sparse_embedding
from backend.db.chroma.connection import chroma_client
from backend.models.database.knowledge import DocumentTable, ChunkTable
from backend.models.schemas.knowledge import DocumentStatus, ModalityType


class IngestionService:
    """文档入库服务。"""

    def __init__(self):
        self.parser = document_parser
        self.chunker = document_chunker
        self._embeddings = None
        self._sparse_embeddings = None
        self.vector_client = chroma_client

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def sparse_embeddings(self):
        if self._sparse_embeddings is None:
            self._sparse_embeddings = get_sparse_embedding()
        return self._sparse_embeddings

    async def ingest_document(
        self,
        file_path: str,
        kb_id: int,
        user_id: int,
        db: Session,
        task_callback: Optional[callable] = None,
        original_filename: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if task_callback:
                await task_callback("parsing", 10, "正在解析文档...")

            parsed = self.parser.parse(file_path)
            doc_id = self.parser.generate_doc_id(file_path)

            if task_callback:
                await task_callback("chunking", 30, "正在进行父子分块...")

            parent_chunks, child_chunks = self.chunker.chunk_document(parsed, doc_id)

            if task_callback:
                await task_callback("image_describing", 40, "正在生成图片语义描述...")

            # 图片块：OCR 文字 + VLM 语义描述拼接（方案 A），供文本检索召回
            self._describe_image_chunks(child_chunks)

            if task_callback:
                await task_callback("embedding", 50, "正在构建 dense/sparse 表示...")

            child_texts = [chunk.content for chunk in child_chunks]
            child_dense_vectors, child_sparse_vectors = self._embed_child_chunks(child_texts)

            if task_callback:
                await task_callback("saving", 72, "正在写入关系数据库...")

            self._save_metadata(
                db=db,
                kb_id=kb_id,
                doc_id=doc_id,
                file_path=file_path,
                original_filename=original_filename,
                file_hash=file_hash,
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                child_sparse_vectors=child_sparse_vectors,
            )

            if task_callback:
                await task_callback("indexing", 88, "正在写入向量数据库...")

            self._upsert_child_vectors(
                user_id=user_id,
                kb_id=kb_id,
                doc_id=doc_id,
                child_chunks=child_chunks,
                child_dense_vectors=child_dense_vectors,
                child_sparse_vectors=child_sparse_vectors,
            )

            from backend.services.retrieval.hybrid import hybrid_retriever
            hybrid_retriever.invalidate_sparse_cache(user_id=user_id, kb_id=kb_id)

            if task_callback:
                await task_callback("completed", 100, "完成")

            # 文档级 modality 与各模态块统计（用于上传响应）
            non_text = {c.modality for c in child_chunks if c.modality != ModalityType.TEXT}
            doc_modality = ModalityType.MIXED if non_text else ModalityType.TEXT

            return {
                "doc_id": doc_id,
                "parent_count": len(parent_chunks),
                "child_count": len(child_chunks),
                "status": "success",
                "modality": doc_modality.value,
                "modality_stats": self._count_modality_stats(child_chunks),
            }
        except Exception as e:
            if task_callback:
                await task_callback("failed", 0, f"失败: {str(e)}")
            raise

    def _embed_child_chunks(self, child_texts: List[str]) -> tuple[List[List[float]], List[Dict[str, float]]]:
        if not child_texts:
            return [], []

        return (
            self.embeddings.embed_documents(child_texts),
            self.sparse_embeddings.embed_documents(child_texts),
        )

    def _describe_image_chunks(self, child_chunks: List[Chunk]) -> None:
        """为 IMAGE 块生成可检索文本：OCR 文字 + VLM 语义描述拼接。

        图片块 content 前缀 `[图片]` 标记，便于阶段2 据此定位 IMAGE 块回填图像向量。
        VLM 描述依赖 VISION_API_KEY；未配置或失败时降级为仅 OCR 文字。
        """
        from backend.services.vision.ocr import ocr_service

        for chunk in child_chunks:
            if chunk.modality != ModalityType.IMAGE:
                continue
            source_uri = chunk.metadata.get("source_uri") or ""
            if not source_uri or not os.path.exists(source_uri):
                continue

            try:
                ocr_text = ocr_service.extract_text(source_uri).get("text") or ""
            except Exception:
                ocr_text = ""
            try:
                desc_text = ocr_service.describe_image(source_uri).get("text") or ""
            except Exception:
                desc_text = ""

            combined = "\n".join(part for part in ["[图片]", ocr_text, desc_text] if part)
            if combined.strip():
                chunk.content = combined
                chunk.metadata["caption"] = chunk.metadata.get("caption") or ""
                chunk.token_count = self.chunker.estimate_tokens(combined)

    def _upsert_child_vectors(
        self,
        user_id: int,
        kb_id: int,
        doc_id: str,
        child_chunks: List[Chunk],
        child_dense_vectors: List[List[float]],
        child_sparse_vectors: List[Dict[str, float]],
    ) -> None:
        if not child_chunks:
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(child_chunks):
            sparse_vector = child_sparse_vectors[idx] if idx < len(child_sparse_vectors) else {}
            lexical_terms = [
                token
                for token in sparse_vector.keys()
                if not token.startswith("__")
            ]
            metadatas.append({
                **self._build_metadata(
                    user_id=user_id,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    chunk=chunk,
                    is_parent=False,
                ),
                # 以轻量 lexical terms 形式保留 sparse 语义线索，便于调试和回溯。
                "lexical_terms": " ".join(lexical_terms[:24]),
            })
            ids.append(chunk.chunk_id)
            embeddings.append(child_dense_vectors[idx])

        collection = self.vector_client.get_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def _build_metadata(
        self,
        user_id: int,
        kb_id: int,
        doc_id: str,
        chunk: Chunk,
        is_parent: bool,
    ) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "parent_id": chunk.parent_id or "",
            "chunk_id": chunk.chunk_id,
            "is_parent": is_parent,
            "modality": chunk.modality.value,
            "language": chunk.language,
            "title": chunk.title or "",
            "section_path": chunk.section_path or "",
            "page_no": chunk.page_no or 0,
            "token_count": chunk.token_count,
            "caption": chunk.metadata.get("caption") or "",
            "source_uri": chunk.metadata.get("source_uri") or "",
            "code_language": chunk.code_language or chunk.metadata.get("code_language") or "",
        }

    def _save_metadata(
        self,
        db: Session,
        kb_id: int,
        doc_id: str,
        file_path: str,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk],
        original_filename: Optional[str] = None,
        file_hash: Optional[str] = None,
        child_sparse_vectors: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """保存文档、父块、子块正文及元数据到 MySQL。"""
        from pathlib import Path

        filename = original_filename or Path(file_path).name
        file_size = Path(file_path).stat().st_size

        # 文档级 modality：含图片/表格/代码 → mixed，否则 text
        non_text = {c.modality for c in child_chunks if c.modality != ModalityType.TEXT}
        doc_modality = ModalityType.MIXED if non_text else ModalityType.TEXT

        doc_record = db.query(DocumentTable).filter(DocumentTable.doc_id == doc_id).first()
        if not doc_record:
            doc_record = DocumentTable(
                kb_id=kb_id,
                doc_id=doc_id,
                filename=filename,
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                status=DocumentStatus.COMPLETED,
                modality=doc_modality,
                language="zh",
                total_chunks=len(child_chunks),
            )
            db.add(doc_record)
        else:
            doc_record.status = DocumentStatus.COMPLETED
            doc_record.total_chunks = len(child_chunks)
            doc_record.filename = filename
            doc_record.file_hash = file_hash
            doc_record.modality = doc_modality

        db.commit()

        sparse_by_chunk_id = {
            chunk.chunk_id: child_sparse_vectors[idx]
            for idx, chunk in enumerate(child_chunks)
            if child_sparse_vectors and idx < len(child_sparse_vectors)
        }

        for chunk in parent_chunks + child_chunks:
            chunk_record = db.query(ChunkTable).filter(ChunkTable.chunk_id == chunk.chunk_id).first()
            if not chunk_record:
                chunk_record = ChunkTable(
                    kb_id=kb_id,
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                )
                db.add(chunk_record)

            chunk_record.parent_id = chunk.parent_id
            chunk_record.is_parent = chunk.parent_id is None
            chunk_record.modality = chunk.modality
            chunk_record.language = chunk.language
            chunk_record.title = chunk.title
            chunk_record.section_path = chunk.section_path
            chunk_record.page_no = chunk.page_no
            chunk_record.token_count = chunk.token_count
            chunk_record.content = chunk.content
            chunk_record.sparse_vector = sparse_by_chunk_id.get(chunk.chunk_id)
            chunk_record.caption = chunk.metadata.get("caption") or chunk_record.caption
            chunk_record.source_uri = chunk.metadata.get("source_uri") or chunk_record.source_uri
            chunk_record.code_language = (
                chunk.code_language
                or chunk.metadata.get("code_language")
                or chunk_record.code_language
            )

        db.commit()

    @staticmethod
    def _count_modality_stats(child_chunks: List[Chunk]) -> Dict[str, int]:
        """统计各模态 child chunk 数量（text/table/code/image）。"""
        stats: Dict[str, int] = {}
        for chunk in child_chunks:
            key = chunk.modality.value
            stats[key] = stats.get(key, 0) + 1
        return stats

    async def delete_document(self, doc_id: str, kb_id: int, db: Session) -> bool:
        """删除文档。"""
        try:
            collection = self.vector_client.get_collection()
            chroma_results = collection.get(where={"doc_id": doc_id}, include=[])
            ids = chroma_results.get("ids") or []
            if ids:
                collection.delete(ids=ids)

            db.query(ChunkTable).filter(ChunkTable.doc_id == doc_id).delete()
            db.query(DocumentTable).filter(
                DocumentTable.doc_id == doc_id,
                DocumentTable.kb_id == kb_id,
            ).delete()
            db.commit()

            from backend.services.retrieval.hybrid import hybrid_retriever
            hybrid_retriever.invalidate_sparse_cache()
            return True
        except Exception as e:
            db.rollback()
            raise e

    def get_document_status(self, doc_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """获取文档状态。"""
        doc = db.query(DocumentTable).filter(DocumentTable.doc_id == doc_id).first()
        if not doc:
            return None

        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "status": doc.status.value,
            "total_chunks": doc.total_chunks,
            "error_message": doc.error_message,
            "created_at": doc.created_at.isoformat(),
        }


ingestion_service = IngestionService()
