# -*- coding: utf-8 -*-
"""
Hybrid retrieval service.

Current target structure:
- Chroma: child dense vectors + retrieval metadata
- MySQL: parent/child content and structural metadata
- Retrieval: dense recall + independent sparse recall + weighted RRF + parent backfill
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from backend.db.chroma.connection import chroma_client
from backend.db.mysql.connection import SessionLocal
from backend.embeddings.embeddings import get_embeddings
from backend.embeddings.sparse import get_sparse_embedding
from backend.models.database.knowledge import ChunkTable
from backend.services.retrieval.diversity import cap_by_group, merge_by_chunk_id


class HybridRetriever:
    def __init__(
        self,
        dense_top_k: int = 40,
        sparse_top_k: int = 40,
        rerank_top_k: int = 8,
    ):
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_top_k = rerank_top_k
        self._embeddings = None
        self._sparse_embedding = None
        self.vector_client = chroma_client
        self._sparse_corpus_cache: Dict[Tuple[int, Optional[str]], List[Dict[str, Any]]] = {}

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def sparse_embedding(self):
        if self._sparse_embedding is None:
            self._sparse_embedding = get_sparse_embedding()
        return self._sparse_embedding

    def invalidate_sparse_cache(
        self,
        user_id: Optional[int] = None,  # kept for compatibility
        kb_id: Optional[int] = None,
        modality: Optional[str] = None,
    ) -> None:
        if kb_id is None:
            self._sparse_corpus_cache.clear()
            return
        self._sparse_corpus_cache.pop((int(kb_id), modality), None)

    async def retrieve_async(
        self,
        query: str,
        user_id: int,
        kb_id: Optional[int] = None,
        modality: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if kb_id is None:
            return []

        desired_top_k = max(6, min(top_k or self.rerank_top_k, 20))
        query_plan = self._build_query_plan(query, query_variants)

        all_dense_results: List[List[Dict[str, Any]]] = []
        all_sparse_results: List[List[Dict[str, Any]]] = []

        for variant, weight in query_plan:
            dense_results, sparse_results = await asyncio.gather(
                self._dense_retrieve_async(
                    query=variant,
                    user_id=user_id,
                    kb_id=kb_id,
                    modality=modality,
                    limit=max(desired_top_k * 5, self.dense_top_k // 2),
                ),
                self._sparse_retrieve_async(
                    query=variant,
                    kb_id=kb_id,
                    modality=modality,
                    limit=max(desired_top_k * 5, self.sparse_top_k // 2),
                ),
            )

            all_dense_results.append([
                {**item, "match_query": variant, "query_weight": weight}
                for item in dense_results
            ])
            all_sparse_results.append([
                {**item, "match_query": variant, "query_weight": weight}
                for item in sparse_results
            ])

        fused_results = self._fuse_multi_query_results(
            dense_groups=all_dense_results,
            sparse_groups=all_sparse_results,
        )
        candidate_limit = min(max(desired_top_k * 5, 50), 100)
        backfilled_results = await asyncio.to_thread(
            self._parent_backfill_sync,
            fused_results[:candidate_limit],
            kb_id,
        )
        return backfilled_results[:candidate_limit]

    def _build_query_plan(
        self,
        query: str,
        query_variants: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        raw_queries = [query] + (query_variants or [])
        seen = set()
        normalized_queries: List[str] = []
        for item in raw_queries:
            cleaned = " ".join((item or "").split())
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized_queries.append(cleaned)

        if not normalized_queries:
            return [(query, 1.0)]

        plan: List[Tuple[str, float]] = []
        for idx, item in enumerate(normalized_queries[:4]):
            if idx == 0:
                weight = 1.0
            elif idx == 1:
                weight = 0.9
            else:
                weight = max(0.5, 0.85 - idx * 0.1)
            plan.append((item, weight))
        return plan

    async def _dense_retrieve_async(
        self,
        query: str,
        user_id: int,
        kb_id: int,
        modality: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        query_vector = await asyncio.to_thread(self.embeddings.embed_query, query)
        collection = self.vector_client.get_collection()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=self._build_where_filter(user_id, kb_id, modality),
            include=["metadatas", "distances"],
        )
        dense_results = self._format_chroma_query_results(results)
        return await asyncio.to_thread(self._hydrate_child_results_sync, dense_results, kb_id)

    async def _sparse_retrieve_async(
        self,
        query: str,
        kb_id: int,
        modality: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        corpus = await self._load_sparse_corpus_async(kb_id=kb_id, modality=modality)
        if not corpus:
            return []

        query_sparse = await asyncio.to_thread(self.sparse_embedding.embed_query, query)
        corpus_stats = await asyncio.to_thread(
            self.sparse_embedding.build_corpus_stats,
            [item.get("sparse_vector") or {} for item in corpus],
        )
        scored_results: List[Dict[str, Any]] = []

        for item in corpus:
            lexical_score = self._sparse_score(
                query_sparse=query_sparse,
                doc_sparse=item.get("sparse_vector") or {},
                corpus_stats=corpus_stats,
            )
            if lexical_score <= 0:
                continue

            result = item.copy()
            result["score"] = lexical_score
            result["retrieval_type"] = "sparse"
            scored_results.append(result)

        scored_results.sort(key=lambda current: current.get("score", 0.0), reverse=True)
        return scored_results[:limit]

    async def _load_sparse_corpus_async(
        self,
        kb_id: int,
        modality: Optional[str],
    ) -> List[Dict[str, Any]]:
        cache_key = (int(kb_id), modality)
        cached = self._sparse_corpus_cache.get(cache_key)
        if cached is not None:
            return cached

        corpus = await asyncio.to_thread(self._load_sparse_corpus_sync, kb_id, modality)
        self._sparse_corpus_cache[cache_key] = corpus
        return corpus

    def _load_sparse_corpus_sync(
        self,
        kb_id: int,
        modality: Optional[str],
    ) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            query = db.query(ChunkTable).filter(
                ChunkTable.kb_id == kb_id,
                ChunkTable.is_parent.is_(False),
            )
            if modality is not None:
                query = query.filter(ChunkTable.modality == modality)

            rows = query.all()
            self._backfill_missing_sparse_vectors(db, rows)
            corpus: List[Dict[str, Any]] = []
            for row in rows:
                content = row.content or ""
                sparse_vector = row.sparse_vector or {}
                corpus.append({
                    "chunk_id": row.chunk_id,
                    "parent_id": row.parent_id or "",
                    "content": content,
                    "title": row.title,
                    "section_path": row.section_path,
                    "page_no": row.page_no,
                    "doc_id": row.doc_id,
                    "token_count": row.token_count,
                    "language": row.language,
                    "score": 0.0,
                    "rank": 0,
                    "retrieval_type": "sparse",
                    "sparse_vector": sparse_vector,
                })
            return corpus
        finally:
            db.close()

    def _backfill_missing_sparse_vectors(self, db, rows: List[ChunkTable]) -> None:
        missing_rows = [
            row
            for row in rows
            if not row.sparse_vector and (row.content or "").strip()
        ]
        if not missing_rows:
            return

        texts = [row.content or "" for row in missing_rows]
        vectors = self.sparse_embedding.embed_documents(texts)
        for row, vector in zip(missing_rows, vectors):
            row.sparse_vector = vector
        db.commit()

    def _build_where_filter(
        self,
        user_id: int,
        kb_id: int,
        modality: Optional[str],
    ) -> Dict[str, Any]:
        filters: List[Dict[str, Any]] = [
            {"kb_id": int(kb_id)},
            {"is_parent": False},
        ]
        if modality is not None:
            filters.append({"modality": str(modality)})
        return {"$and": filters}

    def _format_chroma_query_results(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        ids_group = results.get("ids") or [[]]
        metas_group = results.get("metadatas") or [[]]
        distances_group = results.get("distances") or [[]]

        formatted: List[Dict[str, Any]] = []
        for rank, chunk_id in enumerate(ids_group[0]):
            metadata = metas_group[0][rank] or {}
            distance = (
                distances_group[0][rank]
                if distances_group and distances_group[0]
                else 0.0
            ) or 0.0
            formatted.append({
                "chunk_id": metadata.get("chunk_id") or chunk_id,
                "parent_id": metadata.get("parent_id") or "",
                "title": metadata.get("title"),
                "section_path": metadata.get("section_path"),
                "page_no": metadata.get("page_no"),
                "doc_id": metadata.get("doc_id"),
                "token_count": metadata.get("token_count"),
                "language": metadata.get("language"),
                "score": round(1.0 / (1.0 + float(distance)), 6),
                "rank": rank,
                "retrieval_type": "dense",
            })
        return formatted

    def _hydrate_child_results_sync(
        self,
        dense_results: List[Dict[str, Any]],
        kb_id: int,
    ) -> List[Dict[str, Any]]:
        if not dense_results:
            return []

        chunk_ids = [item["chunk_id"] for item in dense_results]
        db = SessionLocal()
        try:
            rows = db.query(ChunkTable).filter(
                ChunkTable.kb_id == kb_id,
                ChunkTable.chunk_id.in_(chunk_ids),
                ChunkTable.is_parent.is_(False),
            ).all()
            row_map = {row.chunk_id: row for row in rows}

            hydrated: List[Dict[str, Any]] = []
            for item in dense_results:
                row = row_map.get(item["chunk_id"])
                result = item.copy()
                if row:
                    result.update({
                        "parent_id": row.parent_id or item.get("parent_id") or "",
                        "content": row.content or "",
                        "title": row.title or item.get("title"),
                        "section_path": row.section_path or item.get("section_path"),
                        "page_no": row.page_no if row.page_no is not None else item.get("page_no"),
                        "doc_id": row.doc_id or item.get("doc_id"),
                        "token_count": row.token_count or item.get("token_count"),
                        "language": row.language or item.get("language"),
                    })
                hydrated.append(result)
            return hydrated
        finally:
            db.close()

    def _sparse_score(
        self,
        query_sparse: Dict[str, float],
        doc_sparse: Dict[str, float],
        corpus_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not query_sparse or not doc_sparse:
            return 0.0

        if hasattr(self.sparse_embedding, "score"):
            score = self.sparse_embedding.score(query_sparse, doc_sparse, corpus_stats=corpus_stats)
        else:
            score = sum(q_weight * doc_sparse.get(token, 0.0) for token, q_weight in query_sparse.items())
        return round(score, 6)

    def _fuse_multi_query_results(
        self,
        dense_groups: List[List[Dict[str, Any]]],
        sparse_groups: List[List[Dict[str, Any]]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        ranked_groups = [
            (results, 1.0)
            for results in dense_groups
        ] + [
            (results, 0.9)
            for results in sparse_groups
        ]
        fused = merge_by_chunk_id(ranked_groups, k=k)
        return cap_by_group(fused, max_per_doc=5, max_per_parent=2, max_per_chunk=1)

    def _parent_backfill_sync(
        self,
        child_results: List[Dict[str, Any]],
        kb_id: int,
    ) -> List[Dict[str, Any]]:
        parent_ids = list({
            result["parent_id"]
            for result in child_results
            if result.get("parent_id")
        })
        if not parent_ids:
            return child_results

        db = SessionLocal()
        try:
            rows = db.query(ChunkTable).filter(
                ChunkTable.kb_id == kb_id,
                ChunkTable.chunk_id.in_(parent_ids),
                ChunkTable.is_parent.is_(True),
            ).all()
            parent_map = {row.chunk_id: row for row in rows}

            backfilled: List[Dict[str, Any]] = []
            for child in child_results:
                enriched = child.copy()
                parent = parent_map.get(child.get("parent_id"))
                if parent:
                    enriched["parent_content"] = parent.content
                    enriched["parent_title"] = parent.title
                    enriched["parent_section_path"] = parent.section_path
                    enriched["parent_page_no"] = parent.page_no
                    enriched["parent_token_count"] = parent.token_count
                backfilled.append(enriched)
            return cap_by_group(backfilled, max_per_doc=5, max_per_parent=2, max_per_chunk=1)
        finally:
            db.close()


hybrid_retriever = HybridRetriever()
