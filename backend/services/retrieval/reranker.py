# -*- coding: utf-8 -*-
"""Local BGE reranker service."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from core.hf_cache import HF_HOME_PATH, configure_hf_cache
    from core.config import settings
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.core.hf_cache import HF_HOME_PATH, configure_hf_cache
    from backend.core.config import settings

configure_hf_cache()


class BaseReranker(ABC):
    @abstractmethod
    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pass

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.rerank_async(query, documents, top_k))
                    return future.result()
            return asyncio.run(self.rerank_async(query, documents, top_k))
        except RuntimeError:
            return asyncio.run(self.rerank_async(query, documents, top_k))


class BGEReranker(BaseReranker):
    """Rerank candidate parent chunks with BAAI/bge-reranker-v2-m3."""

    def __init__(self):
        configure_hf_cache()

        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "FlagEmbedding is required for BGE reranking. Install it with: pip install -U FlagEmbedding"
            ) from exc

        self.model_name = settings.RERANKER_MODEL or "BAAI/bge-reranker-v2-m3"
        self.batch_size = max(1, int(getattr(settings, "RERANKER_BATCH_SIZE", 8)))
        use_fp16 = _use_fp16_on_cuda(str(getattr(settings, "RERANKER_DEVICE", "cpu")))
        try:
            self.model = FlagReranker(self.model_name, use_fp16=use_fp16)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load BGE reranker from local Hugging Face cache. "
                f"Model={self.model_name}, HF_HOME={HF_HOME_PATH}. "
                "Runtime download is disabled; download the model manually before starting the backend."
            ) from exc

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.rerank(query, documents, top_k))

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [
            [" ".join((query or "").split())[:2000], self._prepare_document(document)]
            for document in documents
        ]
        try:
            scores = self.model.compute_score(pairs, batch_size=self.batch_size, normalize=True)
        except TypeError:
            scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, (float, int)):
            scores = [scores]

        reranked_docs: List[Dict[str, Any]] = []
        for document, score in zip(documents, scores):
            item = document.copy()
            item["rerank_score"] = float(score or 0.0)
            item["rerank_model"] = self.model_name
            reranked_docs.append(item)

        reranked_docs.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        reranked_docs = _deduplicate_ranked_documents(reranked_docs)
        return reranked_docs[:top_k] if top_k else reranked_docs

    @staticmethod
    def _prepare_document(document: Dict[str, Any]) -> str:
        parent_text = " ".join((document.get("parent_content") or "").split())
        child_text = " ".join((document.get("content") or "").split())
        title = " ".join((document.get("parent_title") or document.get("title") or "").split())
        section = " ".join((document.get("parent_section_path") or document.get("section_path") or "").split())

        text = parent_text or child_text or "[EMPTY]"
        headers = " / ".join(part for part in [title, section] if part)
        if headers and headers not in text:
            text = f"{headers}\n{text}"
        if parent_text and child_text and child_text not in parent_text:
            text = f"{text}\n\nChild snippet:\n{child_text[:1200]}"
        return text[:6000]


class SimpleReranker(BaseReranker):
    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        import re

        query_words = set(re.findall(r"\w+", (query or "").lower()))
        reranked: List[Dict[str, Any]] = []

        for doc in documents:
            content = ((doc.get("parent_content") or doc.get("content") or "")).lower()
            title = ((doc.get("parent_title") or doc.get("title") or "")).lower()
            match_count = 0
            for word in query_words:
                if word in content:
                    match_count += 1
                if word in title:
                    match_count += 2
            item = doc.copy()
            item["rerank_score"] = match_count / (len(query_words) * 3) if query_words else 0.0
            reranked.append(item)

        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        reranked = _deduplicate_ranked_documents(reranked)
        return reranked[:top_k] if top_k else reranked


def get_reranker() -> BaseReranker:
    provider = (settings.RERANKER_PROVIDER or "bge").lower()
    if provider in {"bge", "bge_reranker", "bge-reranker"}:
        return BGEReranker()
    if provider == "simple":
        return SimpleReranker()
    raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")


class LazyReranker(BaseReranker):
    def __init__(self):
        self._instance: Optional[BaseReranker] = None

    @property
    def instance(self) -> BaseReranker:
        if self._instance is None:
            self._instance = get_reranker()
        return self._instance

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await self.instance.rerank_async(query, documents, top_k)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.instance.rerank(query, documents, top_k)


reranker = LazyReranker()


def _deduplicate_ranked_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        key = str(document.get("doc_id") or document.get("parent_id") or document.get("chunk_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(document)
    return deduped


def _use_fp16_on_cuda(device: str) -> bool:
    if device.lower() != "cuda":
        return False
    try:
        import torch
    except ImportError:
        print("RERANKER_DEVICE=cuda but torch is not installed; falling back to CPU.")
        return False
    if not torch.cuda.is_available():
        print("RERANKER_DEVICE=cuda but CUDA is not available in this Python environment; falling back to CPU.")
        return False
    return True
