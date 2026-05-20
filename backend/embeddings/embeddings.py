# -*- coding: utf-8 -*-
"""
Embedding adapters.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

try:
    from embeddings.bge_m3 import get_bge_m3_encoder
    from core.config import settings
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.embeddings.bge_m3 import get_bge_m3_encoder
    from backend.core.config import settings


class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION


class OpenAIEmbeddingsAdapter(BaseEmbeddings):
    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


class HuggingFaceEmbeddingsAdapter(BaseEmbeddings):
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


class BGEM3EmbeddingsAdapter(BaseEmbeddings):
    """Dense embedding adapter backed by BGE-M3."""

    def __init__(self):
        self._encoder = get_bge_m3_encoder()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encoder.encode_dense(texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._encoder.encode_dense([text])
        return vectors[0] if vectors else []


class DashScopeEmbeddingsAdapter(BaseEmbeddings):
    """
    DashScope / Qwen embedding adapter.

    Notes:
    - Batch size is capped to 10 to match the HTTP API limit.
    - Texts are normalized and lightly truncated to reduce 400 responses caused by
      empty or overlong inputs.
    """

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    MAX_BATCH_SIZE = 10
    MAX_TEXT_CHARS = 7000

    def __init__(self):
        self.api_key = settings.EMBEDDING_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.output_dimension = settings.EMBEDDING_DIMENSION

    def _prepare_text(self, text: str) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            cleaned = "[EMPTY]"
        if len(cleaned) > self.MAX_TEXT_CHARS:
            cleaned = cleaned[: self.MAX_TEXT_CHARS]
        return cleaned

    async def _request_embeddings(
        self,
        texts: List[str],
        text_type: str,
    ) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": {
                "texts": texts,
            },
            "parameters": {
                "text_type": text_type,
                "dimension": self.output_dimension,
                "output_type": "dense",
            },
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.API_URL, json=payload, headers=headers)

        if response.status_code >= 400:
            body_preview = response.text[:500]
            raise RuntimeError(
                f"DashScope embedding request failed: status={response.status_code}, body={body_preview}"
            )

        data = response.json()
        embeddings = data.get("output", {}).get("embeddings", [])
        embeddings_map = {
            item.get("text_index", index): item.get("embedding", [])
            for index, item in enumerate(embeddings)
        }
        return [embeddings_map.get(index, []) for index in range(len(texts))]

    async def _embed_async(
        self,
        texts: List[str],
        text_type: str,
    ) -> List[List[float]]:
        prepared_texts = [self._prepare_text(text) for text in texts]
        results: List[List[float]] = []

        for offset in range(0, len(prepared_texts), self.MAX_BATCH_SIZE):
            batch = prepared_texts[offset : offset + self.MAX_BATCH_SIZE]
            batch_vectors = await self._request_embeddings(batch, text_type=text_type)
            results.extend(batch_vectors)

        return results

    def _run(self, texts: List[str], text_type: str) -> List[List[float]]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._embed_async(texts, text_type=text_type))
                    return future.result()
            return asyncio.run(self._embed_async(texts, text_type=text_type))
        except RuntimeError:
            return asyncio.run(self._embed_async(texts, text_type=text_type))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._run(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        results = self._run([text], text_type="query")
        return results[0] if results else []


class EmbeddingFactory:
    _models: Dict[str, BaseEmbeddings] = {}

    @classmethod
    def get_model(cls, provider: Optional[str] = None) -> BaseEmbeddings:
        provider = provider or settings.EMBEDDING_PROVIDER

        if provider in cls._models:
            return cls._models[provider]

        if provider in {"bge", "bge_m3", "bge-m3"}:
            model = BGEM3EmbeddingsAdapter()
        elif provider == "openai":
            model = OpenAIEmbeddingsAdapter()
        elif provider == "huggingface":
            model = HuggingFaceEmbeddingsAdapter()
        elif provider in {"qwen", "dashscope"}:
            model = DashScopeEmbeddingsAdapter()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        cls._models[provider] = model
        return model

    @classmethod
    def reset(cls):
        cls._models.clear()


def get_embeddings() -> BaseEmbeddings:
    return EmbeddingFactory.get_model()


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embeddings().embed_documents(texts)


def embed_text(text: str) -> List[float]:
    return get_embeddings().embed_query(text)
