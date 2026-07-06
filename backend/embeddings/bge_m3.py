# -*- coding: utf-8 -*-
"""Shared BGE-M3 encoder for dense and learned sparse retrieval."""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Dict, List, Tuple

from backend.core.hf_cache import HF_HOME_PATH, configure_hf_cache
from backend.core.config import settings

configure_hf_cache()


class BGEM3Encoder:
    """Thin wrapper around FlagEmbedding's BGEM3FlagModel."""

    def __init__(self) -> None:
        configure_hf_cache()

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 dense/sparse encoding. "
                "Install it with: pip install -U FlagEmbedding"
            ) from exc

        self.model_name = settings.EMBEDDING_MODEL or "BAAI/bge-m3"
        self.batch_size = max(1, int(getattr(settings, "EMBEDDING_BATCH_SIZE", 16)))
        self.max_length = int(getattr(settings, "EMBEDDING_MAX_LENGTH", 8192))
        self._lock = threading.RLock()
        use_fp16 = _use_fp16_on_cuda(str(getattr(settings, "EMBEDDING_DEVICE", "cpu")))
        try:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load BGE-M3 from local Hugging Face cache. "
                f"Model={self.model_name}, HF_HOME={HF_HOME_PATH}. "
                "Runtime download is disabled; download the model manually before starting the backend."
            ) from exc

    def encode_dense(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            output = self.model.encode(
                self._prepare_texts(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return self._to_dense_vectors(output["dense_vecs"])

    def encode_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []
        with self._lock:
            output = self.model.encode(
                self._prepare_texts(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        return self._to_sparse_vectors(output["lexical_weights"])

    def encode_dense_and_sparse(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        if not texts:
            return [], []
        with self._lock:
            output = self.model.encode(
                self._prepare_texts(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        return (
            self._to_dense_vectors(output["dense_vecs"]),
            self._to_sparse_vectors(output["lexical_weights"]),
        )

    def sparse_score(self, query_sparse: Dict[str, float], doc_sparse: Dict[str, float]) -> float:
        if not query_sparse or not doc_sparse:
            return 0.0
        if len(query_sparse) > len(doc_sparse):
            query_sparse, doc_sparse = doc_sparse, query_sparse
        return float(sum(weight * doc_sparse.get(token, 0.0) for token, weight in query_sparse.items()))

    @staticmethod
    def _prepare_texts(texts: List[str]) -> List[str]:
        return [" ".join((text or "").split()) or "[EMPTY]" for text in texts]

    @staticmethod
    def _to_dense_vectors(raw_vectors) -> List[List[float]]:
        vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else raw_vectors
        if vectors and isinstance(vectors[0], (float, int)):
            return [[float(value) for value in vectors]]
        return [[float(value) for value in vector] for vector in vectors]

    @staticmethod
    def _to_sparse_vectors(raw_vectors) -> List[Dict[str, float]]:
        if isinstance(raw_vectors, dict):
            raw_vectors = [raw_vectors]
        sparse_vectors: List[Dict[str, float]] = []
        for vector in raw_vectors or []:
            sparse_vectors.append({
                str(token_id): float(weight)
                for token_id, weight in vector.items()
                if float(weight) > 0.0
            })
        return sparse_vectors


@lru_cache(maxsize=1)
def get_bge_m3_encoder() -> BGEM3Encoder:
    return BGEM3Encoder()


def _use_fp16_on_cuda(device: str) -> bool:
    if device.lower() != "cuda":
        return False
    try:
        import torch
    except ImportError:
        print("EMBEDDING_DEVICE=cuda but torch is not installed; falling back to CPU.")
        return False
    if not torch.cuda.is_available():
        print("EMBEDDING_DEVICE=cuda but CUDA is not available in this Python environment; falling back to CPU.")
        return False
    return True
