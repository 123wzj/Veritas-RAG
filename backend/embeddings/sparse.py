# -*- coding: utf-8 -*-
"""BM25 lexical sparse embeddings.

Dense vectors capture semantic similarity. BM25 sparse vectors capture exact
lexical matches such as product names, APIs, section titles, numbers, and
error messages. The corpus-level IDF and average length are computed by the
retrieval layer for the selected knowledge base.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import jieba
except ImportError:  # pragma: no cover - jieba is optional at import time
    jieba = None


DOC_LEN_KEY = "__doc_len__"


class BM25Encoder:
    """A lightweight BM25 encoder for Chinese/English text."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        min_token_len: int = 1,
    ):
        self.k1 = k1
        self.b = b
        self.min_token_len = min_token_len

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        normalized = text.lower()
        english_tokens = re.findall(r"[a-z0-9][a-z0-9_\-.+#/]*", normalized)
        chinese_text = re.sub(r"[a-z0-9_\-.+#/]+", " ", normalized)

        if jieba is not None:
            chinese_tokens = [
                token.strip()
                for token in jieba.lcut(chinese_text)
                if token.strip()
            ]
        else:
            chinese_tokens = re.findall(r"[\u4e00-\u9fff]+", chinese_text)

        tokens = english_tokens + chinese_tokens
        return [
            token
            for token in tokens
            if len(token) >= self.min_token_len and not token.isspace()
        ]

    def encode(self, text: str) -> Dict[str, float]:
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        vector: Dict[str, float] = {
            token: float(freq)
            for token, freq in Counter(tokens).items()
        }
        vector[DOC_LEN_KEY] = float(len(tokens))
        return vector

    def build_corpus_stats(self, sparse_vectors: List[Dict[str, float]]) -> Dict[str, Any]:
        doc_count = len(sparse_vectors)
        if doc_count == 0:
            return {"idf": {}, "avg_doc_len": 1.0}

        doc_freq: Counter[str] = Counter()
        total_len = 0.0

        for vector in sparse_vectors:
            total_len += float(vector.get(DOC_LEN_KEY) or 0.0)
            for token, value in vector.items():
                if token == DOC_LEN_KEY or not value:
                    continue
                doc_freq[token] += 1

        idf = {
            token: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        return {
            "idf": idf,
            "avg_doc_len": max(total_len / max(doc_count, 1), 1.0),
        }

    def score(
        self,
        query_sparse: Dict[str, float],
        doc_sparse: Dict[str, float],
        corpus_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not query_sparse or not doc_sparse:
            return 0.0

        corpus_stats = corpus_stats or {}
        idf = corpus_stats.get("idf") or {}
        avg_doc_len = float(corpus_stats.get("avg_doc_len") or doc_sparse.get(DOC_LEN_KEY) or 1.0)
        doc_len = float(doc_sparse.get(DOC_LEN_KEY) or avg_doc_len or 1.0)

        score = 0.0
        for token, query_weight in query_sparse.items():
            if token == DOC_LEN_KEY:
                continue

            freq = float(doc_sparse.get(token, 0.0) or 0.0)
            if freq <= 0:
                continue

            term_idf = float(idf.get(token, 0.0) or 0.0)
            if term_idf <= 0:
                continue

            denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_len / avg_doc_len)
            term_score = term_idf * ((freq * (self.k1 + 1.0)) / denominator)
            score += term_score * self._token_boost(token) * max(float(query_weight), 1.0)
        return score

    def _token_boost(self, token: str) -> float:
        # Longer identifiers and mixed alpha-numeric terms are usually more
        # discriminative in technical documents.
        boost = 1.0
        if len(token) >= 4:
            boost += 0.15
        if any(char.isdigit() for char in token):
            boost += 0.1
        if re.search(r"[a-z]", token) and re.search(r"\d", token):
            boost += 0.15
        return boost


class SparseEmbedding:
    """Sparse embedding interface used by ingestion and retrieval."""

    def __init__(self):
        self._encoder = BM25Encoder()

    def fit(self, texts: List[str]):
        """Compatibility no-op; documents are indexed incrementally."""
        return None

    def embed_documents(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self._encoder.encode(text) for text in texts]

    def embed_query(self, text: str) -> Dict[str, float]:
        return self._encoder.encode(text)

    def build_corpus_stats(self, sparse_vectors: List[Dict[str, float]]) -> Dict[str, Any]:
        return self._encoder.build_corpus_stats(sparse_vectors)

    def score(
        self,
        query_sparse: Dict[str, float],
        doc_sparse: Dict[str, float],
        corpus_stats: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self._encoder.score(query_sparse, doc_sparse, corpus_stats=corpus_stats)


sparse_embedding: Optional[SparseEmbedding] = None


def get_sparse_embedding() -> SparseEmbedding:
    global sparse_embedding
    if sparse_embedding is None:
        sparse_embedding = SparseEmbedding()
    return sparse_embedding
