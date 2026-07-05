# -*- coding: utf-8 -*-
"""Diversity controls for retrieval candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clean_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def merge_by_chunk_id(
    ranked_groups: Iterable[Tuple[Iterable[Dict[str, Any]], float]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge repeated child chunks from multi-query/multi-retriever RRF hits."""
    scores: Dict[str, float] = {}
    merged: Dict[str, Dict[str, Any]] = {}

    for results, retrieval_weight in ranked_groups:
        for rank, result in enumerate(results):
            chunk_id = _clean_key(result.get("chunk_id"))
            if not chunk_id:
                continue

            query_weight = float(result.get("query_weight", 1.0))
            weighted_rrf = retrieval_weight * query_weight / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weighted_rrf

            existing = merged.get(chunk_id)
            if not existing:
                merged[chunk_id] = result.copy()
                continue

            existing["score"] = max(existing.get("score", 0.0), result.get("score", 0.0))
            existing["match_query"] = existing.get("match_query") or result.get("match_query")
            existing["retrieval_type"] = (
                existing.get("retrieval_type")
                if existing.get("retrieval_type") == result.get("retrieval_type")
                else "hybrid"
            )

    fused: List[Dict[str, Any]] = []
    for chunk_id, result in merged.items():
        fused_result = result.copy()
        fused_result["rrf_score"] = round(scores.get(chunk_id, 0.0), 8)
        fused.append(fused_result)

    fused.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return fused


def cap_by_group(
    results: Sequence[Dict[str, Any]],
    *,
    max_per_doc: Optional[int] = None,
    max_per_parent: Optional[int] = None,
    max_per_chunk: Optional[int] = 1,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Keep ranked order while applying soft caps by metadata groups."""
    counts: Dict[str, Dict[str, int]] = {
        "doc_id": defaultdict(int),
        "parent_id": defaultdict(int),
        "chunk_id": defaultdict(int),
    }
    capped: List[Dict[str, Any]] = []

    for result in results:
        checks = (
            ("doc_id", max_per_doc),
            ("parent_id", max_per_parent),
            ("chunk_id", max_per_chunk),
        )
        should_skip = False
        keys_to_count: List[Tuple[str, str]] = []

        for field, cap in checks:
            key = _clean_key(result.get(field))
            if key is None or cap is None:
                continue
            if counts[field][key] >= cap:
                should_skip = True
                break
            keys_to_count.append((field, key))

        if should_skip:
            continue

        for field, key in keys_to_count:
            counts[field][key] += 1
        capped.append(result)

        if limit is not None and len(capped) >= limit:
            break

    return capped


def select_diverse_results(
    results: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    max_per_doc: Optional[int] = 3,
    max_per_parent: Optional[int] = 1,
) -> List[Dict[str, Any]]:
    """Final ranked selection with staged fallback when caps under-fill top_k."""
    selected = cap_by_group(
        results,
        max_per_doc=max_per_doc,
        max_per_parent=max_per_parent,
        max_per_chunk=1,
        limit=top_k,
    )

    if top_k is None or len(selected) >= top_k:
        return selected

    selected_keys = {_result_identity(item) for item in selected}

    def append_missing(candidates: Sequence[Dict[str, Any]]) -> None:
        nonlocal selected
        for candidate in candidates:
            identity = _result_identity(candidate)
            if identity in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(identity)
            if len(selected) >= top_k:
                return

    parent_relaxed = cap_by_group(
        results,
        max_per_doc=max_per_doc,
        max_per_parent=None,
        max_per_chunk=1,
    )
    append_missing(parent_relaxed)

    if len(selected) >= top_k:
        return selected

    doc_relaxed = cap_by_group(
        results,
        max_per_doc=None,
        max_per_parent=None,
        max_per_chunk=1,
    )
    append_missing(doc_relaxed)

    return selected


def _result_identity(result: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    chunk_id = _clean_key(result.get("chunk_id"))
    if chunk_id:
        return f"chunk:{chunk_id}"
    parent_id = _clean_key(result.get("parent_id"))
    doc_id = _clean_key(result.get("doc_id"))
    if parent_id or doc_id:
        return f"doc-parent:{doc_id or ''}:{parent_id or ''}"
    if fallback_index is not None:
        return f"index:{fallback_index}"
    return f"object:{id(result)}"
