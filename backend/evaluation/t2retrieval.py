# -*- coding: utf-8 -*-
"""Helpers for the public mteb/T2Retrieval evaluation dataset."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO_ID = "mteb/T2Retrieval"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "evaluation" / "t2retrieval"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"

DATASET_FILES = {
    "corpus": "corpus/dev-00000-of-00001.parquet",
    "queries": "queries/dev-00000-of-00001.parquet",
    "qrels": "data/dev-00000-of-00001.parquet",
}


@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    title: str = ""


def prefixed_doc_id(raw_doc_id: str) -> str:
    return f"t2_{raw_doc_id}"


def download_dataset(dataset_dir: Path) -> None:
    """Download dataset files when the caller explicitly requests it."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "data" / "hf_cache" / "hub"))

    from huggingface_hub import hf_hub_download
    import huggingface_hub.constants as hf_constants

    hf_constants.HF_HUB_OFFLINE = False

    for filename in DATASET_FILES.values():
        target = dataset_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
        )
        print(f"downloaded: {downloaded}")


def require_dataset_files(dataset_dir: Path) -> None:
    missing = [
        str(dataset_dir / filename)
        for filename in DATASET_FILES.values()
        if not (dataset_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing T2Retrieval files. Run the import/eval script with --download first.\n"
            + "\n".join(f"- {item}" for item in missing)
        )


def load_dataset(dataset_dir: Path) -> Tuple[Dict[str, CorpusDoc], Dict[str, str], Dict[str, Dict[str, float]]]:
    require_dataset_files(dataset_dir)

    corpus_df = pd.read_parquet(dataset_dir / DATASET_FILES["corpus"])
    queries_df = pd.read_parquet(dataset_dir / DATASET_FILES["queries"])
    qrels_df = pd.read_parquet(dataset_dir / DATASET_FILES["qrels"])

    corpus_id_col = pick_column(corpus_df, ["_id", "id", "docid", "corpus_id"])
    corpus_text_col = pick_column(corpus_df, ["text", "contents", "content"])
    corpus_title_col = pick_optional_column(corpus_df, ["title"])
    query_id_col = pick_column(queries_df, ["_id", "id", "qid", "query_id"])
    query_text_col = pick_column(queries_df, ["text", "query"])
    qrel_query_col = pick_column(qrels_df, ["query-id", "query_id", "qid", "_id"])
    qrel_doc_col = pick_column(qrels_df, ["corpus-id", "corpus_id", "docid", "pid"])
    qrel_score_col = pick_optional_column(qrels_df, ["score", "relevance", "label"])

    corpus: Dict[str, CorpusDoc] = {}
    for _, row in corpus_df.iterrows():
        raw_doc_id = str(row[corpus_id_col])
        title = str(row[corpus_title_col]) if corpus_title_col and not pd.isna(row[corpus_title_col]) else ""
        text = str(row[corpus_text_col]) if not pd.isna(row[corpus_text_col]) else ""
        merged_text = "\n".join(part for part in [title, text] if part).strip()
        if merged_text:
            corpus[prefixed_doc_id(raw_doc_id)] = CorpusDoc(
                doc_id=prefixed_doc_id(raw_doc_id),
                text=merged_text,
                title=title,
            )

    queries = {
        str(row[query_id_col]): str(row[query_text_col])
        for _, row in queries_df.iterrows()
        if not pd.isna(row[query_text_col])
    }

    qrels: Dict[str, Dict[str, float]] = {}
    for _, row in qrels_df.iterrows():
        query_id = str(row[qrel_query_col])
        doc_id = prefixed_doc_id(str(row[qrel_doc_col]))
        score = float(row[qrel_score_col]) if qrel_score_col and not pd.isna(row[qrel_score_col]) else 1.0
        if score <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = score

    return corpus, queries, qrels


def pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of these columns exist: {candidates}. Actual columns: {list(df.columns)}")


def pick_optional_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def select_docs_for_eval(
    corpus: Dict[str, CorpusDoc],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    limit_corpus: int,
    limit_queries: int,
    seed: int,
) -> Dict[str, CorpusDoc]:
    rng = random.Random(seed)
    usable_query_ids = [
        query_id
        for query_id, docs in qrels.items()
        if query_id in queries and any(doc_id in corpus for doc_id in docs)
    ]
    rng.shuffle(usable_query_ids)
    selected_query_ids = usable_query_ids[:limit_queries]
    gold_doc_ids = {
        doc_id
        for query_id in selected_query_ids
        for doc_id in qrels[query_id]
        if doc_id in corpus
    }
    selected_doc_ids = set(gold_doc_ids)
    all_doc_ids = [
        doc_id
        for doc_id in corpus
        if doc_id not in selected_doc_ids
    ]
    rng.shuffle(all_doc_ids)
    for doc_id in all_doc_ids:
        if len(selected_doc_ids) >= limit_corpus:
            break
        selected_doc_ids.add(doc_id)
    return {doc_id: corpus[doc_id] for doc_id in selected_doc_ids}


def select_docs_for_all_gold(
    corpus: Dict[str, CorpusDoc],
    qrels: Dict[str, Dict[str, float]],
    limit_corpus: int,
    seed: int,
) -> Dict[str, CorpusDoc]:
    rng = random.Random(seed)
    gold_doc_ids = {
        doc_id
        for docs in qrels.values()
        for doc_id in docs
        if doc_id in corpus
    }
    selected_doc_ids = set(gold_doc_ids)
    all_doc_ids = list(corpus)
    rng.shuffle(all_doc_ids)
    for doc_id in all_doc_ids:
        if len(selected_doc_ids) >= limit_corpus:
            break
        selected_doc_ids.add(doc_id)
    return {doc_id: corpus[doc_id] for doc_id in selected_doc_ids}


def select_queries_for_eval(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    available_doc_ids: set[str],
    limit_queries: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    usable_query_ids = [
        query_id
        for query_id, docs in qrels.items()
        if query_id in queries and any(doc_id in available_doc_ids for doc_id in docs)
    ]
    rng.shuffle(usable_query_ids)
    return [(query_id, queries[query_id]) for query_id in usable_query_ids[:limit_queries]]
