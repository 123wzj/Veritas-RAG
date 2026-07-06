# -*- coding: utf-8 -*-
"""Download LexRAG and prepare a structurally valid multi-turn benchmark."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "lexrag"
DATASET_URL = "https://raw.githubusercontent.com/CSHaitao/LexRAG/main/data/dataset.json"
LAW_LIBRARY_URL = "https://raw.githubusercontent.com/CSHaitao/LexRAG/main/data/law_library.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LexRAG for Veritas-RAG evaluation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--conversation-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.output_root / "source"
    dataset_path = source_root / "dataset.json"
    law_path = source_root / "law_library.jsonl"
    download(DATASET_URL, dataset_path, args.force_download)
    download(LAW_LIBRARY_URL, law_path, args.force_download)

    raw_dialogues = json.loads(dataset_path.read_text(encoding="utf-8"))
    laws = load_jsonl(law_path)
    law_names = {row["name"] for row in laws}
    valid_dialogues, rejected = filter_valid_dialogues(raw_dialogues, law_names)
    selected = select_balanced(valid_dialogues, args.conversation_count, args.seed)
    conversations = [convert_dialogue(row) for row in selected]
    documents = [
        {
            "dataset": "LexRAG",
            "law_id": row["id"],
            "doc_id": row["name"],
            "title": row["name"],
            "text": clean_text(row["content"]),
        }
        for row in laws
    ]

    prepared_root = args.output_root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    documents_path = prepared_root / "lexrag_law_library.jsonl"
    conversations_path = prepared_root / (
        f"lexrag_conversations_{len(conversations)}x5.jsonl"
    )
    write_jsonl(documents_path, documents)
    write_jsonl(conversations_path, conversations)

    print(json.dumps({
        "dataset": "LexRAG",
        "source_dialogues": len(raw_dialogues),
        "valid_dialogues": len(valid_dialogues),
        "rejected_dialogues": len(rejected),
        "law_documents": len(documents),
        "selected_conversations": len(conversations),
        "selected_turns": sum(len(row["turns"]) for row in conversations),
        "selected_types": dict(sorted(Counter(row["category"] for row in conversations).items())),
        "documents_path": str(documents_path),
        "conversations_path": str(conversations_path),
        "rejected_examples": rejected[:10],
    }, ensure_ascii=False, indent=2))


def download(url: str, path: Path, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        urllib.request.urlretrieve(url, path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_valid_dialogues(
    dialogues: List[Dict[str, Any]],
    law_names: set[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for dialogue in dialogues:
        missing = sorted({
            str(article)
            for turn in dialogue.get("conversation") or []
            for article in turn.get("article") or []
            if article not in law_names
        })
        invalid_turns = sum(
            1
            for turn in dialogue.get("conversation") or []
            if not turn.get("user") or not turn.get("assistant") or not turn.get("article")
        )
        if missing or invalid_turns or len(dialogue.get("conversation") or []) != 5:
            rejected.append({
                "id": dialogue.get("id"),
                "missing_articles": missing,
                "invalid_turns": invalid_turns,
                "turn_count": len(dialogue.get("conversation") or []),
            })
            continue
        valid.append(dialogue)
    return valid, rejected


def select_balanced(
    dialogues: List[Dict[str, Any]],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in dialogues:
        buckets.setdefault(str(row.get("type") or "unknown"), []).append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    selected: List[Dict[str, Any]] = []
    categories = sorted(buckets)
    while len(selected) < count:
        added = False
        for category in categories:
            rows = buckets[category]
            if rows:
                selected.append(rows.pop())
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
    return selected


def convert_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    turns = []
    for index, turn in enumerate(dialogue.get("conversation") or [], start=1):
        turns.append({
            "turn": index,
            "query": clean_text(turn.get("user")),
            "reference": clean_text(turn.get("assistant")),
            "expected_behavior": "answer",
            "requires_history": index > 1,
            "gold_doc_ids": list(dict.fromkeys(str(item) for item in turn.get("article") or [])),
            "gold_contexts": extract_article_contexts(turn.get("article_context") or []),
            "keywords": [clean_text(item) for item in turn.get("keyword") or [] if clean_text(item)],
        })
    return {
        "id": f"lexrag_{dialogue['id']}",
        "source_dialogue_id": dialogue["id"],
        "category": str(dialogue.get("type") or "unknown"),
        "turns": turns,
    }


def extract_article_contexts(items: List[Dict[str, Any]]) -> List[str]:
    contexts = []
    for item in items:
        for value in item.values():
            cleaned = clean_text(value)
            if cleaned and cleaned not in contexts:
                contexts.append(cleaned)
    return contexts


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\\n", " ").replace("\n", " ").split())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
