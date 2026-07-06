# -*- coding: utf-8 -*-
"""Prepare small RGB Chinese fixed-context samples for generation evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RGB_DIR = PROJECT_ROOT / "data" / "evaluation" / "rgb" / "source" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "rgb" / "rgb_zh_fixed_samples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RGB zh files into project fixed-context eval samples.")
    parser.add_argument("--rgb-data-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-type", type=int, default=3)
    parser.add_argument("--noise-samples", type=int, default=None)
    parser.add_argument("--rejection-samples", type=int, default=None)
    parser.add_argument("--integration-samples", type=int, default=None)
    parser.add_argument("--counterfactual-samples", type=int, default=None)
    parser.add_argument("--distractor-samples", type=int, default=0)
    parser.add_argument("--distractor-offset", type=int, default=200)
    parser.add_argument("--distractor-passages", type=int, default=5)
    parser.add_argument("--passage-num", type=int, default=5)
    parser.add_argument("--noise-rate", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2333)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    counts = {
        "noise_robustness": args.noise_samples if args.noise_samples is not None else args.samples_per_type,
        "negative_rejection": args.rejection_samples if args.rejection_samples is not None else args.samples_per_type,
        "information_integration": args.integration_samples if args.integration_samples is not None else args.samples_per_type,
        "counterfactual_robustness": args.counterfactual_samples if args.counterfactual_samples is not None else args.samples_per_type,
    }
    samples: List[Dict[str, Any]] = []
    refine_items = load_jsonl(args.rgb_data_dir / "zh_refine.json")
    samples.extend(build_noise_samples(refine_items, counts["noise_robustness"], args.passage_num, args.noise_rate))
    samples.extend(build_rejection_samples(refine_items, counts["negative_rejection"], args.passage_num, offset=counts["noise_robustness"]))
    samples.extend(build_integration_samples(load_jsonl(args.rgb_data_dir / "zh_int.json"), counts["information_integration"], args.passage_num))
    samples.extend(build_counterfactual_samples(load_jsonl(args.rgb_data_dir / "zh_fact.json"), counts["counterfactual_robustness"], args.passage_num))
    samples.extend(build_distractor_samples(
        refine_items,
        args.distractor_samples,
        args.distractor_passages,
        args.distractor_offset,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(args.output),
        "sample_count": len(samples),
        "types": count_by_type(samples),
    }, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_noise_samples(items: List[Dict[str, Any]], limit: int, passage_num: int, noise_rate: float) -> List[Dict[str, Any]]:
    rows = []
    for item in items[:limit]:
        neg_num = min(len(item["negative"]), max(0, round(passage_num * noise_rate)))
        pos_num = min(len(item["positive"]), max(1, passage_num - neg_num))
        contexts = item["positive"][:pos_num] + item["negative"][: max(0, passage_num - pos_num)]
        labels = ["positive"] * pos_num + ["negative"] * max(0, len(contexts) - pos_num)
        rows.append(make_sample(item, "zh_refine", "noise_robustness", "answer", contexts, labels, positive_count=pos_num))
    return rows


def build_rejection_samples(items: List[Dict[str, Any]], limit: int, passage_num: int, offset: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    start = limit if offset is None else offset
    for item in items[start : start + limit]:
        contexts = item["negative"][:passage_num]
        labels = ["negative"] * len(contexts)
        sample = make_sample(item, "zh_refine", "negative_rejection", "refuse", contexts, labels, positive_count=0)
        sample["original_ground_truth"] = sample["ground_truth"]
        sample["ground_truth"] = "根据给定证据无法回答该问题，应明确说明信息不足，不编造答案。"
        rows.append(sample)
    return rows


def build_integration_samples(items: List[Dict[str, Any]], limit: int, passage_num: int) -> List[Dict[str, Any]]:
    rows = []
    for item in items[:limit]:
        contexts = []
        for group in item["positive"]:
            if group:
                contexts.append(group[0])
        contexts.extend(item["negative"][: max(0, passage_num - len(contexts))])
        context_slice = contexts[:passage_num]
        labels = ["positive"] * min(len(item["positive"]), len(context_slice))
        labels.extend(["negative"] * max(0, len(context_slice) - len(labels)))
        rows.append(make_sample(item, "zh_int", "information_integration", "answer", context_slice, labels, positive_count=len(item["positive"])))
    return rows


def build_counterfactual_samples(items: List[Dict[str, Any]], limit: int, passage_num: int) -> List[Dict[str, Any]]:
    rows = []
    for item in items[:limit]:
        wrong = item.get("positive_wrong") or []
        correct = item.get("positive") or []
        contexts = wrong[: max(1, min(3, passage_num - 1))]
        labels = ["positive_wrong"] * len(contexts)
        if correct:
            contexts.append(correct[0])
            labels.append("positive")
        negatives = (item.get("negative") or [])[: max(0, passage_num - len(contexts))]
        contexts.extend(negatives)
        labels.extend(["negative"] * len(negatives))
        rows.append(make_sample(item, "zh_fact", "counterfactual_robustness", "correct_false_context", contexts[:passage_num], labels[:passage_num], positive_count=len(correct[:1])))
    return rows


def build_distractor_samples(
    items: List[Dict[str, Any]],
    limit: int,
    passage_num: int,
    offset: int,
) -> List[Dict[str, Any]]:
    rows = []
    for item in items[offset : offset + max(0, limit)]:
        contexts = ((item.get("positive") or []) + (item.get("negative") or []))[:passage_num]
        labels = ["distractor"] * len(contexts)
        sample = make_sample(
            item,
            "zh_refine",
            "distractor",
            "not_evaluated",
            contexts,
            labels,
            positive_count=0,
        )
        sample["id"] = f"rgb_zh_refine_distractor_{item['id']}"
        rows.append(sample)
    return rows


def make_sample(
    item: Dict[str, Any],
    source_file: str,
    question_type: str,
    expected_behavior: str,
    contexts: List[str],
    context_labels: List[str],
    positive_count: int,
) -> Dict[str, Any]:
    answer = item.get("answer")
    if isinstance(answer, list):
        ground_truth = "；".join(str(value) for value in answer)
    else:
        ground_truth = str(answer)
    sample_id = f"rgb_{source_file}_{question_type}_{item['id']}"
    return {
        "id": sample_id,
        "question": item["query"],
        "ground_truth": ground_truth,
        "contexts": contexts,
        "context_labels": context_labels,
        "question_type": question_type,
        "expected_behavior": expected_behavior,
        "source_dataset": "RGB",
        "source_file": source_file,
        "rgb_id": item["id"],
        "positive_context_count": positive_count,
    }


def count_by_type(samples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        key = sample["question_type"]
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
