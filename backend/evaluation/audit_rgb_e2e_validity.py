# -*- coding: utf-8 -*-
"""Audit RGB E2E samples for global-pool label and rejection contamination."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rgb"
    / "e2e_global_pool"
    / "rgb_zh_e2e_100.jsonl"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RGB E2E global-pool validity.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def normalize(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    all_contexts = [
        {
            "sample_id": row.get("id"),
            "question_type": row.get("question_type"),
            "text": normalize(context),
        }
        for row in rows
        for context in row.get("contexts") or []
    ]

    rejection_findings: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("question_type") != "negative_rejection":
            continue
        original_reference = normalize(row.get("original_ground_truth"))
        matches = [
            context
            for context in all_contexts
            if original_reference
            and context["sample_id"] != row.get("id")
            and original_reference in context["text"]
        ]
        rejection_findings.append({
            "id": row.get("id"),
            "question": row.get("question"),
            "original_reference": row.get("original_ground_truth"),
            "exact_cross_sample_leak": bool(matches),
            "matching_sample_ids": sorted({item["sample_id"] for item in matches}),
        })

    label_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        type_counts[question_type] = type_counts.get(question_type, 0) + 1
        for label in row.get("context_labels") or []:
            label = str(label)
            label_counts[label] = label_counts.get(label, 0) + 1

    leaked = sum(1 for item in rejection_findings if item["exact_cross_sample_leak"])
    report = {
        "dataset": "RGB",
        "mode": "e2e_global_pool_validity_audit",
        "input": str(args.input),
        "sample_count": len(rows),
        "type_counts": type_counts,
        "context_label_counts": label_counts,
        "rejection": {
            "sample_count": len(rejection_findings),
            "exact_cross_sample_leak_count": leaked,
            "exact_cross_sample_leak_rate": round(
                leaked / max(1, len(rejection_findings)),
                6,
            ),
            "formal_e2e_valid": leaked == 0,
            "note": (
                "Exact matching is a lower-bound audit. Semantic leakage may still exist and "
                "must be checked by the DeepSeek validity judge."
            ),
            "samples": rejection_findings,
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / f"rgb_e2e_validity_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "sample_count": len(rows),
        "rejection_exact_leak_count": leaked,
        "rejection_exact_leak_rate": report["rejection"]["exact_cross_sample_leak_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
