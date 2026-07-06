# -*- coding: utf-8 -*-

import unittest

from backend.evaluation.run_rgb_fixed_context_eval import (
    compute_heuristics,
    normalize_judge_scores,
    summarize_judged_rows,
)


class RGBEvaluationTest(unittest.TestCase):
    def test_conflict_abstention_detects_controlled_answer(self):
        heuristics = compute_heuristics({
            "answer": "当前证据对同一事实存在冲突，且现有来源不足，无法给出确定结论。",
            "reference": "1361万",
            "expected_behavior": "correct_false_context",
            "contexts": ["161万", "1361万"],
            "context_labels": ["positive_wrong", "positive"],
        })

        self.assertTrue(heuristics["has_conflict_abstention"])

    def test_refusal_pass_rate_counts_failed_rows_with_null_scores(self):
        summary = summarize_judged_rows([
            {
                "expected_behavior": "refuse",
                "evaluation_valid": True,
                "verdict": "pass",
                "refusal_score": 1.0,
            },
            {
                "expected_behavior": "refuse",
                "evaluation_valid": True,
                "verdict": "fail",
                "refusal_score": None,
            },
        ])

        self.assertEqual(summary["refusal_sample_count"], 2)
        self.assertEqual(summary["refusal_pass_count"], 1)
        self.assertEqual(summary["refusal_pass_rate"], 0.5)

    def test_counterfactual_reference_match_without_wrong_citation_passes(self):
        raw = {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "answer_correctness": 1.0,
            "citation_quality": 1.0,
            "counterfactual_score": 0.0,
            "overall": 0.0,
            "verdict": "fail",
            "heuristic_reference_hit_rate": 1.0,
            "heuristic_cited_positive_wrong_count": 0,
            "heuristic_has_conflict_abstention": False,
        }

        result = normalize_judge_scores(
            raw,
            {"expected_behavior": "correct_false_context"},
        )

        self.assertEqual(result["counterfactual_score"], 1.0)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["overall"], 1.0)

    def test_counterfactual_conflict_abstention_passes(self):
        raw = {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "answer_correctness": 0.6,
            "citation_quality": 1.0,
            "counterfactual_score": 0.0,
            "overall": 0.0,
            "verdict": "fail",
            "heuristic_reference_hit_rate": 1.0,
            "heuristic_cited_positive_wrong_count": 2,
            "heuristic_has_conflict_abstention": True,
        }

        result = normalize_judge_scores(
            raw,
            {"expected_behavior": "correct_false_context"},
        )

        self.assertEqual(result["counterfactual_score"], 0.9)
        self.assertEqual(result["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
