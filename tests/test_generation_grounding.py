# -*- coding: utf-8 -*-

import unittest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.graph.nodes import generation_nodes
from backend.graph.nodes import retrieval_nodes


class GenerationGroundingTest(unittest.IsolatedAsyncioTestCase):
    def test_support_grade_downgrades_direct_without_direct_ids(self):
        grade = retrieval_nodes._normalize_support_grade(
            {"status": "direct_support", "direct_evidence_ids": []},
            [{"evidence_id": "E1"}],
            web_enabled=False,
        )

        self.assertEqual(grade["status"], "background_only")
        self.assertEqual(grade["action"], "refusal")
        self.assertEqual(grade["allowed_citation_ids"], [])

    async def test_strict_refusal_does_not_call_generator(self):
        with patch.object(
            generation_nodes,
            "_generate_sub_answer",
            new=AsyncMock(side_effect=AssertionError("generator should not run")),
        ) as generate:
            result = await generation_nodes._generate_strict_sub_answer(
                sub_question="谁是导演？",
                route_type="knowledge_base",
                selected_evidence=[{"evidence_id": "E1", "support_snippet": "电影背景介绍"}],
                preferred_language="zh-CN",
                interaction_style="detailed",
                session_summary="",
                prompt_context="",
                working_memory=[],
                long_term_facts=[],
                evidence_grade={
                    "action": "refusal",
                    "allowed_citation_ids": [],
                    "missing_aspects": ["导演姓名"],
                },
                generation_mode="refusal",
            )

        generate.assert_not_awaited()
        self.assertIn("没有明确陈述", result["answer"])
        self.assertEqual(result["citations"], [])

    async def test_strict_generator_filters_background_evidence(self):
        generated = {
            "sub_question": "测试",
            "answer": "答案[E1]",
            "citations": [
                {"evidence_id": "E1"},
                {"evidence_id": "E2"},
            ],
            "confidence": 0.9,
            "reasoning_summary": "",
        }
        with patch.object(
            generation_nodes,
            "_generate_sub_answer",
            new=AsyncMock(return_value=generated),
        ) as generate:
            result = await generation_nodes._generate_strict_sub_answer(
                sub_question="测试",
                route_type="knowledge_base",
                selected_evidence=[
                    {"evidence_id": "E1", "support_snippet": "直接答案"},
                    {"evidence_id": "E2", "support_snippet": "背景材料"},
                ],
                preferred_language="zh-CN",
                interaction_style="detailed",
                session_summary="",
                prompt_context="",
                working_memory=[],
                long_term_facts=[],
                evidence_grade={
                    "action": "normal_answer",
                    "allowed_citation_ids": ["E1"],
                },
                generation_mode="normal_answer",
            )

        passed_evidence = generate.await_args.args[2]
        self.assertEqual([item["evidence_id"] for item in passed_evidence], ["E1"])
        self.assertEqual(result["citations"], [{"evidence_id": "E1"}])

    def test_citations_require_explicit_model_reference(self):
        evidence = [{
            "evidence_id": "E1",
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "support_snippet": "support",
        }]

        self.assertEqual(generation_nodes._build_citations(evidence, []), [])

    async def test_insufficient_plan_refuses_without_calling_llm(self):
        state = {
            "query": "测试问题",
            "route_type": "knowledge_base",
            "evidence_sufficient": False,
            "selected_evidence": [{
                "evidence_id": "E1",
                "doc_id": "doc-1",
                "support_snippet": "相关但不足的证据",
            }],
            "sub_query_plans": [{
                "sub_question": "测试问题",
                "route_type": "knowledge_base",
                "evidence_sufficient": False,
                "selected_evidence": [{
                    "evidence_id": "E1",
                    "doc_id": "doc-1",
                    "support_snippet": "相关但不足的证据",
                }],
            }],
            "memory_context": {},
            "events": [],
        }

        with patch.object(
            generation_nodes,
            "_generate_sub_answer",
            new=AsyncMock(side_effect=AssertionError("LLM should not be called")),
        ) as generate:
            result = await generation_nodes.generate_answer(state)

        generate.assert_not_awaited()
        self.assertIn("证据不足", result["final_answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["confidence"], 0.0)

    async def test_sufficient_plan_still_uses_generator(self):
        state = {
            "query": "测试问题",
            "route_type": "knowledge_base",
            "evidence_sufficient": True,
            "selected_evidence": [{
                "evidence_id": "E1",
                "doc_id": "doc-1",
                "support_snippet": "充分证据",
            }],
            "sub_query_plans": [{
                "sub_question": "测试问题",
                "route_type": "knowledge_base",
                "evidence_sufficient": True,
                "selected_evidence": [{
                    "evidence_id": "E1",
                    "doc_id": "doc-1",
                    "support_snippet": "充分证据",
                }],
            }],
            "memory_context": {},
            "events": [],
        }
        generated = {
            "sub_question": "测试问题",
            "answer": "答案 [E1]",
            "citations": [{"evidence_id": "E1", "doc_id": "doc-1"}],
            "confidence": 0.9,
            "reasoning_summary": "E1",
        }

        with patch.object(
            generation_nodes,
            "_generate_sub_answer",
            new=AsyncMock(return_value=generated),
        ) as generate:
            result = await generation_nodes.generate_answer(state)

        generate.assert_awaited_once()
        self.assertEqual(result["final_answer"], "答案 [E1]")
        self.assertEqual(result["citations"], generated["citations"])


if __name__ == "__main__":
    unittest.main()
