# -*- coding: utf-8 -*-

import asyncio
import unittest

from backend.graph.nodes import web_nodes


class _FailingWebSearchService:
    async def search_with_snippets(self, query: str, max_results: int = 5):
        raise RuntimeError("provider unavailable")


class _SuccessfulWebSearchService:
    async def search_with_snippets(self, query: str, max_results: int = 5):
        return [{
            "title": "权威来源",
            "url": "https://example.com/result",
            "snippet": "联网证据",
            "score": 0.95,
        }]


class WebSearchNodeTest(unittest.TestCase):
    def setUp(self):
        self._original_service = web_nodes.web_search_service

    def tearDown(self):
        web_nodes.web_search_service = self._original_service

    def test_disabled_web_search_closes_pending_web_plans(self):
        state = {
            "web_enabled": False,
            "events": [],
            "sub_query_plans": [{
                "sub_question": "今天北京天气怎么样",
                "route_type": "web_search",
                "need_web_search": True,
                "need_retrieval": False,
                "selected_evidence": [],
            }],
        }

        result = asyncio.run(web_nodes.web_search(state))

        self.assertFalse(result["need_web_search"])
        self.assertFalse(result["sub_query_plans"][0]["need_web_search"])
        self.assertFalse(result["sub_query_plans"][0]["need_retrieval"])
        self.assertEqual(result["events"][-1]["event"], "websearch.skipped")

    def test_provider_failure_closes_pending_web_plans(self):
        web_nodes.web_search_service = _FailingWebSearchService()
        state = {
            "web_enabled": True,
            "events": [],
            "sub_query_plans": [{
                "sub_question": "今天北京天气怎么样",
                "route_type": "web_search",
                "retrieval_queries": ["今天北京天气"],
                "need_web_search": True,
                "need_retrieval": False,
                "selected_evidence": [],
            }],
        }

        result = asyncio.run(web_nodes.web_search(state))

        self.assertTrue(result["used_web_search"])
        self.assertFalse(result["need_web_search"])
        self.assertFalse(result["sub_query_plans"][0]["need_web_search"])
        self.assertEqual(result["events"][-1]["event"], "websearch.failed")

    def test_web_evidence_is_prioritized_for_downstream_evaluation(self):
        web_nodes.web_search_service = _SuccessfulWebSearchService()
        state = {
            "web_enabled": True,
            "events": [],
            "sub_query_plans": [{
                "sub_question": "冲突事实",
                "route_type": "hybrid",
                "retrieval_queries": ["冲突事实"],
                "need_web_search": True,
                "need_retrieval": False,
                "selected_evidence": [{
                    "evidence_id": "E1",
                    "source_type": "knowledge_base",
                    "support_snippet": "知识库证据",
                }],
            }],
        }

        result = asyncio.run(web_nodes.web_search(state))

        evidence = result["sub_query_plans"][0]["selected_evidence"]
        self.assertEqual(evidence[0]["source_type"], "web")
        self.assertEqual(evidence[1]["source_type"], "knowledge_base")
        self.assertTrue(result["used_web_search"])


if __name__ == "__main__":
    unittest.main()
