# -*- coding: utf-8 -*-

import asyncio
import unittest

from backend.graph.nodes import web_nodes


class _FailingWebSearchService:
    async def search_with_snippets(self, query: str, max_results: int = 5):
        raise RuntimeError("provider unavailable")


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


if __name__ == "__main__":
    unittest.main()
