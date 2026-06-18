# -*- coding: utf-8 -*-

import asyncio
import os
import unittest

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.graph.nodes import query_nodes


class _FakeLLM:
    def __init__(self, route_type=None):
        self.route_type = route_type

    async def ainvoke(self, _messages):
        class _Response:
            pass

        response = _Response()
        if self.route_type:
            response.content = (
                '{"route_type": "%s", "route_reason": "fake", '
                '"need_retrieval": false, "need_web_search": false, '
                '"planned_tools": []}'
            ) % self.route_type
        else:
            response.content = "{}"
        return response


class QueryRoutingTest(unittest.TestCase):
    def setUp(self):
        self._original_llm = query_nodes.llm

    def tearDown(self):
        query_nodes.llm = self._original_llm

    def _route(self, query, kb_id=1, web_enabled=True, intent="analysis", llm_route=None):
        query_nodes.llm = _FakeLLM(llm_route)
        plan = query_nodes._build_sub_query_plan(
            question=query,
            kb_id=kb_id,
            web_enabled=web_enabled,
            query_intent=intent,
            fallback_queries=[query],
        )
        state = {
            "query": query,
            "query_rewritten": query,
            "query_intent": intent,
            "kb_id": kb_id,
            "web_enabled": web_enabled,
            "sub_questions": [],
            "retrieval_queries": [query],
            "sub_query_plans": [plan],
            "events": [],
        }
        return asyncio.run(query_nodes.plan_query_route(state))

    def test_chat_is_not_forced_to_knowledge_base_when_kb_selected(self):
        result = self._route("你能介绍一下你自己吗", kb_id=1, web_enabled=True, intent="chat", llm_route="knowledge_base")

        self.assertEqual(result["route_type"], "chat")
        self.assertFalse(result["need_retrieval"])
        self.assertFalse(result["need_web_search"])
        self.assertEqual(result["sub_query_plans"][0]["route_type"], "chat")

    def test_weather_uses_web_search_even_when_kb_selected(self):
        result = self._route("今天北京天气怎么样", kb_id=1, web_enabled=True, llm_route="knowledge_base")

        self.assertEqual(result["route_type"], "web_search")
        self.assertFalse(result["need_retrieval"])
        self.assertTrue(result["need_web_search"])
        self.assertEqual(result["sub_query_plans"][0]["route_type"], "web_search")

    def test_document_plus_latest_external_info_uses_hybrid(self):
        result = self._route("根据这份文档，结合最新行业政策分析影响", kb_id=1, web_enabled=True)

        self.assertEqual(result["route_type"], "hybrid")
        self.assertTrue(result["need_retrieval"])
        self.assertTrue(result["need_web_search"])
        self.assertEqual(result["sub_query_plans"][0]["route_type"], "hybrid")

    def test_plain_kb_question_still_uses_knowledge_base(self):
        result = self._route("总结一下这份文档的核心结论", kb_id=1, web_enabled=True)

        self.assertEqual(result["route_type"], "knowledge_base")
        self.assertTrue(result["need_retrieval"])
        self.assertFalse(result["need_web_search"])

    def test_assistant_style_request_is_not_forced_to_kb(self):
        result = self._route("帮我想一个项目名字", kb_id=1, web_enabled=False)

        self.assertEqual(result["route_type"], "chat")
        self.assertFalse(result["need_retrieval"])
        self.assertFalse(result["need_web_search"])

    def test_kb_scoped_assistant_request_still_uses_kb(self):
        result = self._route("根据这份文档帮我想一个项目标题", kb_id=1, web_enabled=False)

        self.assertEqual(result["route_type"], "knowledge_base")
        self.assertTrue(result["need_retrieval"])
        self.assertFalse(result["need_web_search"])


if __name__ == "__main__":
    unittest.main()
