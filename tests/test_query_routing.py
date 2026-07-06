# -*- coding: utf-8 -*-

import asyncio
import json
import os
import unittest

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

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

    def test_rewrite_generates_working_memory_from_short_context(self):
        class _RewriteLLM:
            def __init__(self):
                self.messages = []

            async def ainvoke(self, messages):
                self.messages = messages

                class _Response:
                    pass

                response = _Response()
                response.content = json.dumps({
                    "rewritten_query": "项目长期记忆如何分类和更新",
                    "search_intent": "procedure",
                    "missing_facets": [],
                    "working_memory": {
                        "current_task": "明确长期记忆分类和更新规则",
                        "constraints": ["记忆必须存数据库"],
                        "open_questions": ["如何处理记忆冲突"],
                        "active_entities": ["长期记忆", "数据库"],
                    },
                }, ensure_ascii=False)
                return response

        fake = _RewriteLLM()
        query_nodes.llm = fake
        state = {
            "query": "那长期记忆怎么分类和更新？",
            "memory_context": {
                "session_summary": {
                    "session_goal": "设计会话记忆系统",
                    "confirmed_decisions": ["记忆必须存数据库"],
                },
                "recent_messages": [
                    {"role": "user", "content": "短期记忆已经确定了"},
                    {"role": "assistant", "content": "接下来讨论长期记忆"},
                ],
                "working_memory": {
                    "current_task": "这份旧 working memory 不应进入改写输入",
                },
            },
            "events": [],
        }

        result = asyncio.run(query_nodes.rewrite_query(state))
        prompt = fake.messages[-1].content

        self.assertEqual(
            result["query_rewritten"],
            "项目长期记忆如何分类和更新",
        )
        self.assertEqual(
            result["memory_context"]["working_memory"]["current_task"],
            "明确长期记忆分类和更新规则",
        )
        self.assertEqual(
            result["memory_context"]["working_memory_generated_at"],
            "query_rewrite",
        )
        self.assertIn("记忆必须存数据库", prompt)
        self.assertIn("接下来讨论长期记忆", prompt)
        self.assertNotIn("旧 working memory", prompt)

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
