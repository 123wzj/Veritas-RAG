# -*- coding: utf-8 -*-

import asyncio
import unittest

from backend.graph.nodes import reflection_nodes


class ReflectionRoutingTest(unittest.TestCase):
    def test_route_reflection_stops_at_max_reflections(self):
        state = {
            "reflection_count": 3,
            "max_reflections": 3,
            "step_count": 0,
            "max_steps": 8,
            "sub_query_plans": [{"need_retrieval": True}],
        }

        route = asyncio.run(reflection_nodes.route_reflection(state))

        self.assertEqual(route, "proceed")

    def test_route_reflection_can_retry_before_max_reflections(self):
        state = {
            "reflection_count": 2,
            "max_reflections": 3,
            "step_count": 0,
            "max_steps": 8,
            "sub_query_plans": [{"need_retrieval": True}],
        }

        route = asyncio.run(reflection_nodes.route_reflection(state))

        self.assertEqual(route, "retrieve")

    def test_route_evidence_generates_when_reflection_budget_exhausted(self):
        state = {
            "reflection_count": 3,
            "max_reflections": 3,
            "evidence_sufficient": False,
            "need_reflection": True,
            "need_web_search": False,
            "sub_query_plans": [{"need_retrieval": True}],
        }

        route = asyncio.run(reflection_nodes.route_evidence(state))

        self.assertEqual(route, "generate")

    def test_route_evidence_keeps_web_search_before_reflection_budget(self):
        state = {
            "reflection_count": 3,
            "max_reflections": 3,
            "evidence_sufficient": False,
            "need_reflection": True,
            "need_web_search": True,
            "sub_query_plans": [{"need_web_search": True}],
        }

        route = asyncio.run(reflection_nodes.route_evidence(state))

        self.assertEqual(route, "web_search")


if __name__ == "__main__":
    unittest.main()
