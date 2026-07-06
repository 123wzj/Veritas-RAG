# -*- coding: utf-8 -*-

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.core.config import settings
from backend.db.mysql.connection import Base
from backend.graph.llm_factory import get_llm
from backend.models.database.user import (
    ConversationBranchTable,
    LongTermMemoryTable,
    MemoryUpdateLogTable,
    MessageTable,
    SessionMemoryTable,
    SessionTable,
    UserProfileTable,
    UserTable,
)
from backend.services.context.context_assembler import ContextAssembler
from backend.services.chat.turn_service import TurnService
from backend.graph.nodes import memory_nodes
import backend.services.memory.memory_service as memory_module
from backend.services.memory.memory_service import MemoryService


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, _messages):
        class _Response:
            pass

        response = _Response()
        response.content = json.dumps(self.payload, ensure_ascii=False)
        return response


class _CapturingFakeLLM(_FakeLLM):
    def __init__(self, payload):
        super().__init__(payload)
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return await super().ainvoke(messages)


class MemoryContextEngineeringTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        tables = [
            UserTable.__table__,
            UserProfileTable.__table__,
            SessionTable.__table__,
            ConversationBranchTable.__table__,
            MessageTable.__table__,
            SessionMemoryTable.__table__,
            LongTermMemoryTable.__table__,
            MemoryUpdateLogTable.__table__,
        ]
        Base.metadata.create_all(engine, tables=tables)
        self.SessionLocal = sessionmaker(bind=engine)
        self.db = self.SessionLocal()
        self.service = MemoryService()
        self.db.add(UserTable(
            id=1,
            username="memory-test",
            hashed_password="test",
            is_active=True,
        ))
        self.db.add(UserProfileTable(
            user_id=1,
            preferred_language="zh-CN",
            interaction_style="detailed",
            long_term_facts=[],
        ))
        self.db.add(SessionTable(
            session_id="session-1",
            user_id=1,
            kb_id=7,
            title="新对话",
            summary="legacy title",
            context={},
            message_count=0,
        ))
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_deepseek_v4_model_tiers_are_explicit(self):
        self.assertEqual(get_llm("flash").model_name, "deepseek-v4-flash")
        self.assertEqual(get_llm("pro").model_name, "deepseek-v4-pro")

    def test_context_assembler_respects_total_budget(self):
        assembler = ContextAssembler()
        result = assembler.assemble(
            query="解释当前项目的长期记忆策略",
            recent_messages=[
                {"role": "user", "content": "历史问题" * 200},
                {"role": "assistant", "content": "历史回答" * 200},
            ],
            session_summary={
                "session_goal": "设计上下文工程",
                "confirmed_decisions": ["数据库作为事实源"] * 20,
            },
            working_memory={"current_task": "实现动态上下文" * 40},
            long_term_memories=[{
                "memory_id": "m1",
                "scope_type": "project",
                "memory_type": "project_decision",
                "content": "长期记忆存数据库" * 100,
            }],
            evidence=[{
                "evidence_id": "E1",
                "title": "设计文档",
                "snippet": "证据内容" * 300,
            }],
            total_budget=320,
        )

        self.assertLessEqual(result["token_usage"]["used"], 320)
        self.assertEqual(result["token_usage"]["budget"], 320)
        self.assertIn("当前问题", result["text"])

    def test_typed_generation_prompt_loads_required_sections_once(self):
        assembler = ContextAssembler()
        result = assembler.assemble_typed(
            "answer_generation",
            values={
                "query": "当前问题",
                "evidence": [{
                    "evidence_id": "E1",
                    "title": "证据标题",
                    "content": "唯一证据内容",
                }],
                "session_summary": {"session_goal": "唯一摘要内容"},
                "long_term_memories": [{
                    "memory_id": "M1",
                    "scope_type": "project",
                    "memory_type": "project_decision",
                    "content": "唯一长期记忆内容",
                }],
            },
            required_sections={"query", "evidence"},
            total_budget=1200,
        )

        self.assertEqual(result["missing_required_sections"], [])
        self.assertEqual(result["text"].count("唯一证据内容"), 1)
        self.assertEqual(result["text"].count("唯一长期记忆内容"), 1)
        self.assertLess(
            result["text"].index("唯一证据内容"),
            result["text"].index("唯一长期记忆内容"),
        )
        self.assertEqual(result["selected_evidence_ids"], ["E1"])
        self.assertEqual(result["selected_memory_ids"], ["M1"])

    def test_recent_messages_excludes_current_request(self):
        for index in range(4):
            request_id = f"r-{index}"
            self.db.add(MessageTable(
                session_id="session-1",
                role="user",
                content=f"question-{index}",
                request_id=request_id,
            ))
            self.db.add(MessageTable(
                session_id="session-1",
                role="assistant",
                content=f"answer-{index}",
                request_id=request_id,
            ))
        self.db.add(MessageTable(
            session_id="session-1",
            role="user",
            content="current question",
            request_id="current",
        ))
        self.db.commit()

        messages = self.service.get_recent_messages(
            "session-1",
            self.db,
            current_request_id="current",
            turns=3,
        )

        self.assertEqual(len(messages), 6)
        self.assertNotIn("current question", [item["content"] for item in messages])
        self.assertEqual(messages[0]["content"], "question-1")

    def test_short_term_load_defers_long_term_memory_selection(self):
        self.db.add(LongTermMemoryTable(
            memory_id="deferred-memory",
            user_id=1,
            kb_id=7,
            scope_type="project",
            memory_type="project_constraint",
            content="The project database is MySQL.",
            normalized_key="project-database",
            keywords=["project", "database", "mysql"],
            confidence=0.95,
            status="active",
        ))
        self.db.commit()

        payload = self.service.get_short_term_memory(
            user_id=1,
            session_id="session-1",
            query="Which database does this project use?",
            db=self.db,
        )

        self.assertFalse(payload["long_term_selection_complete"])
        self.assertEqual(payload["selected_memory_ids"], [])
        self.assertEqual(payload["long_term_memories"], [])

    def test_llm_controls_long_term_merge_plan(self):
        self.db.add(SessionMemoryTable(
            session_id="session-1",
            summary={},
            version=1,
        ))
        self.db.add(LongTermMemoryTable(
            memory_id="memory-1",
            user_id=1,
            kb_id=7,
            scope_type="project",
            memory_type="project_decision",
            content="长期记忆暂时存文件",
            normalized_key="长期记忆存储",
            keywords=["长期记忆", "存储"],
            confidence=0.8,
            status="active",
        ))
        self.db.commit()

        payload = {
            "session_summary": {
                "session_goal": "设计记忆系统",
                "confirmed_decisions": ["长期记忆存数据库"],
                "discarded_ideas": ["长期记忆存文件"],
                "open_questions": [],
            },
            "long_term_actions": [{
                "action": "merge",
                "target_memory_id": "memory-1",
                "scope_type": "project",
                "memory_type": "project_decision",
                "content": "长期记忆统一存数据库",
                "normalized_key": "长期记忆存储",
                "keywords": ["长期记忆", "数据库"],
                "confidence": 0.95,
                "reason": "用户明确纠正",
            }],
            "title": "记忆系统设计",
            "reason": "更新已确认方案",
        }
        original_get_llm = memory_module.get_llm
        original_key = settings.LLM_API_KEY
        memory_module.get_llm = lambda *_args, **_kwargs: _FakeLLM(payload)
        settings.LLM_API_KEY = "test-key"
        try:
            plan = asyncio.run(self.service.build_memory_update_plan(
                user_id=1,
                session_id="session-1",
                query="不对，长期记忆要存数据库",
                answer="已按数据库方案调整。",
                request_id="request-1",
                kb_id=7,
                db=self.db,
            ))
        finally:
            memory_module.get_llm = original_get_llm
            settings.LLM_API_KEY = original_key

        self.assertTrue(plan["model_controlled"])
        self.assertEqual(plan["long_term_actions"][0]["action"], "merge")
        self.assertEqual(
            plan["session_summary"]["discarded_ideas"],
            ["长期记忆存文件"],
        )

    def test_summary_prompt_contains_previous_summary_and_recent_turns(self):
        previous_summary = {
            "session_goal": "设计上下文工程",
            "confirmed_decisions": ["消息写入数据库"],
            "discarded_ideas": [],
            "open_questions": ["摘要如何更新"],
        }
        self.db.add(SessionMemoryTable(
            session_id="session-1",
            summary=previous_summary,
            version=4,
        ))
        for index in range(2):
            self.db.add(MessageTable(
                session_id="session-1",
                role="user",
                content=f"最近问题-{index}",
                request_id=f"summary-{index}",
            ))
            self.db.add(MessageTable(
                session_id="session-1",
                role="assistant",
                content=f"最近回答-{index}",
                request_id=f"summary-{index}",
            ))
        self.db.commit()

        long_goal = "完整摘要" * 500
        fake = _CapturingFakeLLM({
            "session_summary": {
                "session_goal": long_goal,
                "confirmed_decisions": ["保留旧结论", "加入新结论"],
                "discarded_ideas": [],
                "open_questions": [],
            },
            "long_term_actions": [],
            "title": "上下文工程",
            "reason": "增量压缩",
        })
        original_get_llm = memory_module.get_llm
        original_key = settings.LLM_API_KEY
        memory_module.get_llm = lambda *_args, **_kwargs: fake
        settings.LLM_API_KEY = "test-key"
        try:
            plan = asyncio.run(self.service.build_memory_update_plan(
                user_id=1,
                session_id="session-1",
                query="继续调整摘要",
                answer="已继续调整。",
                request_id="summary-current",
                kb_id=7,
                db=self.db,
            ))
        finally:
            memory_module.get_llm = original_get_llm
            settings.LLM_API_KEY = original_key

        prompt = fake.messages[-1].content
        self.assertIn("消息写入数据库", prompt)
        self.assertIn("最近问题-0", prompt)
        self.assertIn("最近回答-1", prompt)
        self.assertIn("summary_target_chars", prompt)
        self.assertTrue(plan["summary_updated"])
        self.assertEqual(plan["base_memory_version"], 4)
        self.assertEqual(plan["session_summary"]["session_goal"], long_goal)

    def test_llm_selects_only_relevant_long_term_memory_ids(self):
        candidates = [
            {
                "memory_id": "m1",
                "scope_type": "user",
                "memory_type": "user_preference",
                "content": "用户喜欢简短回答",
                "confidence": 0.9,
            },
            {
                "memory_id": "m2",
                "scope_type": "project",
                "memory_type": "project_constraint",
                "content": "当前项目必须使用 MySQL",
                "confidence": 0.95,
            },
        ]
        original_get_llm = memory_module.get_llm
        original_key = settings.LLM_API_KEY
        memory_module.get_llm = lambda *_args, **_kwargs: _FakeLLM({
            "selected_memory_ids": ["m2", "unknown"],
            "reason": "当前问题询问项目数据库约束",
        })
        settings.LLM_API_KEY = "test-key"
        try:
            selected = asyncio.run(self.service._llm_select_long_term_memories(
                query="这个项目数据库怎么选？",
                short_context="当前任务是确认数据库约束",
                candidates=candidates,
            ))
        finally:
            memory_module.get_llm = original_get_llm
            settings.LLM_API_KEY = original_key

        self.assertEqual(selected, ["m2"])

    def test_apply_plan_is_idempotent_and_keeps_title_separate(self):
        plan = {
            "request_id": "request-2",
            "user_id": 1,
            "session_id": "session-1",
            "kb_id": 7,
            "title": "上下文工程",
            "reason": "test",
            "session_summary": {
                "session_goal": "完善上下文工程",
                "confirmed_decisions": ["消息存数据库"],
                "discarded_ideas": [],
                "open_questions": [],
            },
            "long_term_actions": [{
                "action": "create",
                "scope_type": "project",
                "kb_id": 7,
                "memory_type": "project_constraint",
                "content": "聊天记录必须存数据库",
                "normalized_key": "聊天记录存储",
                "keywords": ["聊天记录", "数据库"],
                "confidence": 0.95,
                "reason": "用户明确要求",
            }],
        }

        first = self.service.apply_memory_update_plan(
            plan,
            db=self.db,
            assistant_message_id=42,
        )
        self.db.commit()
        second = self.service.apply_memory_update_plan(
            plan,
            db=self.db,
            assistant_message_id=42,
        )
        duplicate_plan = dict(plan)
        duplicate_plan["request_id"] = "request-3"
        duplicate_result = self.service.apply_memory_update_plan(
            duplicate_plan,
            db=self.db,
            assistant_message_id=43,
        )
        self.db.commit()

        session = self.db.query(SessionTable).filter_by(session_id="session-1").one()
        memory = self.db.query(SessionMemoryTable).filter_by(session_id="session-1").one()
        long_term_count = self.db.query(LongTermMemoryTable).count()

        self.assertTrue(first["applied"])
        self.assertTrue(second["idempotent"])
        self.assertEqual(
            duplicate_result["actions"][0]["action"],
            "merge_duplicate",
        )
        self.assertEqual(session.title, "上下文工程")
        self.assertEqual(session.summary, "legacy title")
        self.assertEqual(memory.summary["session_goal"], "完善上下文工程")
        self.assertEqual(long_term_count, 1)

    def test_stale_plan_rebases_summary_without_moving_summary_cursor_back(self):
        self.db.add(SessionMemoryTable(
            session_id="session-1",
            summary={
                "session_goal": "当前目标",
                "confirmed_decisions": ["较新结论"],
                "discarded_ideas": [],
                "open_questions": [],
            },
            summary_through_message_id=100,
            version=5,
        ))
        self.db.commit()
        plan = {
            "request_id": "stale-request",
            "user_id": 1,
            "session_id": "session-1",
            "kb_id": 7,
            "base_memory_version": 4,
            "summary_updated": True,
            "session_summary": {
                "session_goal": "当前目标",
                "confirmed_decisions": ["较早并发结论"],
                "discarded_ideas": [],
                "open_questions": [],
            },
            "long_term_actions": [],
            "title": "并发摘要",
            "reason": "test",
        }

        result = self.service.apply_memory_update_plan(
            plan,
            db=self.db,
            assistant_message_id=90,
        )
        self.db.commit()
        memory = self.db.query(SessionMemoryTable).filter_by(
            session_id="session-1"
        ).one()

        self.assertTrue(result["applied"])
        self.assertEqual(
            memory.summary["confirmed_decisions"],
            ["较新结论", "较早并发结论"],
        )
        self.assertEqual(memory.summary_through_message_id, 100)

    def test_cross_project_long_term_target_is_not_mutated(self):
        self.db.add(LongTermMemoryTable(
            memory_id="other-project-memory",
            user_id=1,
            kb_id=8,
            scope_type="project",
            memory_type="project_constraint",
            content="另一个项目使用 SQLite",
            normalized_key="database",
            keywords=["database"],
            confidence=0.9,
            status="active",
        ))
        self.db.commit()
        plan = {
            "request_id": "cross-project-request",
            "user_id": 1,
            "session_id": "session-1",
            "kb_id": 7,
            "summary_updated": False,
            "session_summary": {},
            "allowed_target_memory_ids": ["other-project-memory"],
            "long_term_actions": [{
                "action": "merge",
                "target_memory_id": "other-project-memory",
                "scope_type": "project",
                "kb_id": 7,
                "memory_type": "project_constraint",
                "content": "当前项目使用 MySQL",
                "normalized_key": "database",
                "keywords": ["database"],
                "confidence": 0.95,
            }],
            "title": "作用域测试",
            "reason": "test",
        }

        result = self.service.apply_memory_update_plan(plan, db=self.db)
        self.db.commit()
        target = self.db.query(LongTermMemoryTable).filter_by(
            memory_id="other-project-memory"
        ).one()

        self.assertEqual(result["actions"], [])
        self.assertEqual(target.content, "另一个项目使用 SQLite")
        self.assertEqual(target.kb_id, 8)

    def test_failed_verification_skips_memory_update_planning(self):
        fake_db = MagicMock()
        planner = AsyncMock()
        state = {
            "user_id": 1,
            "session_id": "session-1",
            "request_id": "failed-verification",
            "query": "问题",
            "final_answer": "不可靠回答",
            "verification": {
                "grounded": False,
                "useful": False,
            },
            "events": [],
        }
        with patch.object(memory_nodes, "get_db", return_value=iter([fake_db])), patch.object(
            memory_nodes.memory_service,
            "build_memory_update_plan",
            planner,
        ):
            result = asyncio.run(memory_nodes.write_memory(state))

        planner.assert_not_awaited()
        self.assertIsNone(result["memory_update_plan"])
        self.assertEqual(result["events"][-1]["event"], "memory.update.skipped")

    def test_long_term_selection_preserves_query_stage_working_memory(self):
        fake_db = MagicMock()
        selector = AsyncMock(return_value={
            "working_memory": {"current_task": "数据库中的旧任务"},
            "long_term_selection_complete": True,
            "long_term_selection_query": "改写后的问题",
            "selected_memory_ids": [],
            "context_token_usage": {},
        })
        state = {
            "user_id": 1,
            "session_id": "session-1",
            "request_id": "working-draft",
            "query": "原问题",
            "query_rewritten": "改写后的问题",
            "sub_query_plans": [],
            "kb_id": 7,
            "memory_context": {
                "working_memory": {"current_task": "阶段二生成的新任务"},
                "working_memory_generated_at": "query_rewrite",
                "long_term_selection_complete": False,
            },
            "events": [],
        }
        with patch.object(memory_nodes, "get_db", return_value=iter([fake_db])), patch.object(
            memory_nodes.memory_service,
            "aget_session_memory",
            selector,
        ):
            result = asyncio.run(memory_nodes.load_generation_memory(state))

        self.assertEqual(
            result["memory_context"]["working_memory"]["current_task"],
            "阶段二生成的新任务",
        )
        self.assertEqual(
            result["memory_context"]["working_memory_generated_at"],
            "query_rewrite",
        )

    def test_turn_service_persists_answer_and_memory_in_one_finalize_step(self):
        turn_service = TurnService()
        started = turn_service.begin_turn(
            db=self.db,
            user_id=1,
            session_id="session-2",
            kb_id=7,
            request_id="turn-request",
            query="记住项目必须使用数据库",
        )
        plan = {
            "request_id": "turn-request",
            "user_id": 1,
            "session_id": "session-2",
            "kb_id": 7,
            "title": "数据库约束",
            "reason": "用户明确要求",
            "session_summary": {
                "session_goal": "确定项目约束",
                "confirmed_decisions": ["项目使用数据库"],
                "discarded_ideas": [],
                "open_questions": [],
            },
            "long_term_actions": [],
        }

        finalized = turn_service.finalize_turn(
            db=self.db,
            user_id=1,
            session_id=started["session"].session_id,
            request_id="turn-request",
            answer="已记录该项目约束。",
            citations=[],
            memory_update_plan=plan,
        )
        retried = turn_service.finalize_turn(
            db=self.db,
            user_id=1,
            session_id=started["session"].session_id,
            request_id="turn-request",
            answer="已记录该项目约束。",
            citations=[],
            memory_update_plan=plan,
        )

        session = self.db.query(SessionTable).filter_by(session_id="session-2").one()
        messages = self.db.query(MessageTable).filter_by(session_id="session-2").all()
        session_memory = (
            self.db.query(SessionMemoryTable)
            .filter_by(session_id="session-2")
            .one()
        )

        self.assertFalse(finalized["idempotent"])
        self.assertTrue(retried["idempotent"])
        self.assertEqual(session.message_count, 2)
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            session_memory.summary["confirmed_decisions"],
            ["项目使用数据库"],
        )


if __name__ == "__main__":
    unittest.main()
