# -*- coding: utf-8 -*-
"""
User memory service.

This module manages user profile preferences and structured session memory.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.config import settings
from db.mysql.connection import get_db
from models.database.user import SessionTable, UserProfileTable

try:
    from graph.llm_factory import get_llm
except Exception:  # pragma: no cover - LLM may be unavailable in minimal envs
    get_llm = None

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - fallback for minimal envs
    jieba = None


class MemoryService:
    """Manage user profile data and structured session memory."""

    @staticmethod
    def _trim_text(text: str, max_length: int) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[: max_length - 1].rstrip() + "…"

    @staticmethod
    def _sanitize_title(title: str, max_length: int = 24) -> str:
        cleaned = " ".join((title or "").split())
        if not cleaned:
            return "新对话"
        return cleaned[:max_length]

    def _extract_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        if jieba is not None:
            keywords = jieba.lcut(text)
        else:
            keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{1,}", text)
        return [keyword for keyword in keywords if len(keyword) > 1]

    def _merge_unique_strings(self, original: Optional[List[str]], additions: List[str], limit: int) -> List[str]:
        merged = list(original or [])
        seen = {item.lower(): item for item in merged if isinstance(item, str)}
        for item in additions:
            cleaned = self._trim_text(item, 80)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen[lowered] = cleaned
            merged.append(cleaned)
        return merged[:limit]

    def _extract_working_preferences(self, query: str, answer: str) -> Dict[str, Any]:
        text = f"{query}\n{answer}".lower()
        preferences: Dict[str, Any] = {}

        if any(token in text for token in ["简短", "精简", "简洁", "concise"]):
            preferences["answer_style"] = "concise"
        elif any(token in text for token in ["详细", "展开", "细一点", "detailed"]):
            preferences["answer_style"] = "detailed"

        if any(token in text for token in ["中文", "汉语", "zh-cn", "中文回答"]):
            preferences["preferred_language"] = "zh-CN"
        elif any(token in text for token in ["english", "英文", "英语"]):
            preferences["preferred_language"] = "en-US"

        return preferences

    def _extract_long_term_facts(self, query: str) -> List[str]:
        candidates = []
        patterns = [
            r"我是(.{2,30})",
            r"我目前在做(.{2,30})",
            r"我主要关注(.{2,30})",
            r"我更喜欢(.{2,30})",
            r"我习惯(.{2,30})",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, query):
                candidates.append(self._trim_text(match, 60))
        return [item for item in candidates if item]

    def _build_topic_snapshot(self, query: str, answer: str, existing_topics: Optional[List[str]] = None) -> List[str]:
        keywords = self._extract_keywords(f"{query} {answer}")
        counts = Counter(keyword.lower() for keyword in keywords)
        ranked_keywords = []
        seen = set()
        for keyword in keywords:
            lowered = keyword.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ranked_keywords.append((keyword, counts[lowered]))
        ranked_keywords.sort(key=lambda item: (-item[1], len(item[0])))
        additions = [keyword for keyword, _ in ranked_keywords[:8]]
        return self._merge_unique_strings(existing_topics, additions, limit=20)

    def _build_session_summary_text(
        self,
        query: str,
        answer: str,
        working_memory: List[str],
        open_loops: List[str],
    ) -> str:
        summary_parts = [
            f"当前问题：{self._trim_text(query, 80)}",
            f"最近回答：{self._trim_text(answer, 120)}",
        ]
        if working_memory:
            summary_parts.append("会话要点：" + "；".join(self._trim_text(item, 50) for item in working_memory[:3]))
        if open_loops:
            summary_parts.append("待继续：" + "；".join(self._trim_text(item, 50) for item in open_loops[:2]))
        return " | ".join(part for part in summary_parts if part)

    def _build_recent_conversation_entry(self, query: str, answer: str) -> Dict[str, Any]:
        return {
            "query": self._trim_text(query, 200),
            "answer": self._trim_text(answer, 300),
            "query_summary": self._trim_text(query, 80),
            "answer_summary": self._trim_text(answer, 120),
            "timestamp": datetime.now().isoformat(),
        }

    def _normalize_session_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        current = context or {}
        recent_conversations = current.get("recent_conversations")
        if recent_conversations is None:
            recent_conversations = current.get("conversations") or []

        normalized = {
            "recent_conversations": recent_conversations[-8:],
            "working_memory": current.get("working_memory") or [],
            "open_loops": current.get("open_loops") or [],
            "last_answer_summary": current.get("last_answer_summary") or "",
            "topic_snapshot": current.get("topic_snapshot") or [],
        }
        return normalized

    def _update_session_context(self, context: Optional[Dict[str, Any]], query: str, answer: str) -> Dict[str, Any]:
        normalized = self._normalize_session_context(context)
        normalized["recent_conversations"] = (
            normalized.get("recent_conversations", []) + [self._build_recent_conversation_entry(query, answer)]
        )[-8:]

        normalized["last_answer_summary"] = self._trim_text(answer, 120)
        normalized["topic_snapshot"] = self._build_topic_snapshot(
            query=query,
            answer=answer,
            existing_topics=normalized.get("topic_snapshot") or [],
        )

        working_items = [
            f"用户最近在问：{self._trim_text(query, 80)}",
            f"上一轮回答重点：{self._trim_text(answer, 120)}",
        ]
        normalized["working_memory"] = self._merge_unique_strings(
            normalized.get("working_memory"),
            working_items,
            limit=6,
        )

        open_loops = normalized.get("open_loops") or []
        if any(token in query for token in ["继续", "接着", "进一步", "怎么优化", "还有"]):
            open_loops = self._merge_unique_strings(open_loops, [query], limit=4)
        normalized["open_loops"] = open_loops[:4]
        return normalized

    def compress_recent_conversations(self, conversations: List[Dict[str, Any]], query: str = "") -> List[str]:
        if not conversations:
            return []

        query_keywords = {item.lower() for item in self._extract_keywords(query)}
        scored: List[tuple[int, Dict[str, Any]]] = []
        for index, item in enumerate(conversations):
            text = f"{item.get('query_summary') or item.get('query', '')} {item.get('answer_summary') or item.get('answer', '')}"
            text_keywords = {keyword.lower() for keyword in self._extract_keywords(text)}
            overlap = len(query_keywords & text_keywords) if query_keywords else 0
            scored.append((overlap * 10 + index, item))

        scored.sort(key=lambda current: current[0], reverse=True)
        selected = [item for _, item in scored[:3]]
        selected.sort(key=lambda item: item.get("timestamp", ""))
        return [
            f"Q: {self._trim_text(item.get('query_summary') or item.get('query', ''), 80)} | A: {self._trim_text(item.get('answer_summary') or item.get('answer', ''), 100)}"
            for item in selected
        ]

    def build_prompt_memory_context(self, profile: Optional[Dict[str, Any]], session_payload: Optional[Dict[str, Any]], query: str) -> Dict[str, Any]:
        session_payload = session_payload or {}
        session_context = self._normalize_session_context((session_payload.get("context") if isinstance(session_payload, dict) else None) or {})
        recent_conversations = self.compress_recent_conversations(session_context.get("recent_conversations") or [], query=query)

        summary = self._trim_text(session_payload.get("summary") or "", 220) if isinstance(session_payload, dict) else ""
        working_memory = [self._trim_text(item, 100) for item in (session_context.get("working_memory") or [])[:4] if item]
        open_loops = [self._trim_text(item, 100) for item in (session_context.get("open_loops") or [])[:3] if item]
        topic_snapshot = [self._trim_text(item, 40) for item in (session_context.get("topic_snapshot") or [])[:6] if item]

        long_term_facts = []
        working_preferences = {}
        if profile:
            long_term_facts = [self._trim_text(item, 100) for item in (profile.get("long_term_facts") or [])[:6] if item]
            working_preferences = profile.get("working_preferences") or {}

        lines: List[str] = []
        if summary:
            lines.append(f"会话摘要：{summary}")
        if recent_conversations:
            lines.append("相关历史：" + "；".join(recent_conversations))
        if working_memory:
            lines.append("工作记忆：" + "；".join(working_memory))
        if open_loops:
            lines.append("待继续问题：" + "；".join(open_loops))
        if long_term_facts:
            lines.append("长期事实：" + "；".join(long_term_facts))
        if topic_snapshot:
            lines.append("近期主题：" + "、".join(topic_snapshot))
        if working_preferences:
            preference_parts = [f"{key}={value}" for key, value in working_preferences.items() if value]
            if preference_parts:
                lines.append("工作偏好：" + "；".join(preference_parts))

        return {
            "session_summary": summary,
            "recent_conversations": recent_conversations,
            "working_memory": working_memory,
            "open_loops": open_loops,
            "topic_snapshot": topic_snapshot,
            "long_term_facts": long_term_facts,
            "working_preferences": working_preferences,
            "prompt_context": "\n".join(lines),
        }

    def get_user_profile(self, user_id: int, db: Session) -> Optional[Dict[str, Any]]:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if not profile:
            return None

        return {
            "user_id": profile.user_id,
            "preferred_language": profile.preferred_language,
            "interests": profile.interests or [],
            "interaction_style": profile.interaction_style,
            "frequently_asked_topics": profile.frequently_asked_topics or [],
            "long_term_facts": profile.long_term_facts or [],
            "working_preferences": profile.working_preferences or {},
        }

    def create_or_update_profile(
        self,
        user_id: int,
        db: Session,
        preferred_language: str = "zh-CN",
        interests: Optional[List[str]] = None,
        interaction_style: str = "concise",
        frequently_asked_topics: Optional[List[str]] = None,
        long_term_facts: Optional[List[str]] = None,
        working_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if not profile:
            profile = UserProfileTable(user_id=user_id)
            db.add(profile)

        profile.preferred_language = preferred_language
        profile.interests = interests or []
        profile.interaction_style = interaction_style
        profile.frequently_asked_topics = frequently_asked_topics or []
        profile.long_term_facts = long_term_facts or []
        profile.working_preferences = working_preferences or {}

        db.commit()
        db.refresh(profile)

        return {
            "user_id": profile.user_id,
            "preferred_language": profile.preferred_language,
            "interests": profile.interests,
            "interaction_style": profile.interaction_style,
            "frequently_asked_topics": profile.frequently_asked_topics,
            "long_term_facts": profile.long_term_facts,
            "working_preferences": profile.working_preferences,
        }

    def generate_session_title(self, query: str, answer: str = "") -> str:
        """
        Generate a concise session title.
        Falls back to a trimmed user query if the LLM is unavailable.
        """
        fallback = self._sanitize_title(query)
        if not query or get_llm is None or not settings.LLM_API_KEY:
            return fallback

        try:
            llm = get_llm()
            response = llm.invoke(
                "请根据这段对话生成一个简短中文会话标题，不超过12个字，不要引号，不要句号。\n"
                f"用户问题：{query}\n"
                f"助手回答：{answer[:200]}"
            )
            return self._sanitize_title(getattr(response, "content", "") or "", max_length=20)
        except Exception:
            return fallback

    def update_interests_from_query(
        self,
        user_id: int,
        query: str,
        answer: str,
        db: Session,
    ) -> None:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if not profile:
            return

        interests = self._build_topic_snapshot(query=query, answer=answer, existing_topics=profile.interests or [])
        profile.interests = interests[:50]
        profile.frequently_asked_topics = self._build_topic_snapshot(
            query=query,
            answer=answer,
            existing_topics=profile.frequently_asked_topics or [],
        )[:20]
        db.commit()

    def update_long_term_memory(self, user_id: int, query: str, answer: str, db: Session) -> None:
        profile = db.query(UserProfileTable).filter(UserProfileTable.user_id == user_id).first()
        if not profile:
            return

        extracted_facts = self._extract_long_term_facts(query)
        profile.long_term_facts = self._merge_unique_strings(profile.long_term_facts, extracted_facts, limit=20)

        working_preferences = dict(profile.working_preferences or {})
        working_preferences.update(self._extract_working_preferences(query, answer))
        profile.working_preferences = working_preferences

        preferred_language = working_preferences.get("preferred_language")
        if preferred_language:
            profile.preferred_language = preferred_language

        answer_style = working_preferences.get("answer_style")
        if answer_style:
            profile.interaction_style = answer_style

        db.commit()

    def get_recent_sessions(
        self,
        user_id: int,
        limit: int = 5,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        owned_db = False
        if db is None:
            db = next(get_db())
            owned_db = True

        try:
            sessions = (
                db.query(SessionTable)
                .filter(SessionTable.user_id == user_id)
                .order_by(SessionTable.last_active.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "session_id": session.session_id,
                    "summary": session.summary,
                    "message_count": session.message_count,
                    "last_active": session.last_active.isoformat(),
                    "context": self._normalize_session_context(session.context),
                }
                for session in sessions
            ]
        finally:
            if owned_db:
                db.close()

    def create_session_summary(
        self,
        user_id: int,
        session_id: str,
        messages: List[Dict[str, Any]],
        db: Session,
    ) -> None:
        session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
        if not session:
            return

        first_user_message = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
        last_assistant_message = next((msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "assistant"), "")
        normalized = self._normalize_session_context(session.context)
        session.summary = self._build_session_summary_text(
            query=first_user_message,
            answer=last_assistant_message,
            working_memory=normalized.get("working_memory") or [],
            open_loops=normalized.get("open_loops") or [],
        )
        session.message_count = len(messages)
        session.last_active = datetime.now()
        db.commit()

    def get_session_memory(self, user_id: int, session_id: Optional[str], query: str, db: Session) -> Dict[str, Any]:
        profile = self.get_user_profile(user_id, db) or {}
        session_payload: Dict[str, Any] = {}
        if session_id:
            session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()
            if session and session.user_id == user_id:
                session_payload = {
                    "session_id": session.session_id,
                    "summary": session.summary,
                    "message_count": session.message_count,
                    "last_active": session.last_active.isoformat() if session.last_active else None,
                    "context": self._normalize_session_context(session.context),
                }

        prompt_memory = self.build_prompt_memory_context(profile=profile, session_payload=session_payload, query=query)
        return {
            "profile": profile,
            "session": session_payload,
            "preferred_language": profile.get("preferred_language", "zh-CN"),
            "interests": profile.get("interests") or [],
            "interaction_style": profile.get("interaction_style", "concise"),
            "frequently_asked_topics": profile.get("frequently_asked_topics") or [],
            **prompt_memory,
        }

    def save_conversation_to_memory(
        self,
        user_id: int,
        session_id: str,
        query: str,
        answer: str,
        db: Session,
    ) -> None:
        session = db.query(SessionTable).filter(SessionTable.session_id == session_id).first()

        if session:
            session.last_active = datetime.now()
            normalized_context = self._update_session_context(session.context, query=query, answer=answer)
            session.context = normalized_context
            session.summary = self._build_session_summary_text(
                query=query,
                answer=answer,
                working_memory=normalized_context.get("working_memory") or [],
                open_loops=normalized_context.get("open_loops") or [],
            )
            if not session.summary:
                session.summary = self.generate_session_title(query, answer)
            db.commit()

        self.update_interests_from_query(user_id, query, answer, db)
        self.update_long_term_memory(user_id, query, answer, db)


memory_service = MemoryService()
