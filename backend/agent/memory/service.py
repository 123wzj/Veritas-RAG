"""Bridge existing durable memory into the ReAct three-layer contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.agent.schemas import AnswerSlot, WorkingMemoryV2
from backend.models.database.user import (
    ConversationBranchTable,
    SessionMemoryTable,
    SessionTable,
    UserProfileTable,
)
from backend.services.memory.memory_service import memory_service


class ReactMemoryService:
    async def load(
        self,
        *,
        run_id: str,
        user_id: int,
        session_id: str,
        kb_id: Optional[int],
        query: str,
        request_id: str,
        db: Session,
        branch_id: Optional[int] = None,
        read_only: bool = True,
    ) -> Dict[str, Any]:
        session = (
            db.query(SessionTable)
            .filter(
                SessionTable.session_id == session_id,
                SessionTable.user_id == user_id,
            )
            .first()
        )
        if not session:
            raise PermissionError("Session is not owned by current user")
        if branch_id is not None:
            branch = (
                db.query(ConversationBranchTable)
                .filter(
                    ConversationBranchTable.id == branch_id,
                    ConversationBranchTable.session_id == session_id,
                )
                .first()
            )
            if not branch:
                raise PermissionError("Branch is not part of current session")

        if not read_only:
            return await memory_service.aget_session_memory(
                user_id=user_id,
                session_id=session_id,
                query=query,
                db=db,
                kb_id=kb_id,
                current_request_id=request_id,
                working_memory={},
                branch_id=branch_id,
            )

        profile_row = (
            db.query(UserProfileTable)
            .filter(UserProfileTable.user_id == user_id)
            .first()
        )
        profile = {
            "user_id": user_id,
            "preferred_language": getattr(profile_row, "preferred_language", "zh-CN"),
            "interests": getattr(profile_row, "interests", None) or [],
            "interaction_style": getattr(profile_row, "interaction_style", "detailed"),
            "frequently_asked_topics": getattr(profile_row, "frequently_asked_topics", None) or [],
            "working_preferences": getattr(profile_row, "working_preferences", None) or {},
        }
        session_memory = (
            db.query(SessionMemoryTable)
            .filter(SessionMemoryTable.session_id == session_id)
            .first()
        )
        recent_messages = memory_service.get_recent_messages(
            session_id,
            db,
            current_request_id=request_id,
            branch_id=branch_id,
        )
        resolved_kb_id = kb_id if kb_id is not None else session.kb_id
        candidates = memory_service._rank_long_term_candidates(
            user_id=user_id,
            kb_id=resolved_kb_id,
            query=query,
            db=db,
        )
        selected_ids = await memory_service._llm_select_long_term_memories(
            query=query,
            candidates=candidates,
            recent_messages=recent_messages,
            session_summary=session_memory.summary if session_memory else {},
            working_memory={},
        )
        selected_set = set(selected_ids)
        selected = [item for item in candidates if item["memory_id"] in selected_set]
        selected.sort(key=lambda item: selected_ids.index(item["memory_id"]))
        return memory_service._build_memory_payload(
            profile=profile,
            session=session,
            session_memory=None if branch_id is not None else session_memory,
            recent_messages=recent_messages,
            selected_memories=selected,
            query=query,
            working_memory={},
        )

    @staticmethod
    def initial_working_memory(*, run_id: str, query: str) -> Dict[str, Any]:
        slot = AnswerSlot(id="slot-1", question=query, required=True)
        return WorkingMemoryV2(
            run_id=run_id,
            user_goal=query,
            answer_slots=[slot],
            unresolved_slots=[slot.id],
            current_focus=query,
        ).model_dump(mode="json")

    @staticmethod
    def update_after_observation(
        working_value: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        fingerprint: str,
    ) -> Dict[str, Any]:
        working = WorkingMemoryV2.model_validate(working_value)
        working.attempted_actions.append({
            "tool": observation.get("tool_name"),
            "action_fingerprint": fingerprint,
            "status": observation.get("status"),
            "evidence_ids": observation.get("evidence_ids") or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        supported = set(observation.get("supported_slots") or [])
        missing = set(observation.get("missing_slots") or [])
        for slot in working.answer_slots:
            if slot.id in supported:
                slot.status = "supported"
                slot.evidence_ids = list(dict.fromkeys([
                    *slot.evidence_ids,
                    *(observation.get("evidence_ids") or []),
                ]))
            elif slot.id in missing and slot.status != "supported":
                slot.status = "missing"
        working.unresolved_slots = [
            slot.id for slot in working.answer_slots
            if slot.required and slot.status != "supported"
        ]
        working.current_focus = (
            working.unresolved_slots[0]
            if working.unresolved_slots
            else "生成并验证最终回答"
        )
        working.iteration += 1
        return working.model_dump(mode="json")


react_memory_service = ReactMemoryService()
