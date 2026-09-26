# -*- coding: utf-8 -*-
"""Independent conversation fork management.

A fork copies messages and the safe session-summary snapshot into a new normal
session. ConversationBranchTable records lineage only; online chat does not load
branch-specific memory or implicitly switch a source session into a branch.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.database.user import (
    ConversationBranchTable,
    MessageTable,
    SessionMemoryTable,
    SessionTable,
)


class ConversationBranchService:
    """Create immutable lineage records for independent forked sessions."""

    @staticmethod
    def _require_db(db: Optional[Session]) -> Session:
        if db is None:
            raise ValueError("Database session is required")
        return db

    def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_message_id: int,
        parent_branch_id: Optional[int] = None,
        forked_session_id: Optional[str] = None,
        db: Session = None,
    ) -> ConversationBranchTable:
        """Create a lineage record after the forked session is prepared."""
        db = self._require_db(db)
        source_session = db.query(SessionTable).filter(
            SessionTable.session_id == session_id
        ).first()
        source_message = db.query(MessageTable).filter(
            MessageTable.id == from_message_id,
            MessageTable.session_id == session_id,
        ).first()
        if not source_session:
            raise ValueError(f"Session {session_id} not found")
        if not source_message:
            raise ValueError(f"Message {from_message_id} not found in session {session_id}")
        if parent_branch_id is None:
            parent = db.query(ConversationBranchTable).filter(
                ConversationBranchTable.forked_session_id == session_id
            ).first()
            parent_branch_id = parent.id if parent else None
        branch = ConversationBranchTable(
            session_id=session_id,
            branch_name=branch_name,
            parent_branch_id=parent_branch_id,
            parent_message_id=from_message_id,
            forked_session_id=forked_session_id,
            is_active=False,
        )
        db.add(branch)
        db.flush()
        return branch

    def list_branches(
        self,
        session_id: str,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        db = self._require_db(db)
        branches = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.session_id == session_id
        ).order_by(ConversationBranchTable.created_at).all()
        result = []
        for branch in branches:
            message_count = 0
            if branch.forked_session_id:
                message_count = db.query(MessageTable).filter(
                    MessageTable.session_id == branch.forked_session_id,
                    MessageTable.branch_id.is_(None),
                ).count()
            result.append({
                "id": branch.id,
                "branch_name": branch.branch_name,
                "parent_branch_id": branch.parent_branch_id,
                "parent_message_id": branch.parent_message_id,
                "forked_session_id": branch.forked_session_id,
                "is_active": False,
                "message_count": message_count,
                "created_at": branch.created_at.isoformat() if branch.created_at else None,
            })
        return result

    def switch_branch(self, branch_id: int, db: Session = None) -> str:
        """Return the independent session the client should navigate to."""
        db = self._require_db(db)
        branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == branch_id
        ).first()
        if not branch:
            raise ValueError(f"Branch {branch_id} not found")
        if not branch.forked_session_id:
            raise ValueError("Legacy branch has no independent session")
        return branch.forked_session_id

    def delete_branch(self, branch_id: int, db: Session = None) -> bool:
        db = self._require_db(db)
        branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == branch_id
        ).first()
        if not branch:
            raise ValueError(f"Branch {branch_id} not found")
        forked_session_id = branch.forked_session_id

        # Descendant forks remain valid independent sessions even when their
        # lineage parent is removed. Detach them before deleting this record.
        db.query(ConversationBranchTable).filter(
            ConversationBranchTable.parent_branch_id == branch.id
        ).update({"parent_branch_id": None}, synchronize_session=False)

        # The branch row references the forked session. Remove that reference
        # before deleting the session so MySQL does not reject the operation.
        branch.forked_session_id = None
        db.flush()
        db.delete(branch)
        db.flush()

        if forked_session_id:
            outgoing_branches = db.query(ConversationBranchTable).filter(
                ConversationBranchTable.session_id == forked_session_id
            ).all()
            outgoing_ids = [item.id for item in outgoing_branches]
            if outgoing_ids:
                db.query(ConversationBranchTable).filter(
                    ConversationBranchTable.parent_branch_id.in_(outgoing_ids)
                ).update({"parent_branch_id": None}, synchronize_session=False)
                db.query(MessageTable).filter(
                    MessageTable.branch_id.in_(outgoing_ids)
                ).update({"branch_id": None}, synchronize_session=False)
            db.query(MessageTable).filter(
                MessageTable.session_id == forked_session_id
            ).delete(synchronize_session=False)
            if outgoing_ids:
                db.query(ConversationBranchTable).filter(
                    ConversationBranchTable.id.in_(outgoing_ids)
                ).delete(synchronize_session=False)
            db.query(SessionMemoryTable).filter(
                SessionMemoryTable.session_id == forked_session_id
            ).delete(synchronize_session=False)
            db.query(SessionTable).filter(
                SessionTable.session_id == forked_session_id
            ).delete(synchronize_session=False)
        db.commit()
        return True

    def merge_branch(
        self,
        source_branch_id: int,
        target_branch_id: int,
        db: Session = None,
    ) -> bool:
        self._require_db(db)
        raise ValueError(
            "Independent session branches cannot be merged implicitly; "
            "use an explicit semantic merge workflow"
        )

    def get_branch_messages(
        self,
        branch_id: int,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        db = self._require_db(db)
        branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == branch_id
        ).first()
        if not branch or not branch.forked_session_id:
            raise ValueError("Independent branch session not found")
        messages = db.query(MessageTable).filter(
            MessageTable.session_id == branch.forked_session_id,
            MessageTable.branch_id.is_(None),
        ).order_by(MessageTable.id).all()
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations,
                "token_count": msg.token_count,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ]

    def fork_from_message(
        self,
        session_id: str,
        from_message_id: int,
        branch_name: Optional[str] = None,
        db: Session = None,
    ) -> ConversationBranchTable:
        """Copy conversation state through a message into a new session."""
        db = self._require_db(db)
        branch_name = branch_name or f"分支 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        source_session = db.query(SessionTable).filter(
            SessionTable.session_id == session_id
        ).first()
        from_message = db.query(MessageTable).filter(
            MessageTable.id == from_message_id,
            MessageTable.session_id == session_id,
        ).first()
        if not source_session or not from_message:
            raise ValueError("Source session or message not found")

        source_branch_id = from_message.branch_id
        messages_query = db.query(MessageTable).filter(
            MessageTable.session_id == session_id,
            MessageTable.id <= from_message.id,
        )
        messages_query = messages_query.filter(
            MessageTable.branch_id == source_branch_id
            if source_branch_id is not None
            else MessageTable.branch_id.is_(None)
        )
        previous_messages = messages_query.order_by(MessageTable.id).all()

        forked_session_id = str(uuid.uuid4())
        db.add(SessionTable(
            session_id=forked_session_id,
            user_id=source_session.user_id,
            kb_id=source_session.kb_id,
            title=branch_name,
            summary=None,
            context=copy.deepcopy(source_session.context or {}),
            category=source_session.category,
            archived=False,
            message_count=len(previous_messages),
        ))
        db.flush()

        copied_message_ids: Dict[int, int] = {}
        for message in previous_messages:
            copied = MessageTable(
                session_id=forked_session_id,
                role=message.role,
                content=message.content,
                citations=copy.deepcopy(message.citations),
                token_count=message.token_count,
                request_id=None,
                branch_id=None,
            )
            db.add(copied)
            db.flush()
            copied_message_ids[message.id] = copied.id

        source_memory = db.query(SessionMemoryTable).filter(
            SessionMemoryTable.session_id == session_id
        ).first()
        summary_is_safe = bool(
            source_memory
            and source_memory.summary_through_message_id
            and source_memory.summary_through_message_id <= from_message_id
            and source_memory.summary_through_message_id in copied_message_ids
        )
        if summary_is_safe:
            summary = copy.deepcopy(source_memory.summary or {})
            summary_text = source_memory.summary_text
            copied_summary_cursor = copied_message_ids[
                source_memory.summary_through_message_id
            ]
        else:
            # A source summary that includes messages after the fork point must
            # never leak future context into the independent session. The copied
            # message history will be summarized normally on a later turn.
            summary = {
                "session_goal": branch_name,
                "confirmed_decisions": [],
                "discarded_ideas": [],
                "open_questions": [],
            }
            summary_text = None
            copied_summary_cursor = None
        db.add(SessionMemoryTable(
            session_id=forked_session_id,
            summary=summary,
            summary_text=summary_text,
            summary_through_message_id=copied_summary_cursor,
            version=1,
        ))

        branch = self.create_branch(
            session_id=session_id,
            branch_name=branch_name,
            from_message_id=from_message_id,
            forked_session_id=forked_session_id,
            db=db,
        )
        db.commit()
        db.refresh(branch)
        return branch


branch_service = ConversationBranchService()
