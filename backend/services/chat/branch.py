# -*- coding: utf-8 -*-
"""
对话分支管理服务
支持创建、切换、合并对话分支
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from models.database.user import SessionTable, MessageTable, ConversationBranchTable


class ConversationBranchService:
    """对话分支管理服务"""

    def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_message_id: int,
        parent_branch_id: Optional[int] = None,
        db: Session = None,
    ) -> ConversationBranchTable:
        """
        创建新的对话分支

        Args:
            session_id: 会话 ID
            branch_name: 分支名称
            from_message_id: 从哪条消息开始创建分支
            parent_branch_id: 父分支 ID
            db: 数据库会话

        Returns:
            新创建的分支
        """
        if not db:
            raise ValueError("Database session is required")

        # 验证会话存在
        session = db.query(SessionTable).filter(
            SessionTable.session_id == session_id
        ).first()

        if not session:
            raise ValueError(f"Session {session_id} not found")

        # 验证消息存在
        message = db.query(MessageTable).filter(
            MessageTable.id == from_message_id,
            MessageTable.session_id == session_id,
        ).first()

        if not message:
            raise ValueError(f"Message {from_message_id} not found in session {session_id}")

        # 如果没有指定父分支，使用当前活动分支
        if parent_branch_id is None:
            active_branch = db.query(ConversationBranchTable).filter(
                ConversationBranchTable.session_id == session_id,
                ConversationBranchTable.is_active == True,
            ).first()

            if active_branch:
                parent_branch_id = active_branch.id

        # 创建新分支
        branch = ConversationBranchTable(
            session_id=session_id,
            branch_name=branch_name,
            parent_branch_id=parent_branch_id,
            parent_message_id=from_message_id,
        )

        db.add(branch)
        db.commit()
        db.refresh(branch)

        return branch

    def list_branches(
        self,
        session_id: str,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """
        列出会话的所有分支

        Args:
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            分支列表
        """
        if not db:
            raise ValueError("Database session is required")

        branches = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.session_id == session_id
        ).order_by(ConversationBranchTable.created_at).all()

        result = []
        for branch in branches:
            # 获取分支消息数量
            message_count = db.query(MessageTable).filter(
                MessageTable.branch_id == branch.id
            ).count()

            result.append({
                "id": branch.id,
                "branch_name": branch.branch_name,
                "parent_branch_id": branch.parent_branch_id,
                "parent_message_id": branch.parent_message_id,
                "is_active": branch.is_active,
                "message_count": message_count,
                "created_at": branch.created_at.isoformat(),
            })

        return result

    def switch_branch(
        self,
        branch_id: int,
        db: Session = None,
    ) -> bool:
        """
        切换到指定分支

        Args:
            branch_id: 分支 ID
            db: 数据库会话

        Returns:
            是否成功
        """
        if not db:
            raise ValueError("Database session is required")

        # 获取目标分支
        branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == branch_id
        ).first()

        if not branch:
            raise ValueError(f"Branch {branch_id} not found")

        # 取消当前活动分支
        db.query(ConversationBranchTable).filter(
            ConversationBranchTable.session_id == branch.session_id,
            ConversationBranchTable.is_active == True,
        ).update({"is_active": False})

        # 激活目标分支
        branch.is_active = True
        db.commit()

        return True

    def delete_branch(
        self,
        branch_id: int,
        db: Session = None,
    ) -> bool:
        """
        删除分支及其消息

        Args:
            branch_id: 分支 ID
            db: 数据库会话

        Returns:
            是否成功
        """
        if not db:
            raise ValueError("Database session is required")

        branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == branch_id
        ).first()

        if not branch:
            raise ValueError(f"Branch {branch_id} not found")

        # 不能删除活动分支
        if branch.is_active:
            raise ValueError("Cannot delete active branch")

        # 删除分支的所有消息
        db.query(MessageTable).filter(
            MessageTable.branch_id == branch_id
        ).delete()

        # 删除分支
        db.delete(branch)
        db.commit()

        return True

    def merge_branch(
        self,
        source_branch_id: int,
        target_branch_id: int,
        db: Session = None,
    ) -> bool:
        """
        合并分支到目标分支

        Args:
            source_branch_id: 源分支 ID
            target_branch_id: 目标分支 ID
            db: 数据库会话

        Returns:
            是否成功
        """
        if not db:
            raise ValueError("Database session is required")

        source_branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == source_branch_id
        ).first()

        target_branch = db.query(ConversationBranchTable).filter(
            ConversationBranchTable.id == target_branch_id
        ).first()

        if not source_branch or not target_branch:
            raise ValueError("Source or target branch not found")

        if source_branch.session_id != target_branch.session_id:
            raise ValueError("Cannot merge branches from different sessions")

        # 将源分支的消息转移到目标分支
        db.query(MessageTable).filter(
            MessageTable.branch_id == source_branch_id
        ).update({"branch_id": target_branch_id})

        # 删除源分支
        db.delete(source_branch)
        db.commit()

        return True

    def get_branch_messages(
        self,
        branch_id: int,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """
        获取分支的所有消息

        Args:
            branch_id: 分支 ID
            db: 数据库会话

        Returns:
            消息列表
        """
        if not db:
            raise ValueError("Database session is required")

        messages = db.query(MessageTable).filter(
            MessageTable.branch_id == branch_id
        ).order_by(MessageTable.created_at).all()

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations,
                "token_count": msg.token_count,
                "created_at": msg.created_at.isoformat(),
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
        """
        从指定消息创建新分支

        Args:
            session_id: 会话 ID
            from_message_id: 从哪条消息开始分支
            branch_name: 分支名称
            db: 数据库会话

        Returns:
            新创建的分支
        """
        if not db:
            raise ValueError("Database session is required")

        # 生成默认分支名称
        if not branch_name:
            branch_name = f"分支 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # 创建新分支
        branch = self.create_branch(
            session_id=session_id,
            branch_name=branch_name,
            from_message_id=from_message_id,
            db=db,
        )

        # 复制该消息之前的所有消息到新分支
        from_message = db.query(MessageTable).filter(
            MessageTable.id == from_message_id
        ).first()

        if from_message:
            # 获取该消息之前的所有消息
            previous_messages = db.query(MessageTable).filter(
                MessageTable.session_id == session_id,
                MessageTable.created_at <= from_message.created_at,
            ).all()

            # 复制消息到新分支
            for msg in previous_messages:
                new_msg = MessageTable(
                    session_id=session_id,
                    role=msg.role,
                    content=msg.content,
                    citations=msg.citations,
                    token_count=msg.token_count,
                    branch_id=branch.id,
                )
                db.add(new_msg)

            db.commit()

        return branch


# 全局分支服务实例
branch_service = ConversationBranchService()
