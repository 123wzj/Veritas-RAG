# -*- coding: utf-8 -*-
"""
访问控制服务
处理知识库的权限检查和授权
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models.database.knowledge import KnowledgeBaseTable, KnowledgeBasePermissionTable


class PermissionService:
    """访问控制服务"""

    def check_permission(
        self,
        kb_id: int,
        user_id: int,
        required_permission: str = "read",
        db: Session = None,
    ) -> bool:
        """
        检查用户对知识库的权限

        Args:
            kb_id: 知识库 ID
            user_id: 用户 ID
            required_permission: 需要的权限类型 (read, write, admin)
            db: 数据库会话

        Returns:
            是否有权限
        """
        if not db:
            return False

        # 获取知识库
        kb = db.query(KnowledgeBaseTable).filter(
            KnowledgeBaseTable.id == kb_id
        ).first()

        if not kb:
            return False

        # 所有者拥有所有权限
        if kb.user_id == user_id:
            return True

        # 检查可见性
        if kb.visibility == "private":
            # 私有知识库，只有所有者可以访问
            return False
        elif kb.visibility == "public":
            # 公开知识库，所有人可读
            if required_permission == "read":
                return True

        # 检查显式权限
        permission = db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == kb_id,
            KnowledgeBasePermissionTable.user_id == user_id,
        ).first()

        if permission:
            # 检查权限是否过期
            if permission.expires_at and permission.expires_at < datetime.utcnow():
                return False

            # 检查权限级别
            if required_permission == "read":
                return permission.permission_type in ["read", "write", "admin"]
            elif required_permission == "write":
                return permission.permission_type in ["write", "admin"]
            elif required_permission == "admin":
                return permission.permission_type == "admin"

        return False

    def grant_permission(
        self,
        kb_id: int,
        target_user_id: int,
        permission_type: str,
        granted_by: int,
        db: Session,
        expires_at: Optional[datetime] = None,
    ) -> KnowledgeBasePermissionTable:
        """
        授予用户知识库权限

        Args:
            kb_id: 知识库 ID
            target_user_id: 被授权用户 ID
            permission_type: 权限类型 (read, write, admin)
            granted_by: 授权者用户 ID
            db: 数据库会话
            expires_at: 过期时间

        Returns:
            权限记录
        """
        # 验证授权者是否有权限
        if not self.check_permission(kb_id, granted_by, "admin", db):
            raise PermissionError("Only admins can grant permissions")

        # 检查是否已有权限记录
        existing = db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == kb_id,
            KnowledgeBasePermissionTable.user_id == target_user_id,
        ).first()

        if existing:
            # 更新现有权限
            existing.permission_type = permission_type
            existing.granted_by = granted_by
            existing.expires_at = expires_at
            db.commit()
            db.refresh(existing)
            return existing
        else:
            # 创建新权限
            permission = KnowledgeBasePermissionTable(
                kb_id=kb_id,
                user_id=target_user_id,
                permission_type=permission_type,
                granted_by=granted_by,
                expires_at=expires_at,
            )
            db.add(permission)
            db.commit()
            db.refresh(permission)
            return permission

    def revoke_permission(
        self,
        kb_id: int,
        target_user_id: int,
        revoked_by: int,
        db: Session,
    ) -> bool:
        """
        撤销用户对知识库的权限

        Args:
            kb_id: 知识库 ID
            target_user_id: 被撤销权限的用户 ID
            revoked_by: 撤销者用户 ID
            db: 数据库会话

        Returns:
            是否成功
        """
        # 验证撤销者是否有权限
        if not self.check_permission(kb_id, revoked_by, "admin", db):
            raise PermissionError("Only admins can revoke permissions")

        # 删除权限记录
        permission = db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == kb_id,
            KnowledgeBasePermissionTable.user_id == target_user_id,
        ).first()

        if permission:
            db.delete(permission)
            db.commit()
            return True

        return False

    def list_permissions(
        self,
        kb_id: int,
        user_id: int,
        db: Session,
    ) -> List[Dict[str, Any]]:
        """
        列出知识库的所有权限

        Args:
            kb_id: 知识库 ID
            user_id: 查询者用户 ID
            db: 数据库会话

        Returns:
            权限列表
        """
        # 验证查询者是否有权限
        if not self.check_permission(kb_id, user_id, "admin", db):
            raise PermissionError("Only admins can list permissions")

        permissions = db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == kb_id
        ).all()

        return [
            {
                "id": p.id,
                "user_id": p.user_id,
                "permission_type": p.permission_type,
                "granted_by": p.granted_by,
                "granted_at": p.granted_at.isoformat(),
                "expires_at": p.expires_at.isoformat() if p.expires_at else None,
            }
            for p in permissions
        ]

    def get_accessible_knowledge_bases(
        self,
        user_id: int,
        db: Session,
    ) -> List[int]:
        """
        获取用户可访问的知识库 ID 列表

        Args:
            user_id: 用户 ID
            db: 数据库会话

        Returns:
            可访问的知识库 ID 列表
        """
        # 1. 用户拥有的知识库
        owned = db.query(KnowledgeBaseTable.id).filter(
            KnowledgeBaseTable.user_id == user_id
        ).all()

        # 2. 公开的知识库
        public = db.query(KnowledgeBaseTable.id).filter(
            KnowledgeBaseTable.visibility == "public"
        ).all()

        # 3. 被授权的知识库
        permissions = db.query(KnowledgeBasePermissionTable.kb_id).filter(
            KnowledgeBasePermissionTable.user_id == user_id
        ).all()

        # 合并并去重
        kb_ids = set()
        kb_ids.update([k[0] for k in owned])
        kb_ids.update([k[0] for k in public])
        kb_ids.update([k[0] for k in permissions])

        return list(kb_ids)


# 全局权限服务实例
permission_service = PermissionService()
