# -*- coding: utf-8 -*-
"""
MySQL connection and lightweight schema sync helpers.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import settings

engine = create_engine(
    f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
    f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_column(table_name: str, column_name: str, ddl: str) -> None:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return

    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name in existing_columns:
        return

    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE `{table_name}` ADD COLUMN {ddl}"))
    print(f"[schema-sync] Added column {table_name}.{column_name}")


def _ensure_index(table_name: str, index_name: str, ddl: str) -> None:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return
    existing_indexes = {item["name"] for item in inspector.get_indexes(table_name)}
    existing_indexes.update(
        item.get("name")
        for item in inspector.get_unique_constraints(table_name)
        if item.get("name")
    )
    if index_name in existing_indexes:
        return
    with engine.begin() as connection:
        connection.execute(text(ddl))
    print(f"[schema-sync] Added index {index_name}")


def _sync_legacy_schema() -> None:
    """
    Bring old tables up to the current minimal schema.
    `create_all()` creates missing tables but does not alter existing ones.
    """
    _ensure_column("sessions", "kb_id", "`kb_id` INT NULL")
    _ensure_column("sessions", "last_active", "`last_active` DATETIME NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_column("sessions", "message_count", "`message_count` INT NOT NULL DEFAULT 0")
    _ensure_column("sessions", "title", "`title` VARCHAR(100) NULL")
    _ensure_column("sessions", "summary", "`summary` TEXT NULL")
    _ensure_column("sessions", "context", "`context` JSON NULL")
    _ensure_column("sessions", "category", "`category` VARCHAR(50) NULL")

    _ensure_column("user_profiles", "long_term_facts", "`long_term_facts` JSON NULL")
    _ensure_column("user_profiles", "working_preferences", "`working_preferences` JSON NULL")

    _ensure_column("messages", "citations", "`citations` JSON NULL")
    _ensure_column("messages", "token_count", "`token_count` INT NOT NULL DEFAULT 0")
    _ensure_column("messages", "request_id", "`request_id` VARCHAR(64) NULL")
    _ensure_column("messages", "branch_id", "`branch_id` INT NULL")
    _ensure_index(
        "messages",
        "uq_messages_request_role",
        "CREATE UNIQUE INDEX `uq_messages_request_role` "
        "ON `messages` (`request_id`, `role`)",
    )

    _ensure_column("conversation_branches", "branch_name", "`branch_name` VARCHAR(255) NULL")
    _ensure_column("conversation_branches", "parent_branch_id", "`parent_branch_id` INT NULL")
    _ensure_column("conversation_branches", "parent_message_id", "`parent_message_id` INT NULL")
    _ensure_column("conversation_branches", "is_active", "`is_active` TINYINT(1) NOT NULL DEFAULT 1")

    _ensure_column("knowledge_bases", "description", "`description` TEXT NULL")
    _ensure_column("knowledge_bases", "acl_tags", "`acl_tags` JSON NULL")
    _ensure_column("knowledge_bases", "visibility", "`visibility` VARCHAR(20) NOT NULL DEFAULT 'private'")
    _ensure_column("knowledge_bases", "updated_at", "`updated_at` DATETIME NULL")

    _ensure_column("documents", "file_path", "`file_path` VARCHAR(500) NULL")
    _ensure_column("documents", "file_hash", "`file_hash` VARCHAR(64) NULL")
    _ensure_column("documents", "file_size", "`file_size` INT NOT NULL DEFAULT 0")
    _ensure_column("documents", "status", "`status` VARCHAR(20) NOT NULL DEFAULT 'pending'")
    _ensure_column("documents", "modality", "`modality` VARCHAR(20) NOT NULL DEFAULT 'text'")
    _ensure_column("documents", "language", "`language` VARCHAR(10) NOT NULL DEFAULT 'zh'")
    _ensure_column("documents", "total_chunks", "`total_chunks` INT NOT NULL DEFAULT 0")
    _ensure_column("documents", "error_message", "`error_message` TEXT NULL")
    _ensure_column("documents", "doc_metadata", "`doc_metadata` JSON NULL")
    _ensure_column("documents", "updated_at", "`updated_at` DATETIME NULL")

    _ensure_column("chunks", "parent_id", "`parent_id` VARCHAR(64) NULL")
    _ensure_column("chunks", "is_parent", "`is_parent` TINYINT(1) NOT NULL DEFAULT 0")
    _ensure_column("chunks", "modality", "`modality` VARCHAR(20) NOT NULL DEFAULT 'text'")
    _ensure_column("chunks", "language", "`language` VARCHAR(10) NOT NULL DEFAULT 'zh'")
    _ensure_column("chunks", "title", "`title` VARCHAR(255) NULL")
    _ensure_column("chunks", "section_path", "`section_path` VARCHAR(500) NULL")
    _ensure_column("chunks", "page_no", "`page_no` INT NULL")
    _ensure_column("chunks", "content", "`content` LONGTEXT NULL")
    _ensure_column("chunks", "sparse_vector", "`sparse_vector` JSON NULL")
    _ensure_column("chunks", "caption", "`caption` TEXT NULL")
    _ensure_column("chunks", "source_uri", "`source_uri` VARCHAR(500) NULL")
    _ensure_column("chunks", "token_count", "`token_count` INT NULL")

    _ensure_column("document_index_tasks", "status", "`status` VARCHAR(20) NOT NULL DEFAULT 'pending'")
    _ensure_column("document_index_tasks", "progress", "`progress` INT NOT NULL DEFAULT 0")
    _ensure_column("document_index_tasks", "current_step", "`current_step` VARCHAR(100) NULL")
    _ensure_column("document_index_tasks", "error_message", "`error_message` TEXT NULL")
    _ensure_column("document_index_tasks", "started_at", "`started_at` DATETIME NULL")
    _ensure_column("document_index_tasks", "completed_at", "`completed_at` DATETIME NULL")
    _ensure_column("document_index_tasks", "created_at", "`created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")

    _ensure_column("kb_permissions", "permission_type", "`permission_type` VARCHAR(20) NOT NULL DEFAULT 'read'")
    _ensure_column("kb_permissions", "granted_by", "`granted_by` INT NULL")
    _ensure_column("kb_permissions", "granted_at", "`granted_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_column("kb_permissions", "expires_at", "`expires_at` DATETIME NULL")

    with engine.begin() as connection:
        connection.execute(text(
            "UPDATE `sessions` SET `title` = LEFT(`summary`, 100) "
            "WHERE (`title` IS NULL OR `title` = '') AND `summary` IS NOT NULL"
        ))


def init_db() -> None:
    """Initialize tables, sync old schemas, and seed the default dev user."""
    Base.metadata.create_all(bind=engine)
    _sync_legacy_schema()

    db = SessionLocal()
    try:
        from backend.models.database.user import UserProfileTable, UserTable

        default_user = db.query(UserTable).filter(UserTable.id == 1).first()
        if not default_user:
            default_user = UserTable(
                id=1,
                username="default_user",
                email="default@example.com",
                hashed_password="test_password",
                is_active=True,
            )
            db.add(default_user)
            db.commit()

            profile = UserProfileTable(
                user_id=1,
                preferred_language="zh-CN",
                interaction_style="detailed",
            )
            db.add(profile)
            db.commit()

            print("Default user created (ID=1, username=default_user)")
        else:
            print("Default user already exists (ID=1)")
    except Exception as exc:
        db.rollback()
        print(f"Failed to initialize default user: {exc}")
    finally:
        db.close()


def close_db() -> None:
    engine.dispose()
