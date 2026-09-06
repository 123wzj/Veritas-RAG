# -*- coding: utf-8 -*-
"""
知识库管理 API 路由
"""

import aiofiles
import hashlib
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List, Dict, Any, Optional

from backend.core.config import settings
from backend.db.mysql.connection import get_db
from backend.models.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBase,
    Document,
    DocumentStatus,
    CapabilitiesResponse,
)
from backend.models.database.knowledge import KnowledgeBaseTable, DocumentTable, ChunkTable, KnowledgeBasePermissionTable
from backend.api.deps.common import get_required_user
from backend.services.ingestion.ingestion import ingestion_service
from backend.services.acl.permission import permission_service
from backend.db.chroma.connection import chroma_client

router = APIRouter()

# 创建上传目录
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_upload_path(upload_dir: Path, filename: str, file_hash: str) -> Path:
    candidate = upload_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    short_hash = file_hash[:8]
    for index in range(1, 1000):
        next_candidate = upload_dir / f"{stem}_{short_hash}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
    raise RuntimeError("Unable to allocate a unique upload filename")


async def _save_upload_to_temp(file: UploadFile, temp_path: Path) -> tuple[str, int]:
    """Stream an upload to disk while enforcing the configured size limit."""
    hasher = hashlib.sha256()
    total_size = 0

    async with aiofiles.open(temp_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.MAX_UPLOAD_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds max upload size: {settings.MAX_UPLOAD_FILE_SIZE} bytes",
                )
            hasher.update(chunk)
            await f.write(chunk)

    return hasher.hexdigest(), total_size


# ========== 知识库管理 ==========

@router.post("/", response_model=KnowledgeBase)
async def create_knowledge_base(
    kb_create: KnowledgeBaseCreate,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    创建知识库
    """
    db_kb = KnowledgeBaseTable(
        user_id=current_user.id,
        name=kb_create.name,
        description=kb_create.description,
        acl_tags=kb_create.acl_tags,
    )
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)
    permission_service.grant_permission(
        kb_id=db_kb.id,
        target_user_id=current_user.id,
        permission_type="admin",
        granted_by=current_user.id,
        db=db,
    )

    return KnowledgeBase(
        id=db_kb.id,
        user_id=db_kb.user_id,
        name=db_kb.name,
        description=db_kb.description,
        acl_tags=db_kb.acl_tags,
        document_count=0,
        created_at=db_kb.created_at,
        updated_at=db_kb.updated_at,
    )


@router.get("/", response_model=List[KnowledgeBase])
async def list_knowledge_bases(
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取用户可访问的知识库列表
    """
    # 获取用户可访问的知识库 ID 列表
    accessible_kb_ids = permission_service.get_accessible_knowledge_bases(
        user_id=current_user.id,
        db=db,
    )

    kbs = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id.in_(accessible_kb_ids)
    ).all()
    doc_counts = dict(
        db.query(DocumentTable.kb_id, func.count(DocumentTable.id))
        .filter(DocumentTable.kb_id.in_(accessible_kb_ids))
        .group_by(DocumentTable.kb_id)
        .all()
    )

    return [
        KnowledgeBase(
            id=kb.id,
            user_id=kb.user_id,
            name=kb.name,
            description=kb.description,
            acl_tags=kb.acl_tags,
            document_count=int(doc_counts.get(kb.id, 0)),
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        )
        for kb in kbs
    ]


@router.get("/{kb_id}", response_model=KnowledgeBase)
async def get_knowledge_base(
    kb_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取指定知识库详情
    """
    # 检查权限
    if not permission_service.check_permission(kb_id, current_user.id, "read", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    document_count = db.query(func.count(DocumentTable.id)).filter(DocumentTable.kb_id == kb_id).scalar() or 0

    return KnowledgeBase(
        id=kb.id,
        user_id=kb.user_id,
        name=kb.name,
        description=kb.description,
        acl_tags=kb.acl_tags,
        document_count=int(document_count),
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.put("/{kb_id}", response_model=KnowledgeBase)
async def update_knowledge_base(
    kb_id: int,
    kb_update: KnowledgeBaseUpdate,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    更新知识库
    """
    # 检查权限（需要写权限）
    if not permission_service.check_permission(kb_id, current_user.id, "write", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    if kb_update.name is not None:
        kb.name = kb_update.name
    if kb_update.description is not None:
        kb.description = kb_update.description
    if kb_update.acl_tags is not None:
        kb.acl_tags = kb_update.acl_tags

    db.commit()
    db.refresh(kb)

    document_count = db.query(func.count(DocumentTable.id)).filter(DocumentTable.kb_id == kb_id).scalar() or 0

    return KnowledgeBase(
        id=kb.id,
        user_id=kb.user_id,
        name=kb.name,
        description=kb.description,
        acl_tags=kb.acl_tags,
        document_count=int(document_count),
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    删除知识库
    """
    # 检查权限（需要管理员权限）
    if not permission_service.check_permission(kb_id, current_user.id, "admin", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    try:
        docs = db.query(DocumentTable).filter(DocumentTable.kb_id == kb_id).all()
        for doc in docs:
            file_path = Path(doc.file_path)
            if file_path.exists():
                file_path.unlink()

        collection = chroma_client.get_collection()
        chroma_results = collection.get(where={"kb_id": kb_id}, include=[])
        vector_ids = chroma_results.get("ids") or []
        if vector_ids:
            collection.delete(ids=vector_ids)

        db.query(ChunkTable).filter(ChunkTable.kb_id == kb_id).delete()
        db.query(DocumentTable).filter(DocumentTable.kb_id == kb_id).delete()
        db.query(KnowledgeBasePermissionTable).filter(
            KnowledgeBasePermissionTable.kb_id == kb_id
        ).delete()
        db.delete(kb)
        db.commit()

        kb_upload_dir = UPLOAD_DIR / str(kb.user_id) / str(kb_id)
        if kb_upload_dir.exists():
            shutil.rmtree(kb_upload_dir, ignore_errors=True)

        from backend.services.retrieval.hybrid import hybrid_retriever
        hybrid_retriever.invalidate_sparse_cache()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete knowledge base: {str(e)}"
        )

    return {"message": "Knowledge base deleted successfully"}


# ========== 文档管理 ==========

@router.post("/{kb_id}/upload")
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    confirm_same_name: bool = Query(False, description="Confirm upload when another document has the same filename"),
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    上传文档到知识库
    """
    # 验证知识库存在且用户有写权限
    if not permission_service.check_permission(kb_id, current_user.id, "write", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    # 验证文件扩展名
    filename = file.filename or "uploaded_file"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_file_type",
                "message": "V1 仅支持 Markdown（.md）文档。",
                "allowed_types": settings.ALLOWED_FILE_EXTENSIONS,
            },
        )

    file_path: Optional[Path] = None
    temp_path: Optional[Path] = None
    try:
        user_upload_dir = UPLOAD_DIR / str(current_user.id) / str(kb_id)
        user_upload_dir.mkdir(parents=True, exist_ok=True)

        temp_path = user_upload_dir / f".upload_{uuid.uuid4().hex}.tmp"
        file_hash, file_size = await _save_upload_to_temp(file, temp_path)

        duplicate_doc = db.query(DocumentTable).filter(
            DocumentTable.kb_id == kb_id,
            DocumentTable.file_hash == file_hash,
        ).first()
        if duplicate_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "duplicate_file",
                    "message": f"文件“{filename}”与已上传文档“{duplicate_doc.filename}”内容完全一致，请勿重复上传。",
                    "duplicate_doc_id": duplicate_doc.doc_id,
                    "duplicate_filename": duplicate_doc.filename,
                    "file_hash": file_hash,
                },
            )

        same_name_doc = db.query(DocumentTable).filter(
            DocumentTable.kb_id == kb_id,
            DocumentTable.filename == filename,
        ).first()
        if same_name_doc and not confirm_same_name:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "same_filename",
                    "message": f"知识库中已存在同名文件“{filename}”，请确认是否继续上传。",
                    "requires_confirmation": True,
                    "existing_doc_id": same_name_doc.doc_id,
                    "existing_filename": same_name_doc.filename,
                },
            )

        file_path = _safe_upload_path(user_upload_dir, filename, file_hash)
        temp_path.replace(file_path)
        temp_path = None

        # 2. 调用入库服务
        async def task_callback(status: str, progress: int, message: str):
            """任务进度回调"""
            # 可以在这里通过 WebSocket 或 SSE 推送进度到前端
            pass

        result = await ingestion_service.ingest_document(
            file_path=str(file_path),
            kb_id=kb_id,
            user_id=current_user.id,
            db=db,
            task_callback=task_callback,
            original_filename=filename,
            file_hash=file_hash,
        )

        return {
            "message": "Document uploaded and indexed successfully",
            "doc_id": result["doc_id"],
            "filename": filename,
            "file_hash": file_hash,
            "file_size": file_size,
            "kb_id": kb_id,
            "status": result["status"],
            "parent_count": result["parent_count"],
            "child_count": result["child_count"],
            "modality": result.get("modality"),
            "modality_stats": result.get("modality_stats"),
        }

    except Exception as e:
        # 清理已上传的文件
        if temp_path and temp_path.exists():
            temp_path.unlink()
        if file_path and file_path.exists():
            file_path.unlink()

        if isinstance(e, HTTPException):
            raise e

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/{kb_id}/documents", response_model=List[Document])
async def list_documents(
    kb_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取知识库中的文档列表
    """
    # 验证知识库存在且用户有读权限
    if not permission_service.check_permission(kb_id, current_user.id, "read", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    docs = db.query(DocumentTable).filter(DocumentTable.kb_id == kb_id).all()

    return [
        Document(
            id=doc.id,
            kb_id=doc.kb_id,
            doc_id=doc.doc_id,
            filename=doc.filename,
            file_path=doc.file_path,
            file_size=doc.file_size,
            status=DocumentStatus(doc.status),
            modality=doc.modality,
            language=doc.language,
            total_chunks=doc.total_chunks,
            error_message=doc.error_message,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc in docs
    ]


@router.get("/{kb_id}/documents/{doc_id}", response_model=Document)
async def get_document(
    kb_id: int,
    doc_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取指定文档详情
    """
    # 验证知识库存在且用户有读权限
    if not permission_service.check_permission(kb_id, current_user.id, "read", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    doc = db.query(DocumentTable).filter(
        DocumentTable.kb_id == kb_id,
        DocumentTable.doc_id == doc_id,
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return Document(
        id=doc.id,
        kb_id=doc.kb_id,
        doc_id=doc.doc_id,
        filename=doc.filename,
        file_path=doc.file_path,
        file_size=doc.file_size,
        status=DocumentStatus(doc.status),
        modality=doc.modality,
        language=doc.language,
        total_chunks=doc.total_chunks,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: int,
    doc_id: str,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    删除指定文档
    """
    # 验证知识库存在且用户有写权限
    if not permission_service.check_permission(kb_id, current_user.id, "write", db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    kb = db.query(KnowledgeBaseTable).filter(
        KnowledgeBaseTable.id == kb_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )

    # 验证文档存在
    doc = db.query(DocumentTable).filter(
        DocumentTable.kb_id == kb_id,
        DocumentTable.doc_id == doc_id,
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # 删除文件
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()

    # 调用入库服务删除
    await ingestion_service.delete_document(
        doc_id=doc_id,
        kb_id=kb_id,
        db=db,
    )

    return {"message": "Document deleted successfully"}


@router.get("/{kb_id}/capabilities", response_model=CapabilitiesResponse)
async def knowledge_capabilities(
    kb_id: int,
    current_user=Depends(get_required_user),
    db: Session = Depends(get_db),
):
    if not permission_service.check_permission(kb_id, current_user.id, "read", db):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return {
        "ingestion": {"allowed_extensions": [".md"], "label": "Markdown only"},
        "retrieval": {"dense": "BGE-M3", "sparse": "BM25", "reranker": "BGE"},
    }


# ========== 权限管理 ==========

@router.get("/{kb_id}/permissions")
async def list_permissions(
    kb_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    获取知识库的权限列表
    """
    try:
        permissions = permission_service.list_permissions(
            kb_id=kb_id,
            user_id=current_user.id,
            db=db,
        )
        return {"permissions": permissions}
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/{kb_id}/permissions")
async def grant_permission(
    kb_id: int,
    target_user_id: int = Query(..., description="被授权用户 ID"),
    permission_type: str = Query(..., pattern="^(read|write|admin)$", description="权限类型"),
    expires_at: Optional[str] = Query(None, description="过期时间 (ISO 8601 格式)"),
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    授予用户知识库权限
    """
    try:
        # 解析过期时间
        expiry = None
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid expires_at format. Use ISO 8601 format."
                )

        permission = permission_service.grant_permission(
            kb_id=kb_id,
            target_user_id=target_user_id,
            permission_type=permission_type,
            granted_by=current_user.id,
            db=db,
            expires_at=expiry,
        )

        return {
            "message": "Permission granted successfully",
            "permission": {
                "id": permission.id,
                "kb_id": permission.kb_id,
                "user_id": permission.user_id,
                "permission_type": permission.permission_type,
                "expires_at": permission.expires_at.isoformat() if permission.expires_at else None,
            }
        }
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.delete("/{kb_id}/permissions/{target_user_id}")
async def revoke_permission(
    kb_id: int,
    target_user_id: int,
    current_user = Depends(get_required_user),
    db: Session = Depends(get_db),
):
    """
    撤销用户对知识库的权限
    """
    try:
        success = permission_service.revoke_permission(
            kb_id=kb_id,
            target_user_id=target_user_id,
            revoked_by=current_user.id,
            db=db,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Permission not found"
            )

        return {"message": "Permission revoked successfully"}
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
