# -*- coding: utf-8 -*-
"""
Veritas RAG 主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import settings
from db.mysql.connection import init_db, close_db
from db.redis.connection import redis_client
from db.chroma.connection import chroma_client

# 导入所有模型以确保 SQLAlchemy 能自动创建表
from models.database import user, knowledge


# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting Veritas RAG...")
    init_db()
    logger.info("Database initialized")
    yield
    # 关闭时执行
    logger.info("Shutting down Veritas RAG...")
    close_db()
    redis_client.close()
    chroma_client.close()
    logger.info("Connections closed")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 LangChain + LangGraph + Chroma 的 Veritas RAG 智能知识库问答系统",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 健康检查 ==========
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ========== API 路由 ==========
from api.v1.endpoints import rag, knowledge, users, memory

app.include_router(users.router, prefix=f"{settings.API_V1_PREFIX}/users", tags=["Users"])
app.include_router(knowledge.router, prefix=f"{settings.API_V1_PREFIX}/knowledge", tags=["Knowledge"])
app.include_router(rag.router, prefix=f"{settings.API_V1_PREFIX}/rag", tags=["RAG"])
app.include_router(memory.router, prefix=f"{settings.API_V1_PREFIX}/memory", tags=["Memory"])


# ========== 根路径 ==========
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to Veritas RAG",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
