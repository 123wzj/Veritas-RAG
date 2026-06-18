# -*- coding: utf-8 -*-
"""
核心配置模块
包含所有环境变量和系统配置
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, List

try:
    from core.hf_cache import configure_hf_cache
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.core.hf_cache import configure_hf_cache


configure_hf_cache()

BASE_DIR = Path(__file__).resolve().parents[2]


def _resolve_project_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((BASE_DIR / path).resolve())


class Settings(BaseSettings):
    """系统配置"""

    # ========== 应用配置 ==========
    APP_NAME: str = "Veritas RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # ========== 服务器配置 ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ========== 数据库配置 ==========
    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DB: str = "agentic_rag"

    # Redis (可选)
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Chroma 向量数据库
    CHROMA_PERSIST_DIR: str = "data/chroma"
    CHROMA_COLLECTION_NAME: str = "knowledge_chunks"

    # ========== 对象存储配置 (可选) ==========
    # S3 / MinIO
    S3_ENDPOINT: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "veritas-rag-docs"
    S3_REGION: str = "us-east-1"

    # ========== 模型配置 ==========
    # LLM - OpenAI-compatible by default. Local values are loaded from the
    # project root .env only, so backend/.env cannot accidentally override them.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.4"
    OPENAI_BASE_URL: Optional[str] = "http://127.0.0.1:8317/v1"
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-5.4"
    LLM_BASE_URL: Optional[str] = None
    BASE_URL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # Embedding - local open-source model by default.
    EMBEDDING_PROVIDER: str = "bge_m3"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_MAX_LENGTH: int = 512

    # Reranker - local open-source model by default.
    RERANKER_PROVIDER: str = "bge"
    RERANKER_API_KEY: str = ""
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_BASE_URL: Optional[str] = None
    RERANKER_DEVICE: str = "cuda"
    RERANKER_BATCH_SIZE: int = 8

    # Vision / OCR
    VISION_PROVIDER: str = "qwen"
    VISION_API_KEY: str = ""
    VISION_MODEL: str = "qwen-vl-max"
    VISION_BASE_URL: Optional[str] = None

    # ========== RAG 配置 ==========
    # 检索配置
    DEFAULT_TOP_K: int = 40
    DEFAULT_RERANK_TOP_K: int = 6
    HYBRID_SEARCH_WEIGHT: float = 0.5  # Dense 权重, Sparse 为 1-weight

    # 分块配置
    PARENT_CHUNK_SIZE: int = 1000
    PARENT_CHUNK_OVERLAP: int = 100
    CHILD_CHUNK_SIZE: int = 300
    CHILD_CHUNK_OVERLAP: int = 50

    # Agentic 配置
    MAX_REFLECTION_ROUNDS: int = 3
    MAX_TOOL_STEPS: int = 8
    GRAPH_RECURSION_LIMIT: int = 50
    ENABLE_REFLECTION: bool = True

    # RAGAS evaluation uses the same OpenAI-compatible endpoint, but a lower
    # temperature and lower concurrency make judge-style JSON outputs steadier.
    RAGAS_LLM_MODEL: Optional[str] = None
    RAGAS_LLM_TEMPERATURE: float = 0.0
    RAGAS_LLM_MAX_TOKENS: int = 2048
    RAGAS_LLM_TIMEOUT: int = 180
    RAGAS_LLM_MAX_RETRIES: int = 5
    RAGAS_RUN_MAX_RETRIES: int = 5
    RAGAS_RUN_MAX_WAIT: int = 30
    RAGAS_RUN_MAX_WORKERS: int = 4
    RAGAS_BATCH_SIZE: int = 2

    # 联网增强配置
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_PROVIDER: str = "tavily"  # tavily, serpapi, etc.
    WEB_SEARCH_API_KEY: str = ""

    # ========== 用户与安全配置 ==========
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # ========== 可观测性配置 ==========
    # 日志
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    # Tracing (LangSmith)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "veritas-rag"

    # ========== 其他配置 ==========
    MAX_UPLOAD_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".pptx", ".md", ".html", ".txt", ".png", ".jpg", ".jpeg"
    ]
    # 文件上传目录
    UPLOAD_DIR: str = "data/uploads"

    class Config:
        env_file = str(BASE_DIR / ".env")
        case_sensitive = True
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        self.CHROMA_PERSIST_DIR = _resolve_project_path(self.CHROMA_PERSIST_DIR)
        self.UPLOAD_DIR = _resolve_project_path(self.UPLOAD_DIR)
        if self.OPENAI_API_KEY and not self.LLM_API_KEY:
            self.LLM_API_KEY = self.OPENAI_API_KEY
        if self.LLM_API_KEY and not self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = self.LLM_API_KEY
        configured_fields = getattr(self, "model_fields_set", set())
        if self.OPENAI_MODEL and "LLM_MODEL" not in configured_fields:
            self.LLM_MODEL = self.OPENAI_MODEL
        if self.BASE_URL:
            self.LLM_BASE_URL = self.BASE_URL
        elif self.OPENAI_BASE_URL and not self.LLM_BASE_URL:
            self.LLM_BASE_URL = self.OPENAI_BASE_URL


# 全局配置实例
settings = Settings()
