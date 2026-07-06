# Veritas RAG - Backend

基于 LangChain + LangGraph + Chroma 的 Agentic RAG 问答系统后端服务。

## 目录结构

```
backend/
├── api/                    # API 层
│   ├── deps/              # 依赖注入
│   └── v1/endpoints/      # API 路由端点
├── core/                  # 核心配置
├── models/                # 数据模型
│   ├── schemas/          # Pydantic 模型
│   └── database/         # 数据库表模型
├── services/              # 业务逻辑层
│   ├── ingestion/        # 文档入库服务
│   ├── retrieval/        # 检索服务
│   ├── rag/              # RAG 编排服务
│   ├── memory/           # 记忆服务
│   └── web_search/       # 联网搜索服务
├── graph/                 # LangGraph 状态机
│   ├── nodes/            # 节点定义
│   ├── routes/           # 路由定义
│   └── state/            # 状态定义
├── db/                    # 数据库连接
│   ├── mysql/            # MySQL 连接
│   ├── redis/            # Redis 连接
│   └── chroma/           # Chroma 连接
├── embeddings/            # 向量化模块
├── prompts/               # Prompt 模板
├── utils/                 # 工具函数
├── tests/                 # 测试
├── main.py               # 应用入口
└── requirements.txt      # 依赖列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env .env
```

### 3. 初始化数据库

```bash
# 创建 MySQL 数据库
mysql -u root -p -e "CREATE DATABASE agentic_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### 4. 启动服务

在仓库根目录执行：

```bash
conda activate cook-rag-1
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

开发时启用热重载：

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

后端内部统一使用 `backend.*` 绝对导入，不需要额外设置 `PYTHONPATH`，
也不要从 `backend/` 目录以 `main:app` 启动。

### 5. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### 用户管理
- `GET /api/v1/users/me` - 获取当前用户信息
- `GET /api/v1/users/me/profile` - 获取用户画像
- `PUT /api/v1/users/me/profile` - 更新用户画像

### 知识库管理
- `POST /api/v1/knowledge/` - 创建知识库
- `GET /api/v1/knowledge/` - 获取知识库列表
- `GET /api/v1/knowledge/{kb_id}` - 获取知识库详情
- `PUT /api/v1/knowledge/{kb_id}` - 更新知识库
- `DELETE /api/v1/knowledge/{kb_id}` - 删除知识库
- `POST /api/v1/knowledge/{kb_id}/upload` - 上传文档
- `GET /api/v1/knowledge/{kb_id}/documents` - 获取文档列表

### RAG 查询
- `POST /api/v1/rag/query` - RAG 查询（非流式，暂未实现）
- `POST /api/v1/rag/query/stream` - RAG 查询（流式 SSE）

### 记忆管理
- `GET /api/v1/memory/` - 获取用户记忆
- `POST /api/v1/memory/sync` - 同步用户记忆
- `DELETE /api/v1/memory/sessions/{session_id}` - 删除会话记忆

## 数据库 Schema

### MySQL 表

- `users` - 用户表
- `user_profiles` - 用户画像表
- `sessions` - 会话表
- `knowledge_bases` - 知识库表
- `documents` - 文档表
- `chunks` - Chunk 元数据表
- `document_index_tasks` - 文档索引任务表

### Chroma Collection

- `knowledge_chunks` - 知识 Chunk 向量存储
  - Dense 向量：语义检索
  - 元数据过滤：用户、知识库、父子块、模态、语言

## 开发状态

### ✅ 已完成
- 项目目录结构
- 核心配置管理
- 数据库连接（MySQL、Redis、Chroma）
- 数据模型定义
- API 路由骨架
- 流式输出框架

### 🚧 开发中
- Phase 1: 文档入库与基础检索
- Phase 2: 重排与用户记忆
- Phase 3: LangGraph Agentic 流程
- Phase 4: 多模态与联网增强

## 技术栈

- **框架**: FastAPI
- **数据库**: MySQL + Redis + Chroma
- **LLM**: LangChain + LangGraph
- **对象存储**: S3/MinIO
- **文档处理**: PyPDF, python-docx, unstructured

## 许可证

MIT
