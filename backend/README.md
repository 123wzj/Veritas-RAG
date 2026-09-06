# Veritas RAG - Backend

基于 FastAPI、LangGraph、Chroma 和 MySQL 的 Agentic RAG 问答系统后端服务。当前主链路已包含文档解析与 Parent-Child 入库、Dense + BM25 混合检索、RRF 融合、Parent 回补、Rerank、证据评估、Reflection、引用验证和会话记忆。

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

### 1. 准备运行环境

项目后端使用 Conda 环境 `cook-rag-1`（当前已验证 Python 3.12.7）：

```bash
conda activate cook-rag-1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp backend/.env.example .env
```

### 4. 初始化数据库

```bash
# 创建 MySQL 数据库
mysql -u root -p -e "CREATE DATABASE agentic_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

### 5. 启动服务

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

### 6. 访问 API 文档

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
- `POST /api/v1/rag/query` - RAG 查询（非流式）
- `POST /api/v1/rag/query/stream` - RAG 查询（流式 SSE）
- `POST /api/v1/rag/test` - RAG 调试/测试入口

### 记忆管理
- `GET /api/v1/memory/` - 获取用户记忆
- `POST /api/v1/memory/sync` - 同步用户记忆
- `DELETE /api/v1/memory/sessions/{session_id}` - 删除会话记忆
- `GET/PATCH/DELETE /api/v1/memory/long-term...` - 长期记忆管理与审计查询

### 会话与分支
- `GET/POST/PATCH/DELETE /api/v1/users/sessions...` - 会话生命周期管理
- `GET/POST/DELETE /api/v1/users/sessions/{session_id}/branches...` - 会话分支管理

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

## 当前边界

- 独立 PNG/JPG/JPEG 支持 OCR 后按文本入库；PDF、DOCX、PPTX 内嵌图片的完整图文联合索引尚未完成。
- Sparse 检索当前是独立 BM25，不应描述为 BGE-M3 learned sparse。
- DuckDuckGo provider 仍是占位实现，联网搜索主 provider 为 Tavily/SerpApi。
- 生产环境需要收紧 CORS、替换默认密钥，并按部署环境配置 MySQL、Chroma、Redis 与模型服务。

完整的项目结构、调用链、验证命令和已知边界见 [`docs/项目现状梳理.md`](../docs/项目现状梳理.md)。

## 技术栈

- **框架**: FastAPI
- **数据库**: MySQL + Redis + Chroma
- **LLM**: LangChain + LangGraph
- **对象存储**: S3/MinIO
- **文档处理**: PyPDF, python-docx, unstructured

## 许可证

MIT
