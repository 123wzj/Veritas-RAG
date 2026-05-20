# Veritas RAG

基于 LangChain + LangGraph + Chroma 的智能问答系统，支持知识库检索、联网搜索、用户记忆等功能。

## 项目结构

```
veritas-rag/
├── backend/                # FastAPI 后端服务
│   ├── api/               # API 路由层
│   │   ├── deps/          # 依赖注入（认证、数据库）
│   │   └── v1/endpoints/  # RESTful API 端点
│   ├── core/              # 核心配置模块
│   ├── models/            # 数据模型（Pydantic schemas, Database tables）
│   ├── services/          # 业务逻辑层
│   │   ├── ingestion/     # 文档入库服务
│   │   ├── retrieval/     # 混合检索服务
│   │   ├── memory/        # 用户记忆服务
│   │   └── web_search/    # 联网搜索服务
│   ├── graph/             # LangGraph 状态机
│   │   ├── nodes/         # 节点实现
│   │   ├── state/         # 状态定义
│   │   └── llm_factory.py # LLM 工厂
│   ├── db/                # 数据库连接
│   ├── embeddings/        # 向量化模块
│   └── main.py            # 应用入口
├── frontend/              # React + TypeScript 前端
│   └── src/
│       ├── components/    # UI 组件
│       ├── pages/         # 页面（聊天、知识库、设置）
│       ├── stores/        # Zustand 状态管理
│       ├── services/      # API 服务层
│       └── types/         # TypeScript 类型
└── agentic-rag-dev-doc.md # 开发文档
```

## 核心功能

### 1. LangGraph Agentic 状态机

14 个节点实现完整的智能问答流程：

| 节点 | 功能 | 状态 |
|------|------|------|
| load_memory | 加载用户画像和偏好 | ✅ |
| rewrite_query | 问题改写和优化 | ✅ |
| decompose_query | 复杂问题拆解 | ✅ |
| retrieve | 混合检索（Dense + Sparse） | ✅ |
| rerank | 候选重排 | ✅ |
| pack_evidence | 证据打包 | ✅ |
| judge_evidence | 证据充分性判断 | ✅ |
| reflect | 反思与自纠 | ✅ |
| web_search | 联网搜索增强 | ✅ |
| generate_answer | 答案生成 | ✅ |
| verify_answer | 答案验证 | ✅ |
| write_memory | 记忆写入 | ✅ |

### 2. 混合检索服务

- **Dense 检索**: 基于语义向量的检索
- **Sparse 检索**: 基于词法特征的关键词检索
- **RRF 融合**: Rank-based Reciprocal Fusion 融合算法
- **Parent 回补**: 使用父级 Chunk 提供完整上下文

### 3. 向量化模块

支持多种 Embedding 模型：
- **Qwen (通义千问)**: text-embedding-v3（默认）
- **OpenAI**: text-embedding-3-large
- **HuggingFace**: shibing624/text2vec-base-chinese（本地）

### 4. 用户记忆模块

- 用户画像（偏好语言、交互风格、兴趣领域）
- 会话历史记录
- 常问主题统计

### 5. 联网搜索

- 支持 Tavily API
- 可在前端开关控制

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI + Python 3.10+ |
| **LLM 框架** | LangChain + LangGraph |
| **数据库** | MySQL (元数据) + Chroma (向量) |
| **前端** | React 18 + TypeScript + Vite |
| **UI 组件** | shadcn/ui + Tailwind CSS |
| **状态管理** | Zustand |
| **LLM 模型** | Qwen (通义千问) |
| **Embedding** | Qwen text-embedding-v3 |

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- MySQL 8.0+
- 通义千问 API Key: [申请地址](https://dashscope.aliyun.com/)

### 后端启动

```bash
cd backend

# 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 配置环境变量（编辑 .env 文件）
# LLM_API_KEY=sk-xxxxx
# EMBEDDING_API_KEY=sk-xxxxx

# 创建数据库
mysql -u root -p -e "CREATE DATABASE agentic_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 初始化数据库并添加默认用户
mysql -u root -p agentic_rag < scripts/add_default_user.sql

# 启动服务
python -m uvicorn main:app --reload
```

### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 配置 API 地址（可选，默认 http://localhost:8000）
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# 启动开发服务器
npm run dev
```

访问: http://localhost:5173

## 配置说明

### 后端环境变量 (.env)

```bash
# ========== 数据库配置 ==========
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=agentic_rag

# ========== Qwen 模型配置 ==========
LLM_PROVIDER=qwen
LLM_API_KEY=sk-your-api-key
LLM_MODEL=qwen-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# ========== Embedding 配置 ==========
EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=sk-your-api-key
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSION=1024

# ========== 联网搜索配置 ==========
WEB_SEARCH_ENABLED=false
WEB_SEARCH_PROVIDER=tavily
WEB_SEARCH_API_KEY=tvly-your-api-key
```

## API 文档

启动后端后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要端点

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/rag/query/stream` | 流式 RAG 查询 |
| GET | `/api/v1/users/me` | 获取当前用户 |
| GET | `/api/v1/users/me/profile` | 获取用户画像 |
| PUT | `/api/v1/users/me/profile` | 更新用户画像 |
| GET | `/api/v1/knowledge/` | 获取知识库列表 |
| POST | `/api/v1/knowledge/` | 创建知识库 |
| GET | `/api/v1/memory/` | 获取用户记忆 |

## 功能演示

### 聊天界面
- 支持流式输出
- 实时显示思考过程
- 联网搜索开关
- 知识库检索开关
- 引用来源展示

### 设置页面
- 用户偏好语言
- 交互风格选择（简洁/详细/友好）
- 兴趣领域管理

## 开发状态

### ✅ 已完成
- [x] LangGraph 状态机（14 节点）
- [x] 混合检索服务（Dense + Sparse + RRF）
- [x] 向量化模块（Qwen/OpenAI/HuggingFace）
- [x] 用户记忆模块
- [x] 联网搜索服务（Tavily）
- [x] 前端聊天界面
- [x] 前端设置页面
- [x] SSE 流式响应
- [x] 默认用户认证（无需登录）
- [x] 知识库检索开关控制
- [x] 知识库管理页面
- [x] 文档上传功能（集成 Ingestion Pipeline）
- [x] 会话历史列表
- [x] 图片 OCR 与多模态检索（支持 Qwen Vision 和 PaddleOCR）
- [x] Reranker 模型集成（支持 Qwen Reranker 和 Cohere Reranker）
- [x] 知识库访问权限控制（基于 ACL 的权限管理）
- [x] 导出对话记录（支持 JSON、Markdown、TXT 格式）
- [x] 对话分支管理（支持创建、切换、合并分支）

### 🚧 待完善
- [ ] 单元测试覆盖
- [ ] 前端对话分支管理 UI

### 📋 计划中
- [ ] 知识库访问权限管理 UI
- [ ] 多模态向量检索优化

## 常见问题

### 1. 向量化失败怎么办？

检查 `EMBEDDING_API_KEY` 是否正确配置，确保使用了通义千问的 API Key。

### 2. 知识库检索没有结果？

- 确保已上传文档到知识库
- 检查 Chroma 本地持久化目录和向量入库是否正常
- 尝试调整检索参数（Top-K、阈值等）

### 3. 联网搜索不工作？

- 检查 `WEB_SEARCH_API_KEY` 是否配置
- 确保前端已开启"联网搜索"开关
- 验证 Tavily API 配额是否充足

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
