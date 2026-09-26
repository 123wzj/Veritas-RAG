# Veritas RAG 后端

后端基于 FastAPI、LangGraph、MySQL 和 Chroma。当前同时保留旧固定 Graph 与新受控 ReAct Runtime，运行模式由 `AGENT_RUNTIME_MODE` 控制。

## 主要目录

```text
backend/
├── agent/                 # 新 ReAct Runtime
│   ├── context/           # 动态上下文
│   ├── evidence/          # Evidence Ledger
│   ├── memory/            # ReAct 三层记忆桥接
│   └── tools/             # Registry / Policy / Gateway / Tool Adapter
├── api/v1/endpoints/      # FastAPI 路由
├── graph/                 # Legacy 与 Shadow 对照基线
├── services/
│   ├── ingestion/         # Markdown 入库与父子分块
│   ├── retrieval/         # Dense、BM25、RRF、Rerank
│   ├── memory/            # 会话与长期记忆持久化
│   └── web_search/        # 联网 Provider
├── models/                # Pydantic 与数据库模型
├── db/mysql/migrations/   # 正式数据库迁移
└── main.py                # 应用入口
```

## 运行环境

统一使用 Conda 环境 `cook-rag-1`：

```powershell
conda activate cook-rag-1
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

从仓库根目录启动，使用 `backend.*` 绝对导入。配置示例见 `backend/.env.example`，所有 LLM 角色统一使用 `deepseek-v4-flash`。

## Runtime

| 模式 | 行为 |
| --- | --- |
| `legacy` | 旧固定 Graph 返回答案，当前默认 |
| `react_shadow` | Legacy 返回答案，ReAct 只读执行并记录对比指标 |
| `react` | 新 ReAct Graph 直接返回答案 |

新图：

```text
hydrate_context → decide → act → observe → decide → verify → memory
```

当前 Trace 可查看 Run、Span、Tool Call 和 Observation，但尚不支持节点级 Checkpoint/Resume。

## 主要 API

- `POST /api/v1/rag/query`：非流式问答；
- `POST /api/v1/rag/query/stream`：SSE 流式问答；
- `GET /api/v1/knowledge/{kb_id}/capabilities`：知识库能力边界；
- `POST /api/v1/knowledge/{kb_id}/upload`：Markdown 上传；
- `/api/v1/memory/...`：长期记忆查询、确认、拒绝、编辑与删除；
- `/api/v1/users/sessions/...`：会话、分支和归档；
- `GET /api/v1/users/sessions/{session_id}/trace`：当前用户的运行 Trace。

## 数据库

部署时按顺序检查并执行：

```text
backend/db/mysql/migrations/20260906_v15_acceptance.sql
backend/db/mysql/migrations/20260926_react_runtime.sql
```

生产环境不能只依赖 `create_all` 或开发期 Schema 同步逻辑。

## 验证

```powershell
conda run -n cook-rag-1 pytest -q
```

## 当前边界

- 上传 API 仅支持 Markdown；
- Sparse 为独立 BM25，不是 BGE-M3 learned sparse；
- `backend/graph/` 仍是回退基线，切流完成前不能删除；
- 默认 Runtime 仍为 `legacy`；
- 目标数据库迁移、真实 Shadow 和生产认证尚未完成；
- Checkpoint 和失败节点恢复尚未实现。

完整说明见 [项目文档中心](../docs/README.md)。
