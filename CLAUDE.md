# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Veritas RAG 是一个面向私有知识库的 Agentic RAG 问答系统，由 LangGraph 编排的完整链路：文档入库 → 混合检索 → 上下文工程 → 证据判断 → 答案生成 → 反思验证 → 记忆更新。前端为 React + TypeScript，后端为 FastAPI + LangGraph。

## 常用命令

后端与前端通过 `start-dev.bat` 启动（依赖 conda 环境 `cook-rag-1`，已在批处理中配置）。手动启动：

```bash
# 后端（需先激活 conda 环境 cook-rag-1）
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000  # 默认 http://localhost:8000，文档在 /docs
# 前端
cd frontend && npm install && npm run dev
```

前端脚本（`frontend/package.json`）：
- `npm run dev` — 启动 Vite 开发服务器
- `npm run build` — `tsc -b && vite build`
- `npm run lint` — ESLint

测试文件位于 `tests/`，其中既有 unittest 风格测试，也可由 pytest 统一收集；需要真实 LLM 的测试运行前需设置 `LLM_API_KEY`/`DEEPSEEK_API_KEY`：

```bash
python -m unittest tests.test_query_routing -v
python -m unittest discover -s tests
# 或
pytest -q
```

## 架构总览

后端按 分层 + 图编排 组织（`backend/`）：

- **API 层** `api/v1/endpoints/`：`rag.py`（问答/SSE）、`knowledge.py`（知识库/上传）、`users.py`（用户/会话）、`memory.py`（记忆）
- **图编排** `graph/`：`graph.py` 定义 LangGraph 状态机，`state/state.py` 定义 `RAGState`（TypedDict，节点间传递所有中间结果）
- **节点** `graph/nodes/`：`query_nodes`（改写/拆解/路由）、`retrieval_nodes`（检索/证据）、`reflection_nodes`（反思）、`generation_nodes`（生成/验证）、`memory_nodes`（记忆）、`web_nodes`（联网搜索）
- **服务层** `services/`：`ingestion/`（解析/切分/入库）、`retrieval/`（hybrid 融合/精排/多样性）、`memory/`（记忆服务）、`context/`（上下文拼装）、`web_search/`、`chat/`、`acl/`、`vision/`
- **存储** `db/`：`chroma/`（向量）、`mysql/`（结构化）、`redis/`（可选）
- **配置** `core/config.py`：`Settings`（pydantic-settings），只从根目录 `.env` 读取，**不读** `backend/.env`

### LangGraph 主流程（`graph/graph.py`）

```
load_memory → rewrite_query → decompose_query → plan_query_route
  → (retrieve → rerank → pack_evidence → judge_evidence)
  → reflect / web_search（不足或失败时的循环）
  → load_generation_memory → generate_answer → verify_answer
  → write_memory → END
```

条件路由：`plan_query_route` 按意图决定走 `retrieve`/`web_search`/`generate`；`judge_evidence` 证据不足时进 `reflect`；`verify_answer` 校验失败时回 `reflect`。反思由 `MAX_REFLECTION_ROUNDS`（默认 3）和 `GRAPH_RECURSION_LIMIT` 限制防死循环。

### 数据存储职责划分

- **Chroma**：只存 Child Chunk 的稠密向量 + 轻量检索元数据（`chunk_id`/`kb_id`/`doc_id`/`parent_id`）
- **MySQL**：文档、Parent/Child 正文、层级关系、稀疏向量、元数据（事实来源）
- Parent-Child 分层切分：Parent ~1000 tokens 保留上下文，Child ~300 tokens 用于召回；检索命中 Child 后回补 Parent 再进行生成

### 检索链路

混合检索：Dense（BGE-M3，Chroma）+ Sparse（BM25，MySQL）并行召回 → Weighted RRF 融合 → 文档/父块多样性控制 → Parent 回补 → `BAAI/bge-reranker-v2-m3` 精排 → 证据打包。证据按 `direct_support`/`partial_support`/`background_only`/`no_support`/`conflict` 分级，只有 direct_support 能用于确定结论和引用。

### 上下文工程与记忆

MySQL 是会话与记忆的事实来源。记忆分层：最近对话（2-3 轮）、会话摘要、工作记忆（当前请求）、长期记忆（跨会话）。长期记忆不全量注入，先按作用域筛选 → 词法预排序 → DeepSeek 选择 top-k。Prompt 拼装按优先级：当前问题 → 证据 → 工作记忆 → 会话摘要 → 最近对话 → 长期记忆 → 用户信息，各区域有独立 token 预算。记忆更新只在答案通过验证后、于事务中写入。

### 模型分工

- 生成/反思/持久记忆：`deepseek-v4-pro`
- 结构化判断/验证等高频率任务：`deepseek-v4-flash`
- Embedding：`BAAI/bge-m3`（本地）；Reranker：`BAAI/bge-reranker-v2-m3`（本地）
- 联网搜索：Tavily / SerpApi / DuckDuckGo（DuckDuckGo 为占位实现）

## 关键文档

`docs/` 目录是架构与链路的权威说明，`backend-code-nav` skill（`.claude/skills/backend-code-nav/SKILL.md`）据此把问题路由到具体文件。核心文档：
- `docs/Agentic RAG 实现说明.md` — 当前生效的新架构
- `docs/检索融合与端到端流程.md` — 检索参数与完整链路
- `docs/会话记忆与上下文工程实现说明.md` — 记忆与 Prompt 拼装
- `docs/RAG检索与生成评估完整流程.md`、`docs/RAGAS答案生成评估使用说明.md` — 评估流程

## 注意事项

- 仓库可能存在历史遗留的旧版本实现（如旧向量库），docs 说明当前运行时以 Chroma 为主，代码若与 docs 不一致以仓库真实文件为准。
- 编辑后端文件后，PostToolUse hook 会用 prettier 检查被改文件（`.claude/settings.json`），排除 `data/hf_cache`、`frontend/dist`、`data/chroma` 等目录。
- Visual 设计约束见 `.impeccable.md`（暗色、桌面优先、信息层级清晰）。
