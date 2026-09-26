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

后端按 API、Runtime、工具/证据、业务服务和存储分层组织：

- **API 层** `api/v1/endpoints/`：问答/SSE、知识库、用户/会话、记忆与 Trace。
- **Runtime 入口** `agent/runtime.py`：在线请求只执行 `react`，旧配置值也会被归一化为 `react`。
- **新 ReAct 图** `agent/graph.py`：状态见 `agent/state.py`，稳定契约见 `agent/schemas.py`。
- **工具系统** `agent/tools/`：Registry、Policy、Gateway、`knowledge_search`、`web_search`。
- **证据与上下文** `agent/evidence/`、`agent/context/`：Evidence Ledger、引用校验和动态 Prompt。
- **记忆** `agent/memory/`、`services/memory/`：工作、短期、长期三层记忆。
- **历史图** `graph/`：不再进入在线请求，仅供尚未迁移的评测脚本参考。
- **存储** `db/`：Chroma 保存 Dense 向量，MySQL 保存业务数据、记忆、Trace 和工具审计，Redis 可选。
- **配置** `core/config.py`：`Settings` 只从根目录 `.env` 读取，不读 `backend/.env`。

### Runtime 主流程

```text
hydrate_context
  → decide
    → act → observe → decide
    → verify
      → decide（仍有预算且验证失败）
      → propose_memory_update → END
```

所有在线请求都由新 ReAct 图直接返回答案。工具调用必须经过服务端权限、预算、超时、重试与幂等校验。当前 Trace 保存 Run、Span、Tool Call 和 Observation，但没有节点级 Checkpoint/Resume。

### 数据存储职责划分

- **Chroma**：只存 Child Chunk 的稠密向量 + 轻量检索元数据（`chunk_id`/`kb_id`/`doc_id`/`parent_id`）
- **MySQL**：文档、Parent/Child 正文、层级关系、稀疏向量、元数据（事实来源）
- Parent-Child 分层切分：Parent ~1000 tokens 保留上下文，Child ~300 tokens 用于召回；检索命中 Child 后回补 Parent 再进行生成

### 检索链路

`knowledge_search` 复用 Dense（BGE-M3，Chroma）+ Sparse（BM25，MySQL）并行召回 → Weighted RRF → 文档/父块多样性控制 → Parent 回补 → `BAAI/bge-reranker-v2-m3` 精排。工具结果转为 `ToolObservation` 并进入 Evidence Ledger，最终答案只能引用其中实际存在的 `E#`。

### 上下文工程与记忆

MySQL 是会话与记忆的事实来源。记忆分为 Run-scoped 工作记忆、Session/Branch-scoped 短期记忆和 User/Project-scoped 长期记忆。每轮 `decide` 都按 System、策略、当前问题、工具、工作记忆、Observation、Evidence、会话摘要、最近消息、长期记忆和画像重新构造上下文；只有答案通过验证后才规划记忆更新。

### 模型配置

- 所有 LLM 角色统一使用 `deepseek-v4-flash`，优先控制成本。
- Embedding：`BAAI/bge-m3`（本地）；Reranker：`BAAI/bge-reranker-v2-m3`（本地）
- 联网搜索：Tavily / SerpApi / DuckDuckGo（DuckDuckGo 为占位实现）

## 关键文档

`docs/` 目录是架构与链路的权威说明。核心文档：
- `docs/01-项目现状与阅读指南.md` — 当前能力、边界和源码入口
- `docs/02-系统架构与运行流程.md` — ReAct Runtime 与会话隔离
- `docs/03-状态、追踪与失败恢复.md` — State、Trace 和 Checkpoint 差距
- `docs/04-工具系统与检索.md` — Tool Gateway、入库与检索
- `docs/05-上下文工程与记忆机制.md` — 三层记忆与动态 Prompt
- `docs/06-数据库迁移与ReAct上线验收.md` — ReAct 默认值迁移、上线验收与旧图清理

## 注意事项

- 仓库可能存在历史遗留的旧版本实现（如旧向量库），docs 说明当前运行时以 Chroma 为主，代码若与 docs 不一致以仓库真实文件为准。
- 编辑后端文件后，PostToolUse hook 会用 prettier 检查被改文件（`.claude/settings.json`），排除 `data/hf_cache`、`frontend/dist`、`data/chroma` 等目录。
- Visual 设计约束见 `.impeccable.md`（暗色、桌面优先、信息层级清晰）。
