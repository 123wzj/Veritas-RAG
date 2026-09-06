# Veritas RAG 文档树

这是 `docs/` 的唯一导航入口。文档按“项目总览 → 产品/架构 → 模块实现 → 评测与优化 → 规划”组织。后续优化某个模块时，先从本页进入对应的模块文档，再回写实现说明、评测记录和变更状态。

## 文档树

```text
Veritas RAG 文档
├── 00 项目总览
│   ├── 项目现状梳理.md                    # 当前代码、运行环境、边界和入口
│   ├── 项目学习路线与代码阅读指南.md       # 按调用链阅读源码
│   └── 成熟 Agentic RAG 工程落地设计方案.md # 长期工程目标与演进路线
│
├── 10 产品与版本规划
│   ├── V1.5 更懂用户的个人助手 PRD.md      # 当前版本需求与验收基线
│   └── 多模态解析与检索增强PRD.md          # Markdown 多模态及后续扩展规划
│
├── 20 架构与实现
│   ├── Agentic RAG 实现说明.md              # 全链路实现主文档
│   ├── 检索融合与端到端流程.md              # 检索链路专题
│   ├── 会话记忆与上下文工程实现说明.md      # 记忆、摘要、Prompt 和预算专题
│   └── 会话记忆与上下文工程.md              # 记忆设计背景与原则
│
├── 30 评测与优化记录
│   ├── RAG检索与生成评估完整流程.md         # 检索/生成评测总流程
│   ├── RAGAS答案生成评估使用说明.md         # RAGAS 指标和使用方法
│   └── T2Retrieval检索评估扩容与去重优化记录.md # 检索专项实验记录
│
└── 40 表达与面试材料
    └── Agentic RAG 项目面试准备.md          # 面试表达，不作为实现事实唯一来源
```

## 阅读和更新关系

```text
项目现状梳理
  ├── 读取 -> Agentic RAG 实现说明
  │             ├── 入库 -> parser/chunker/ingestion
  │             ├── 检索 -> 检索融合与端到端流程
  │             ├── 记忆 -> 会话记忆与上下文工程实现说明
  │             └── 评测 -> RAG 检索与生成评估完整流程
  ├── 读取 -> V1.5 PRD（需求与验收）
  └── 读取 -> 成熟 Agentic RAG 工程落地设计方案（目标架构）

模块优化
  现状梳理 -> 对应实现说明 -> 代码/迁移 -> 专项测试 -> 评测记录 -> 返回现状梳理
```

## 按模块定位

| 模块 | 首选文档 | 代码入口 | 评测/验证 |
|---|---|---|---|
| Markdown 解析与分块 | [实现说明](<Agentic RAG 实现说明.md>)、[多模态 PRD](<多模态解析与检索增强PRD.md>) | `backend/services/ingestion/parser.py`、`chunker.py` | `tests/test_markdown_multimodal.py` |
| 向量化与入库 | [实现说明](<Agentic RAG 实现说明.md>) | `backend/services/ingestion/ingestion.py`、`backend/embeddings/` | 入库测试、Chroma/MySQL 检查 |
| Dense/BM25/RRF | [检索融合流程](<检索融合与端到端流程.md>) | `backend/services/retrieval/hybrid.py` | T2Retrieval 评测 |
| 多样性与重排 | [检索融合流程](<检索融合与端到端流程.md>)、[T2 优化记录](<T2Retrieval检索评估扩容与去重优化记录.md>) | `backend/services/retrieval/diversity.py`、`reranker.py` | Recall/MRR/NDCG、Unique Parent Ratio |
| Query/路由/Reflection | [实现说明](<Agentic RAG 实现说明.md>) | `backend/graph/nodes/query_nodes.py`、`reflection_nodes.py` | `test_query_routing.py`、`test_reflection_routing.py` |
| 证据/引用/拒答 | [实现说明](<Agentic RAG 实现说明.md>) | `backend/graph/nodes/retrieval_nodes.py`、`generation_nodes.py` | `test_generation_grounding.py`、Citation 指标 |
| 会话/摘要/长期记忆 | [记忆实现说明](<会话记忆与上下文工程实现说明.md>) | `backend/services/memory/`、`backend/services/chat/` | `test_memory_context_engineering.py`、V1.5 测试 |
| 上下文预算 | [记忆实现说明](<会话记忆与上下文工程实现说明.md>) | `backend/services/context/context_assembler.py` | token、裁剪和摘要保留率 |
| Trace/Feedback | [V1.5 PRD](<V1.5 更懂用户的个人助手 PRD.md>)、[成熟方案](<成熟 Agentic RAG 工程落地设计方案.md>) | `backend/services/trace_service.py`、`backend/api/v1/endpoints/rag.py` | `test_v15_trace.py` |
| Markdown 能力边界 | [项目现状](<项目现状梳理.md>)、[V1.5 PRD](<V1.5 更懂用户的个人助手 PRD.md>) | `backend/api/v1/endpoints/knowledge.py` | `test_v15_markdown_only.py` |
| 前端聊天/知识库/记忆 | [V1.5 PRD](<V1.5 更懂用户的个人助手 PRD.md>) | `frontend/src/pages/`、`frontend/src/services/` | `npm run build` |

## 文档职责和事实优先级

1. 代码和数据库迁移是运行事实。
2. `项目现状梳理.md` 是当前状态摘要和已知边界。
3. `Agentic RAG 实现说明.md` 是当前主链路实现细节。
4. 专题实现文档负责深入一个模块，不重复维护整个系统概览。
5. PRD 负责需求范围和验收，不代表功能已经实现；完成状态必须回写现状文档。
6. 评测记录负责实验数据和参数，不直接替代实现说明。
7. 面试文档是表达材料，若与代码冲突，以代码和现状文档为准。

## 模块文档拆分规则

当单个文档超过约 400 行，或一个模块出现 3 个以上独立优化主题时，拆成：

```text
模块总览.md
├── 模块-数据模型.md
├── 模块-运行流程.md
├── 模块-策略与参数.md
├── 模块-测试与评测.md
└── 模块-优化记录.md
```

总览文档只保留职责、调用链、关键参数、当前状态和子文档链接；详细实现放入子文档，避免复制粘贴导致事实漂移。

## 模块优化时的固定流程

1. 在本页确定模块和首选文档。
2. 阅读现状、PRD 和对应实现专题。
3. 修改代码、迁移、测试和文档锚点。
4. 运行 `conda run -n cook-rag-1 pytest -q` 与 `frontend/npm run build`。
5. 将参数、指标、已知边界和未完成项写入对应专题/评测记录。
6. 更新 `项目现状梳理.md` 的状态摘要。
7. 提交时说明影响的模块和文档节点。

## 当前建议的下一批模块

```text
1. Trace/Span：从事件汇总升级为统一埋点和 token/cost 统计
2. Memory：确认态、冲突、过期、tombstone 和跨项目隔离
3. Markdown Ingestion：版本、hash、增量索引和结构化 block
4. Evidence：claim-level citation 与拒答质量
5. Session：分支、归档、恢复和并发 checkpoint
6. Tool Gateway：为后续 MCP/Skill 预留权限、超时、幂等和审计
```
