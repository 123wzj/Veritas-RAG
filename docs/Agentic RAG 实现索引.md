# Agentic RAG 实现索引

## 文档入口

| 主题 | 说明 | 文件 |
|------|------|------|
| 实现说明 | 当前项目整体架构、端到端链路和关键实现细节 | `docs/Agentic RAG 实现说明.md` |
| 模块与方法说明 | 各模块用了什么方法、为什么这样选、解决什么问题 | `docs/Agentic RAG 模块与方法说明.md` |
| 学习路线与代码阅读 | 新人按什么顺序看代码、每一步从哪里进到哪里出 | `docs/项目学习路线与代码阅读指南.md` |
| RAGAS 答案评估 | 如何跑端到端答案评估、指标怎么解释、数据集怎么选 | `docs/RAGAS答案生成评估使用说明.md` |
| 检索与生成评估 | 检索评估和生成评估的完整流程、当前报告和排查方式 | `docs/RAG检索与生成评估完整流程.md` |
| 检索融合与端到端流程 | Dense/Sparse 独立召回、RRF 融合、父块回补和完整问答流程 | `docs/检索融合与端到端流程.md` |
| 简历项目描述 | 面向简历和口述介绍的项目描述 | `docs/简历项目描述.md` |
| 项目面试准备 | 面试高频问题、回答思路和可背诵总结 | `docs/Agentic RAG 项目面试准备.md` |

## 当前架构速览

```text
文档上传
  -> 文档解析
  -> Parent-Child 分块
  -> child dense / sparse 表示构建
  -> child vector + metadata 写入 Chroma
  -> parent / child 正文写入 MySQL

用户提问
  -> 加载记忆
  -> Query Rewrite
  -> Query Decomposition
  -> Route Planning
      -> chat
      -> web_search
      -> knowledge_base
      -> hybrid
  -> Dense child recall
  -> Independent Sparse child recall
  -> Weighted RRF
  -> MySQL parent backfill
  -> Reranker
  -> Evidence Packing
  -> Evidence Grading
  -> Reflection，最多 3 次，到上限后必须收口
  -> Answer Generation
  -> Answer Verification
  -> 返回答案和引用
```

## 新人学习顺序

如果是第一次读这个项目，建议按这个顺序走：

1. 先看 `docs/项目学习路线与代码阅读指南.md`，知道代码入口和阅读路线。
2. 再看 `backend/core/config.py`，确认配置只从根目录 `.env` 读取。
3. 然后看 `backend/api/v1/endpoints/rag.py`，这是用户提问入口。
4. 接着看 `backend/graph/state/state.py` 和 `backend/graph/graph.py`，理解状态和流程怎么串起来。
5. 最后按节点看 `query_nodes.py`、`retrieval_nodes.py`、`reflection_nodes.py`、`generation_nodes.py`。

读每个节点时只抓三个问题：

- 它从 `RAGState` 读什么？
- 它往 `RAGState` 写什么？
- 下一个节点靠这些字段做什么？

## 核心代码位置

| 模块 | 代码位置 |
|------|----------|
| 知识库 API 入口 | `backend/api/v1/endpoints/knowledge.py` |
| RAG / SSE API 入口 | `backend/api/v1/endpoints/rag.py` |
| 用户 / 会话 API | `backend/api/v1/endpoints/users.py` |
| 记忆 API | `backend/api/v1/endpoints/memory.py` |
| 文档解析 | `backend/services/ingestion/parser.py` |
| 文档分块 | `backend/services/ingestion/chunker.py` |
| 文档入库 | `backend/services/ingestion/ingestion.py` |
| Dense Embedding | `backend/embeddings/embeddings.py` |
| Sparse Embedding | `backend/embeddings/sparse.py` |
| 配置加载 | `backend/core/config.py` |
| LLM 初始化 | `backend/graph/llm_factory.py` |
| 混合检索与融合 | `backend/services/retrieval/hybrid.py` |
| Reranker | `backend/services/retrieval/reranker.py` |
| Query 改写、拆解、路由 | `backend/graph/nodes/query_nodes.py` |
| Retrieval 节点 | `backend/graph/nodes/retrieval_nodes.py` |
| Reflection 节点 | `backend/graph/nodes/reflection_nodes.py` |
| Web Search 节点 | `backend/graph/nodes/web_nodes.py` |
| Generation / Verification 节点 | `backend/graph/nodes/generation_nodes.py` |
| Memory 节点 | `backend/graph/nodes/memory_nodes.py` |
| 状态机 | `backend/graph/graph.py` |
| 状态定义 | `backend/graph/state/state.py` |
| RAGAS 答案评估 | `backend/evaluation/run_ragas_answer_eval.py` |
| HotpotQA 导入 | `backend/evaluation/import_hotpotqa_dataset.py` |
| T2Retrieval 检索评估 | `backend/evaluation/run_t2retrieval_eval.py` |
| ACL 权限控制 | `backend/services/acl/permission.py` |
| 对话分支 | `backend/services/chat/branch.py` |
| Chroma 连接 | `backend/db/chroma/connection.py` |
| MySQL 连接和 schema sync | `backend/db/mysql/connection.py` |
| 知识库数据模型 | `backend/models/database/knowledge.py` |
| 用户 / 会话数据模型 | `backend/models/database/user.py` |

## 当前关键点

- 向量库使用 Chroma，不再保留 Milvus 旧架构。
- Chroma 只存 child chunk 向量和检索元数据。
- MySQL 存 parent / child 正文、会话、用户画像和业务元数据。
- 默认 LLM 通过 `backend/graph/llm_factory.py` 按 `backend/core/config.py` 中的 OpenAI-compatible 配置初始化。
- 本地配置只从项目根目录 `.env` 读取，不再读取 `backend/.env`，避免模型、base url、key 被第二份配置覆盖。
- LangGraph 执行时显式设置 `GRAPH_RECURSION_LIMIT`，Reflection 达到 `MAX_REFLECTION_ROUNDS` 后会收口，不再继续重试。
- Dense Embedding 默认使用本地 `BGE-M3`，Reranker 默认使用本地 `BAAI/bge-reranker-v2-m3`。
- Sparse 是应用层 Independent Sparse Recall，不是 dense 候选池里的重排。
- 检索融合使用 Weighted RRF。
- 父块回补完全通过 MySQL 的 `parent_id` 查询实现。
- 知识库访问控制由 `backend/services/acl/permission.py` 负责。
- RAG SSE 入口会自动创建 session、写入消息，并把事件流返回前端。
- 联网搜索当前支持 `Tavily / SerpApi / DuckDuckGo` provider；如果联网不可用，web 节点会关闭本轮 web 请求，避免循环。
- HotpotQA 200 条样本适合作为 RAGAS 生成评估回归基准，但企业落地还需要补中文业务 QA、无答案、权限、表格、版本日期等专项样本。

- Agent 控制流包含 Route Planning、Evidence Grading、Reflection 和 Verification。
