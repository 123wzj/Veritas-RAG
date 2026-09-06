# 成熟 Agentic RAG 工程落地设计方案

> 版本：V1.0  
> 日期：2026-09-06  
> 适用项目：Veritas RAG 个人知识库助手  
> 目标：把当前 Agentic RAG 原型演进为可长期使用、可审计、可评测、可扩展的个人助手系统。

## 1. 设计结论

一个可以完整落地的 Agent，不是“一个大模型加几个工具”，而是一套受约束的任务执行系统。它至少需要：

```text
用户身份与权限
  -> 会话与任务生命周期
  -> 上下文和记忆管理
  -> 意图识别与计划
  -> 知识检索与工具调用
  -> 结果验证与安全策略
  -> 可解释输出
  -> 反馈、评测和持续改进
```

当前 Veritas RAG 已经具备 RAG、LangGraph、混合检索、证据分级、反思、会话摘要、长期记忆和基础 Trace。下一阶段不应继续堆叠 Agent 数量，而应优先完成四个工程闭环：

1. **任务闭环**：请求必须可开始、可暂停、可恢复、可取消、可失败收口。
2. **证据闭环**：每个事实都能追溯到知识库、工具结果或用户确认记忆。
3. **状态闭环**：会话、分支、记忆、任务状态不能互相污染。
4. **运营闭环**：每轮请求可观测，质量和成本可量化，反馈能进入评测集。

## 2. Agent 的完整能力模型

### 2.1 基础能力

| 能力域 | 必须解决的问题 | 成熟实现的标准 |
|---|---|---|
| 身份 | 谁在使用，能访问什么 | user/workspace/project/role/ACL 明确，所有请求鉴权 |
| 会话 | 当前对话属于哪里 | session、branch、归档、恢复和并发控制 |
| 任务 | 当前要完成什么 | goal、constraints、done criteria、状态机和取消 |
| 记忆 | 什么值得长期保留 | 来源、置信度、确认、过期、冲突和审计 |
| 上下文 | 本轮应该给模型什么 | 分区、预算、裁剪、注入防护、可解释 |
| 规划 | 下一步做什么 | 结构化计划、工具选择、步骤上限、重试策略 |
| 执行 | 如何调用检索和工具 | schema 校验、权限、超时、幂等、重试、隔离 |
| 证据 | 为什么相信结果 | evidence ID、来源、时间、可信度和引用绑定 |
| 验证 | 是否可以交付 | grounded、useful、schema、policy、citation 校验 |
| 输出 | 用户得到什么 | 答案、引用、状态、下一步、失败原因 |
| 观测 | 出问题如何定位 | trace/span、token、耗时、错误、输入输出摘要 |
| 反馈 | 系统如何变好 | 点赞/点踩、纠错、标注、离线评测、版本对比 |

### 2.2 不应交给模型的职责

模型可以负责语义判断，但不能独立决定：

- 用户是否有权限访问数据；
- 是否可以读取另一个项目的记忆；
- 是否可以执行高风险工具；
- 是否可以写入长期记忆；
- 是否可以修改数据库、代码或系统配置；
- 是否可以绕过证据、验证和步骤上限。

这些必须由服务端、策略层和数据库事务强制执行。

## 3. 推荐总体架构

```text
┌─────────────────────────────────────────────────────────┐
│                    Web / API / SSE                      │
└──────────────────────────┬──────────────────────────────┘
                           │ Auth + Request Context
┌──────────────────────────▼──────────────────────────────┐
│                    Assistant Runtime                    │
│  Session Manager | Task Manager | Policy Guard           │
│  Context Builder | Planner | Executor | Verifier         │
└───────┬───────────────┬───────────────┬──────────────────┘
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│ Knowledge    │ │ Memory      │ │ Tool Gateway    │
│ Parser/Index │ │ Store/Policy│ │ MCP/Skill/HTTP  │
└───────┬──────┘ └──────┬──────┘ └─────┬──────────┘
        │               │               │
┌───────▼────────────────▼──────────────▼──────────┐
│ MySQL: business/state/audit | Chroma: vectors    │
│ Object storage: source files | Trace backend     │
└──────────────────────────────────────────────────┘
```

### 3.1 分层职责

1. **API 层**：鉴权、参数校验、SSE、错误码，不编排业务细节。
2. **Runtime 层**：创建 Run、恢复状态、调用 Graph、取消和收口。
3. **Graph 层**：负责节点编排和条件路由，不直接操作外部权限。
4. **Policy 层**：工具权限、记忆写入、证据使用和风险控制。
5. **Service 层**：检索、记忆、工具、文档、Trace 等可测试能力。
6. **Storage 层**：事实数据、派生状态、向量、文件、审计和追踪分开保存。

## 4. 一次 Agent Run 的标准生命周期

```text
CREATED
  -> AUTHORIZED
  -> CONTEXT_LOADED
  -> PLANNED
  -> EXECUTING
  -> VERIFYING
  -> COMMITTING
  -> COMPLETED

任意阶段可进入：PAUSED / CANCELLED / FAILED
```

每个 Run 必须有：

- `run_id`：一次执行实例；
- `request_id`：用户可追踪的请求 ID；
- `parent_run_id`：重试、分支或子任务来源；
- `user_id/workspace_id/project_id/session_id/branch_id`；
- `goal`、`constraints`、`plan`、`status`；
- `max_steps`、`deadline`、`budget`；
- `created_at/started_at/completed_at`；
- `cancelled_by/cancel_reason/error_code`。

### 4.1 幂等和恢复

- 同一 `request_id` 不重复写入 user/assistant 消息。
- 工具调用使用 `idempotency_key`，重试不能重复扣费或重复写数据。
- 节点完成后写 checkpoint，进程重启可以从最近安全点恢复。
- 用户断开 SSE 不等于 Run 取消；取消必须有明确 API。
- 达到步骤、时间、token 或费用上限后必须进入可解释失败状态。

## 5. 当前项目的 Agent 主流程设计

### 5.1 V1.5 单 Agent 图

```text
load_request_context
  -> load_session_memory
  -> rewrite_query
  -> decompose_query
  -> plan_route
  -> retrieve_markdown_kb
       ├─ dense_recall
       ├─ bm25_recall
       ├─ weighted_rrf
       ├─ diversity_select
       ├─ parent_backfill
       └─ rerank
  -> grade_evidence
  -> reflect_or_search_web (optional)
  -> assemble_generation_context
  -> generate_answer
  -> verify_answer
  -> propose_memory_update
  -> policy_check_and_commit
  -> emit_trace_and_response
```

### 5.2 规划输出必须结构化

规划模型只输出 JSON，服务端用 Pydantic 校验：

```json
{
  "goal": "回答用户问题",
  "intent": "knowledge_qa",
  "answer_slots": [
    {"id": "slot-1", "question": "...", "required": true}
  ],
  "steps": [
    {"id": "step-1", "kind": "retrieve", "target": "slot-1"}
  ],
  "stop_conditions": ["all_required_slots_supported", "budget_exhausted"],
  "risk_level": "low"
}
```

服务端必须检查：步骤数量、工具白名单、目标 slot 是否存在、是否包含停止条件、是否超过预算。

## 6. 知识库和 RAG 设计

### 6.1 V1 入库范围

当前只开放 Markdown：

```text
.md -> block parser -> heading/paragraph/table/code/image
   -> Parent chunks -> Child chunks
   -> BGE-M3 Dense + BM25
   -> Chroma + MySQL
```

其他文件格式待后续版本，不应在 API、前端和文档中宣称已完整支持。

### 6.2 推荐索引模型

每个索引对象保留：

- `workspace_id/project_id/kb_id/document_id/document_version`；
- `block_id/parent_id/chunk_id`；
- `heading_path/page_or_line`；
- `content_hash`、`parser_version`、`chunker_version`；
- `modality`、`language`；
- `dense_vector`、`sparse_terms`；
- `source_uri` 和 ACL 标签。

### 6.3 检索质量保护

- Dense 与 BM25 分别评测，不只看融合结果。
- RRF 后执行 doc/parent diversity，防止一个文档占满候选。
- Parent 回补不应覆盖原始 Child 命中片段。
- Reranker 输入应包含标题、章节、Parent 和 Child。
- Evidence Grading 必须按 answer slot 判断，不按整体相似度判断。
- 引用只允许来自 `allowed_citation_ids`。
- 文档更新使用版本号和 hash，删除需要同时删除向量、正文和缓存。

## 7. 记忆系统设计

### 7.1 四层记忆

| 层 | 内容 | 生命周期 | 是否自动写入 |
|---|---|---|---|
| 最近消息 | 最近 2~3 轮原文 | 会话 | 自动保存 |
| Session Summary | 目标、决定、否定项、未决问题 | 会话 | 达到阈值更新 |
| Working Memory | 当前任务、约束、实体 | 单次 Run | 不持久化 |
| Long-term Memory | 偏好、画像、项目状态 | 跨会话 | 提议后确认/策略通过 |

### 7.2 长期记忆状态机

```text
PROPOSED -> PENDING_CONFIRMATION -> ACTIVE
                    ├────────────> REJECTED
ACTIVE -> INACTIVE / EXPIRED / DELETED / SUPERSEDED
```

每条记忆至少包含：

```text
memory_id, user_id, workspace_id, project_id, scope,
type, content, source, confidence, status,
created_at, updated_at, last_confirmed_at, expires_at,
source_session_id, source_message_id, superseded_by
```

### 7.3 记忆使用策略

```text
作用域过滤 -> 状态/过期过滤 -> 词法预筛 -> 语义/模型选择
            -> Policy 检查 -> 注入 Context -> 记录 selected_memory_ids
```

记忆优先级：用户明确确认 > 用户明确表达 > 项目决定 > 模型推断。冲突记忆不能静默覆盖，必须保留历史并降低置信度或请求用户确认。

## 8. 工具、MCP 和 Skill 的工程设计

### 8.1 Tool Gateway

所有工具都通过统一网关，不允许 Graph 节点直接访问任意 HTTP、文件系统或数据库。

工具注册信息：

```text
name, version, description, input_schema, output_schema,
permission, risk_level, timeout_ms, retry_policy,
idempotent, cost_class, audit_policy
```

执行流程：

```text
Planner 选择工具
  -> Tool Gateway 校验 schema
  -> Policy 校验用户/项目/风险
  -> 用户确认（高风险）
  -> 执行、超时、重试和结果校验
  -> 转成 Evidence
  -> 写入 Trace
```

### 8.2 MCP 接入

MCP 只作为外部工具/资源适配层：

- 每个 MCP server 有独立身份和权限；
- 工具 schema 必须转成内部统一 schema；
- MCP 返回结果先做大小限制、敏感信息过滤和来源标记；
- 不允许 MCP 直接写长期记忆或改变系统策略；
- 记录 server、tool、参数摘要、结果摘要和耗时。

### 8.3 Skill 接入

Skill 是面向任务的高层工作流，不是任意 Prompt 文件。每个 Skill 必须包含：

- 目标和适用条件；
- 输入/输出 schema；
- 可用工具白名单；
- 步骤上限和失败策略；
- 所需记忆类型；
- 验收检查；
- 版本和变更记录。

## 9. 安全与治理

### 9.1 必须防护的风险

- Prompt Injection：文档、网页、历史消息只能作为 data；
- 越权检索：所有 query 带 user/kb/ACL 过滤；
- 记忆污染：模型不能直接持久化事实；
- 工具滥用：高风险操作需要确认；
- 数据泄露：Trace 脱敏，日志不存完整敏感正文；
- 无限循环：步骤、反思、时间和 token 上限；
- 供应链风险：MCP/Skill 版本、来源和权限可审计。

### 9.2 Policy Guard

Policy Guard 至少提供：

```text
can_read_kb(user, kb, resource)
can_use_memory(user, memory, session)
can_call_tool(user, tool, args)
can_write_memory(plan, evidence, user_confirmation)
can_publish_answer(answer, evidence, policy)
```

## 10. 可观测性设计

### 10.1 Trace 结构

```text
Trace(run)
├── request_context
├── plan
├── spans[]
├── evidence_summary
├── memory_summary
├── answer_summary
├── verification
└── cost_and_latency
```

Span 推荐名称：

`auth`、`memory.load`、`query.rewrite`、`query.decompose`、`route.plan`、`retrieval.dense`、`retrieval.sparse`、`rrf`、`rerank`、`evidence.grade`、`tool.call`、`reflection`、`generation`、`verification`、`memory.update`。

### 10.2 关键指标

质量：Recall@K、MRR、NDCG、Citation Precision、Citation Recall、Faithfulness、拒答 precision、记忆选择准确率。  
性能：首 token 延迟、总耗时、P50/P95、各节点耗时、并发数、失败率。  
成本：输入/输出 token、模型调用次数、工具调用次数、每轮估算费用。  
安全：越权拒绝数、Prompt Injection 拦截数、错误记忆写入数、敏感数据脱敏数。

### 10.3 用户可见运行详情

默认折叠显示：

- request_id/run_id；
- 路由和回答模式；
- 检索来源及 evidence IDs；
- 使用的 memory IDs；
- 各阶段耗时；
- 反思次数和停止原因；
- 验证结果和反馈入口。

不向用户展示隐藏思维链，只展示可验证的过程摘要和状态。

## 11. 评测体系

### 11.1 数据集分层

1. 单事实 Markdown 问答；
2. 多段、多跳和跨章节问题；
3. 无答案和证据不足问题；
4. 冲突版本和时间敏感问题；
5. 长对话指代和记忆问题；
6. 会话、分支和项目隔离问题；
7. Prompt Injection 和恶意工具参数问题。

### 11.2 评测门禁

每次修改至少运行：

```bash
conda run -n cook-rag-1 pytest -q
cd frontend && npm run build
```

上线前需要比较基线和候选版本：

```text
检索：Recall/MRR/NDCG/Unique Parent Ratio
回答：Faithfulness/Citation/Refusal/Completeness
记忆：Selection Accuracy/Write Error/Conflict Rate
体验：P50/P95/首 token/失败率/成本
```

任何核心指标下降超过阈值，禁止直接上线。

## 12. 当前项目需要优先优化的工程项

### P0：先把边界做硬

- 继续保持 Markdown-only；
- 所有 LLM 固定 `deepseek-v4-flash`；
- 统一错误码和响应 schema；
- 修正文档和运行配置中的过期模型描述；
- 给 Run 增加取消、超时、步骤和 token 预算。

### P1：把个人助手核心做稳

- 完善 workspace/session/branch 数据边界；
- 记忆新增来源、确认、过期、冲突和 tombstone；
- 前端完成记忆确认卡片、编辑、删除、审计；
- 长对话摘要增加漂移检测；
- 真实 tokenizer 和上下文预算记录。

### P2：把 Trace 做成生产能力

- Run/Span 正式持久化；
- 每个图节点统一埋点，而不是仅依赖事件映射；
- 记录模型、token、候选、证据、记忆和停止原因；
- 增加 Trace 查询、导出和脱敏策略；
- SSE 断开后 Run 仍能查询状态。

### P3：把质量做成门禁

- 建立 Markdown 真实数据集；
- 增加 citation-level 自动检查；
- 增加记忆污染和隔离测试；
- 记录反馈并自动生成待标注样本；
- 对每次 RRF、chunk、prompt 和模型修改做 A/B 回归。

### P4：再接工具生态

- Tool Gateway；
- 内部工具 schema 和权限模型；
- MCP adapter；
- Skill manifest 和版本管理；
- 高风险操作确认和审计。

### P5：最后考虑多 Agent

推荐结构：

```text
Supervisor
├── Retrieval Worker
├── Memory Worker
├── Tool Worker
└── Verification Worker
```

多 Agent 只有在单 Agent + Tool Gateway 明确遇到上下文、权限或并行执行瓶颈时才引入。所有 Worker 必须共享 Run/Trace、预算、权限和最终验证，不允许各自维护不可见状态。

## 13. 推荐工程目录

```text
backend/
├── api/v1/endpoints/
├── runtime/
│   ├── run_manager.py
│   ├── checkpoint.py
│   └── cancellation.py
├── graph/
│   ├── graph.py
│   ├── state/
│   └── nodes/
├── policy/
│   ├── authorization.py
│   ├── tool_policy.py
│   └── memory_policy.py
├── services/
│   ├── ingestion/
│   ├── retrieval/
│   ├── memory/
│   ├── context/
│   ├── tools/
│   └── trace_service.py
├── models/database/
├── models/schemas/
└── evaluation/
```

## 14. 成熟度验收标准

只有满足以下条件，才可以称为“可长期使用的 Agentic RAG”，而不是 Demo：

- 请求失败、超时、取消和重试都有明确状态；
- 任何答案都能追溯到证据、工具结果或用户确认记忆；
- 不同用户、项目、知识库、会话和分支零越权；
- 长对话不会无限增长，摘要和上下文预算有回归测试；
- 记忆可查看、可确认、可删除、可过期、可审计；
- 工具调用具备 schema、权限、超时、重试和幂等；
- 每轮 Run/Span 可查询，敏感数据默认脱敏；
- 检索、回答、拒答、记忆和延迟都有基线指标；
- 代码、数据库迁移、前端构建和测试可重复执行；
- 新增 MCP、Skill 或 Agent 不会绕过现有 Policy、Trace 和 Verification。

## 15. 最终产品路线

```text
V1       Markdown 知识库 + 证据型问答
V1.5     个人助手 + 记忆治理 + 会话隔离 + Trace
V2       Tool Gateway + 日历/文件/网页等受控工具
V2.5     MCP Adapter + Skill Manifest
V3       Supervisor + 多 Agent 协作
```

核心原则：

> 先让单 Agent 可靠，再让它会用工具；先让状态可追踪，再让系统自动化；先让记忆可治理，再谈自进化；先让评测可重复，再扩大模型和 Agent 数量。
