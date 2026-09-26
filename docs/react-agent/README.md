# V1.6 ReAct Agentic RAG 架构重构需求

上级导航：[`docs/README.md`](../README.md)

子文档：

- [ReAct 运行时与工具协议](<ReAct运行时与工具协议.md>)
- [上下文与三层记忆设计](<../会话记忆与上下文工程.md>)
- [上下文与三层记忆实现契约](<../会话记忆与上下文工程实现说明.md>)
- [迁移计划与验收标准](<迁移计划与验收标准.md>)

> 版本：V1.6 架构基线  
> 日期：2026-09-26
> 状态：核心代码完成，默认 `legacy`，待数据库迁移与真实 Shadow 验收
> 范围：个人知识库问答、多轮对话、知识库检索、联网搜索、三层记忆、引用、拒答与逐轮可观测性。暂不包含长任务、后台任务编排、多 Agent 协作和高风险写工具。

## 1. 为什么需要重构

当前项目是“预定义 LangGraph 工作流 + 条件路由 + Reflection”：

```text
load_memory
  -> rewrite_query
  -> decompose_query
  -> plan_query_route
  -> retrieve / web_search / direct
  -> rerank / evidence judge
  -> reflection
  -> generate / verify
  -> write_memory
```

它已经具备 Agentic RAG 的部分特征，但不属于目标 ReAct Runtime，主要差异是：

1. 知识库检索和联网搜索是 Graph 的固定节点，不是统一注册、统一校验的 Tool。
2. 路由、检索、证据判断和反思分别由多个固定节点驱动，模型不能基于每次工具 Observation 自主选择下一步动作。
3. `reflection_count` 只允许回到预设路径，没有统一的 `Thought/Decision -> Action -> Observation` 循环状态。
4. 工具没有统一的输入/输出 schema、权限、超时、重试、幂等、预算与审计协议。
5. 上下文以多个节点分别装配为主，缺少 ReAct 每轮迭代时的 scratchpad、Evidence Ledger 和 Observation 压缩策略。
6. 记忆系统虽然已有 session summary 和 long-term memory，但尚未完全融入每轮 Agent 决策与冲突/时效治理。

本次重构不追求“让模型更自由”，而是把固定 RAG 流程改造成**有工具、有观察、有停止条件、受策略约束的有界 ReAct Agent**。

### 1.1 2026-09-26 实现状态

已落地：

- `backend/agent/` 下的 schema、Working Memory v2、Evidence Ledger、Context Builder、Controller、Verifier 和新 LangGraph。
- Tool Registry/Gateway/Policy、数据库幂等审计、`knowledge_search` 与 `web_search`。
- `legacy/react_shadow/react` Dispatcher；线上 API 不再直接导入旧 Graph。
- Shadow 只读执行、独立 Run ID、deadline、答案 hash/引用数量/迭代/停止原因指标。
- 用户、session、branch、kb 四层作用域校验；活动分支自动解析，主线与分支历史隔离。
- 长期记忆的 valid/stale/expiry、scope、conflict group、pending confirmation 和 tombstone 规则。
- Trace API/前端运行详情显示 runtime、预算 profile、Tool 数、迭代和停止原因。

仍未完成：

- 尚未在目标 MySQL 执行 `20260926_react_runtime.sql`。
- 尚未用真实业务问题运行 `react_shadow` 并达到切流门槛，因此默认仍是 `legacy`。
- 当前身份依赖仍是项目原有的开发用户实现；Runtime 内部隔离已完成，但生产多用户上线前仍需接入真实认证。
- 分支拥有独立最近消息，但当前 schema 没有独立 branch summary；为避免污染，分支轮次暂不更新共享 session summary。

## 2. ReAct 在本项目中的定义

本项目采用以下工程化定义：

```text
Decision：根据用户问题、记忆、已知证据和预算，决定回答或调用哪个工具
Action：输出经过 schema 校验的工具调用
Observation：工具返回的结构化结果、证据覆盖、冲突、错误和建议
Loop：模型读取新的 Observation，决定继续调用工具、调整查询、回答或拒答
```

不保存、不展示模型原始思维链。系统只记录可审计的结构化决策摘要：

```json
{
  "decision": "call_tool|answer|refuse|ask_clarification",
  "reason_summary": "缺少项目当前实现的直接证据",
  "selected_tool": "knowledge_search",
  "query_strategy": "使用模块名和关键类名重新检索",
  "unresolved_slots": ["当前默认 token 预算"],
  "confidence": 0.62
}
```

`reason_summary` 是简短、面向审计的决策说明，不是模型的私有推理过程。

## 3. 目标架构

```text
┌──────────────── Web / API / SSE ─────────────────┐
│ Auth | Session | Request ID | Streaming | Cancel │
└───────────────────────┬──────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────┐
│                Assistant Runtime                 │
│ Run Manager | Context Policy | Budget | Trace    │
└───────────────────────┬──────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────┐
│                 ReAct Controller                 │
│  hydrate -> decide -> act -> observe -> decide   │
│              └──── answer / refuse ────┘         │
└──────────────┬──────────────────────┬────────────┘
               │                      │
┌──────────────▼────────────┐ ┌──────▼─────────────┐
│       Tool Gateway       │ │ Context & Memory    │
│ registry/policy/timeout  │ │ WM / Session / LTM  │
│ retry/idempotency/audit  │ │ Prompt / Compression│
└───────┬───────────┬──────┘ └────────┬────────────┘
        │           │                 │
┌───────▼──────┐ ┌──▼──────────┐ ┌────▼────────────┐
│ KB Search    │ │ Web Search  │ │ Memory Store    │
│ RRF/Rerank   │ │ Tavily/...  │ │ MySQL/Vector    │
└───────┬──────┘ └──┬──────────┘ └────┬────────────┘
        └───────────┬┴─────────────────┘
                    ▼
        Evidence Ledger / Tool Observations
                    │
          Verify -> Commit -> Response
```

### 3.1 分层职责

| 层 | 职责 | 禁止事项 |
|---|---|---|
| API | 鉴权、请求校验、SSE、取消、错误映射 | 不直接执行检索和记忆写入 |
| Runtime | 创建 Run、加载上下文、预算、恢复和收口 | 不把业务权限交给模型 |
| ReAct Controller | 调用模型、解析工具调用、循环和停止 | 不直接访问数据库/网络 |
| Tool Gateway | schema、权限、超时、重试、幂等、审计 | 不允许未注册工具绕过网关 |
| Tool Adapter | 调用现有检索/搜索服务，规范化结果 | 不生成最终答案，不写长期记忆 |
| Context/Memory | 三层记忆、上下文装配、压缩和更新计划 | 不把检索证据当用户记忆 |
| Evidence/Verifier | 管理证据 ID、覆盖度、冲突、引用和拒答 | 不允许 Memory ID 充当 Evidence ID |
| Storage/Trace | 消息、记忆、Run、Tool Call、Observation、Span | 不记录原始思维链 |

## 4. ReAct 主循环

```text
START
  -> initialize_run
  -> load_session_and_memory
  -> build_agent_context
  -> agent_decide
       ├─ tool_calls -> policy_check -> execute_tools
       │                -> normalize_observations
       │                -> update_working_memory/evidence_ledger
       │                -> compact_context_if_needed
       │                -> agent_decide
       ├─ answer -> verify_answer
       │            ├─ pass -> propose_memory_update -> commit -> END
       │            └─ fail -> agent_decide（预算允许）
       ├─ ask_clarification -> persist_turn -> END
       └─ refuse -> persist_turn -> END
```

### 4.1 每轮决策

Agent 每轮只能选择以下一种结果：

```json
{
  "type": "tool_calls|final_answer|refusal|clarification",
  "tool_calls": [],
  "answer": null,
  "reason_summary": "简短的决策依据",
  "unresolved_slots": [],
  "confidence": 0.0
}
```

若底层模型支持 native tool calling，优先使用模型原生 `tool_calls`；结构化字段由适配层补全和校验。不要使用把 Thought 文本和 Action JSON 混在一起的旧式字符串解析。

### 4.2 停止条件

满足任一条件即停止循环：

- 所有必答槽位都有足够证据，Agent 输出最终答案；
- 用户问题不需要工具，Agent 直接回答；
- 证据不足且继续调用无法提高覆盖度，执行拒答或澄清；
- 达到 `max_iterations`，默认 6；
- 达到单工具或总工具调用上限，默认 KB 3 次、Web 2 次、总调用 6 次；
- 达到输入 token、输出 token、耗时或费用预算；
- 连续两次 Action 的规范化参数相同，判定循环无进展；
- 用户取消、策略拒绝或不可恢复错误。

达到上限时不能编造答案，应返回“已尝试哪些来源、仍缺什么、为什么无法可靠回答”。

## 5. RAG 与联网搜索改造成工具

### 5.1 `knowledge_search`

模型只提供语义参数，安全参数由 Runtime 注入：

```json
{
  "query": "要检索的问题或改写后的查询",
  "search_intent": "fact|comparison|procedure|summary|analysis",
  "filters": {
    "document_ids": [],
    "section_paths": [],
    "time_range": null
  },
  "top_k": 6,
  "need_parent_context": true
}
```

Runtime 注入且模型不可覆盖：`user_id/workspace_id/kb_id/ACL/max_top_k`。

工具内部继续复用当前成熟检索链路：

```text
query normalization
  -> Dense + BM25
  -> Weighted RRF
  -> diversity
  -> parent backfill
  -> rerank
  -> evidence packing
```

工具输出不是最终回答，而是 Observation：

```json
{
  "tool_call_id": "tc-1",
  "tool_name": "knowledge_search",
  "status": "success",
  "query": "...",
  "evidence": [
    {
      "evidence_id": "E1",
      "source_type": "knowledge_base",
      "title": "...",
      "section_path": "...",
      "support_snippet": "...",
      "source_ref": {"doc_id": 1, "chunk_id": "..."},
      "retrieval_score": 0.0,
      "rerank_score": 0.0
    }
  ],
  "coverage": {
    "supported_slots": ["slot-1"],
    "missing_slots": ["slot-2"],
    "conflicts": []
  },
  "next_hint": "refine_query|try_web|answer|stop",
  "truncated": false
}
```

### 5.2 `web_search`

```json
{
  "query": "需要外部公开信息的问题",
  "recency_days": 30,
  "domains": [],
  "max_results": 5
}
```

工具只在用户允许联网且 Policy 通过时可用。输出统一转换为 Evidence，保留 URL、标题、抓取时间、发布日期（可得时）和 snippet。网页内容属于不可信数据，不能修改系统指令。

### 5.3 Agent 如何选择工具

```text
知识库中的项目事实、文档内容、内部决策 -> knowledge_search
时效性公开信息、知识库缺失且用户允许联网 -> web_search
两者都可能需要 -> 先 KB，观察缺口后再 Web
用户闲聊、写作、基于已充分上下文的问题 -> 直接回答
权限不足、问题不清楚、来源不足 -> 澄清或拒答
```

固定“先路由一次再永不改变”的方式改为每轮根据 Observation 重新决策，但 Policy 仍限制工具可用范围。

## 6. Observation 与 Evidence Ledger

工具原始结果存入外部 Run Store，Prompt 中只加载压缩后的 Observation 和入选证据。`EvidenceLedger` 是本轮事实依据的唯一索引：

```json
{
  "entries": {
    "E1": {
      "tool_call_id": "tc-1",
      "source_type": "knowledge_base",
      "claim_slots": ["slot-1"],
      "support_snippet": "...",
      "source_ref": {},
      "freshness": null,
      "conflict_group": null,
      "usable": true
    }
  },
  "slot_coverage": {
    "slot-1": {"status": "supported", "evidence_ids": ["E1"]},
    "slot-2": {"status": "missing", "evidence_ids": []}
  }
}
```

每次 Action 后执行 Observation Normalizer：去重证据、保留最高质量 snippet、检测冲突、更新槽位覆盖度和工具调用摘要。Agent 看到的是“新增观察 + 当前 Ledger 摘要”，而不是每轮重复塞入全部原始结果。

## 7. 三层记忆在 ReAct 中的位置

### 7.1 工作记忆：Run State

```json
{
  "schema_version": "wm.v2",
  "run_id": "...",
  "user_goal": "用户本轮真正要解决的问题",
  "answer_slots": [
    {"id": "slot-1", "question": "...", "required": true, "status": "missing"}
  ],
  "constraints": ["只依据当前知识库", "回答使用中文"],
  "active_entities": ["MemoryService", "ContextAssembler"],
  "attempted_actions": [
    {"tool": "knowledge_search", "query_hash": "...", "result": "partial"}
  ],
  "unresolved_slots": ["slot-1"],
  "current_focus": "补足长期记忆冲突处理证据",
  "iteration": 2
}
```

工作记忆是 ReAct 循环的显式业务状态，不包含原始思维链。它在 Observation 后由确定性代码和结构化模型输出共同更新，Run 结束后释放；Trace 只保存脱敏摘要。

### 7.2 短期记忆：Session State

```json
{
  "schema_version": "ss.v2",
  "session_goal": "本会话持续目标",
  "active_topic": "ReAct 架构重构",
  "confirmed_decisions": [],
  "corrections": [],
  "discarded_ideas": [],
  "open_questions": [],
  "referenced_entities": [],
  "summary_through_message_id": 0,
  "version": 1
}
```

它和最近消息共同解决当前会话的连续性。Agent 初始化时加载摘要和最近轮次；循环中不反复读取数据库。达到阈值后，在回答完成阶段生成新版摘要。

### 7.3 长期记忆：Cross-session Store

```json
{
  "memory_id": "...",
  "namespace": ["user", "1", "project", "10"],
  "memory_type": "user_preference|user_profile|project_context|project_decision|project_constraint|interaction_episode",
  "normalized_key": "project.embedding_model",
  "content": "项目默认使用 BGE-M3 作为 embedding 模型",
  "status": "pending_confirmation|active|stale|superseded|inactive|deleted",
  "source": "user_explicit|user_confirmed|inferred|imported",
  "confidence": 0.95,
  "valid_from": "...",
  "valid_to": null,
  "expires_at": null,
  "last_confirmed_at": "...",
  "conflict_group": "project.embedding_model",
  "superseded_by": null,
  "provenance": {"session_id": "...", "message_id": 123},
  "embedding_ref": "..."
}
```

加载顺序：作用域/权限 → 状态/有效期 → 关键词 + 向量候选 → freshness/来源/置信度加权 → 冲突过滤 → 模型从候选 ID 中选择 → 注入 Prompt 并记录 selected IDs。

长期记忆默认由 Runtime 在 ReAct 开始前选择；只有用户明确询问旧对话或大量历史时，才允许使用 `memory_search`/`conversation_search` 只读工具。记忆写入不作为普通 ReAct Tool 暴露给模型，避免 Agent 在循环中直接污染持久状态。

## 8. 长期记忆更新和冲突解决

### 8.1 更新时机

分为两条通路：

```text
Hot path：用户明确说“记住、改成、以后都……”
  -> 回答提交前生成小型更新计划
  -> 用户明确事实可 active；推断事实 pending_confirmation

Background path：普通对话结束后异步提取和合并
  -> 不阻塞回答
  -> 只生成 candidate/proposed memory
  -> 策略或用户确认后生效
```

知识库内容、联网结果和助手生成内容不能自动变成用户长期记忆。记忆事实必须能追溯到用户表达、用户确认或受信任导入源。

### 8.2 冲突算法

```text
1. 以 namespace + memory_type + normalized_key 查找同槽位记忆
2. 判断 exact duplicate / compatible / contradictory / unrelated
3. 按来源权威排序：user_explicit_current > user_confirmed
   > user_explicit_old > trusted_import > inferred
4. 同槽位、明确新事实推翻旧事实：SUPERSEDE
5. 新事实补充旧事实且不冲突：MERGE
6. 用户明确取消：INVALIDATE
7. 语义冲突但无法判断：ASK_CONFIRMATION，双方均不进入关键 Prompt
8. 用户删除：写 tombstone，阻止旧消息重新提取同一事实
```

`created_at/updated_at` 是系统时间；`valid_from/valid_to` 是事实有效时间。不能仅按“最后写入时间”解决现实事实冲突。

## 9. 上下文窗口与动态 Prompt

### 9.1 300k 是硬上限，不是目标

运行时预算必须满足：

```text
effective_input_budget = min(
  selected_profile_budget,
  model_context_limit - output_reserve - safety_margin
)
```

若目标模型窗口约 300k，可预留 16k 输出和 12k 安全余量，最大输入不超过约 272k；普通请求仍使用较小档位：

| Profile | 输入预算 | 使用场景 |
|---|---:|---|
| `chat` | 8k～16k | 闲聊、简单多轮 |
| `rag_standard` | 24k～32k | 常规知识库问答 |
| `rag_deep` | 64k～96k | 多文档综合、多轮工具观察 |
| `long_context` | 128k～192k | 明确要求回顾长历史或大量原文 |
| `max_context` | ≤272k | 用户明确要求且成本/延迟允许 |

模型实际窗口必须由 Provider Capability 配置决定，不能只因产品目标写成 300k 就绕过真实限制。

### 9.2 每轮 Prompt 结构

```text
SYSTEM KERNEL（稳定、可缓存）
  身份、行为边界、证据/引用规则、工具规则、输出协议

RUNTIME POLICY（服务端生成）
  用户/项目作用域、可用工具、预算、迭代数、联网许可

MEMORY CONTEXT
  用户确认的核心偏好
  相关长期记忆（MEM IDs）
  会话摘要
  最近消息

RUN STATE
  working memory、answer slots、已尝试动作、未解决槽位

EVIDENCE & OBSERVATIONS
  当前 Evidence Ledger 摘要
  最近一轮 Tool Observation

CURRENT USER INPUT
  当前问题原文
```

所有动态块使用结构化边界：

```text
<CONTEXT_SECTION name="long_term_memory" trust="memory" version="1">
[MEM1] scope=user; status=active; valid_to=null; content=...
</CONTEXT_SECTION>
```

工具/文档/历史消息中的指令都是数据，不得覆盖 System Kernel。

### 9.3 动态装配优先级

```text
P0 必须：System、Runtime Policy、当前问题、工具 schema、输出协议
P1 必须：working memory、未解决槽位、最近 Observation
P2 高：支持当前答案的 Evidence Ledger
P3 中：session summary、最近消息、选中的长期记忆
P4 低：旧 Observation、历史 recall、非关键画像字段
```

预算不足时按 P4 → P3 → 旧证据顺序压缩；不得裁掉当前问题、工具协议、未解决槽位和必需证据。完整工具结果存 Run Store，Prompt 只保留可恢复的 ID 和摘要。

## 10. 状态模型

目标 `AgentState`：

```python
class AgentState(TypedDict):
    run_id: str
    request_id: str
    user_id: int
    session_id: str
    branch_id: int | None
    kb_id: int | None
    messages: list
    current_query: str
    memory_context: dict
    working_memory: dict
    available_tools: list[str]
    tool_calls: list[dict]
    observations: list[dict]
    evidence_ledger: dict
    decision_summary: dict | None
    iteration: int
    budgets: dict
    final_answer: str | None
    citations: list[dict]
    verification: dict | None
    memory_update_plan: dict | None
    events: list[dict]
    error: str | None
```

`retrieved_docs/reranked_docs/sub_query_plans/need_retrieval/need_web_search` 等旧字段逐步收敛到 Tool Call、Observation、Evidence Ledger 和 Working Memory，避免多个布尔状态互相矛盾。

## 11. 验证和记忆提交

Agent 输出 final answer 后仍需服务端验证：

- 知识库/联网事实是否有可用 `E#`；
- 引用 ID 是否属于当前 Run 的 Ledger；
- 必答槽位是否覆盖；
- 是否存在未披露冲突；
- 是否应该拒答而没有拒答；
- 输出 schema、敏感信息和权限是否合规。

验证失败时，若仍有预算，则把结构化失败 Observation 返回 Agent 再执行一轮；否则拒答或输出带缺口说明的部分答案。只有最终答案保存成功后，才应用 session summary 和长期记忆更新计划。

## 12. 本次重构的非目标

- 不保存或展示模型原始 Chain-of-Thought。
- 不开放任意 Shell、数据库写入、文件写入或 MCP 工具。
- 不做长任务、定时任务、后台计划恢复。
- 不做多 Agent 分工和 Agent 间消息协议。
- 不用 ReAct 替代检索器、Reranker、Evidence Judge 和服务端验证。
- 不让模型直接决定权限、记忆持久化或最终引用合法性。

## 13. 最终决策

知识库检索和联网搜索适合改造成 ReAct 的 Action，但工具结果必须先规范化为 Observation 和 Evidence，再交给 Agent 决定下一步。ReAct 的“思考”在工程上表现为结构化决策、查询策略和未解决槽位，而不是存储/展示原始思维链。

目标框架是：

```text
LangGraph 持久化状态机
  + Native Tool Calling
  + 受控 ReAct 循环
  + Tool Gateway
  + Evidence Ledger
  + 三层记忆
  + Typed Context Builder
  + Verification / Refusal
  + Trace / Budget / Policy
```

下一步代码重构必须以本文件、工具协议和迁移验收文档为基线，不再继续扩展旧的固定路由节点。

## 14. 开源方案取舍结论

本方案参考 LangGraph 的显式 State/Node/Tool 执行方式、Letta 的上下文内 memory blocks、Mem0 的记忆 CRUD 与选择机制、Graphiti 的时间有效事实模型。采用的是可复用的设计原则，不会为了对齐某个框架而引入不需要的基础设施：

- 使用 LangGraph 自定义图，不套用无法满足 Evidence/Memory/Policy 约束的黑盒 AgentExecutor。
- 使用 MySQL + 当前向量能力实现长期记忆，V1.6 不强制引入图数据库。
- 使用 native tool calling 传递 Action，不解析 `Thought:` 文本。
- 使用结构化决策摘要替代原始 Chain-of-Thought 的存储和展示。
- 先实现单 Agent + 两个只读工具，MCP、Skill 和多 Agent 留在后续版本。

当前环境已验证 `ChatDeepSeek.bind_tools()` 和 `langgraph.prebuilt.ToolNode` 可用；依赖版本和实现路径见 [ReAct 运行时与工具协议](<ReAct运行时与工具协议.md>)。
