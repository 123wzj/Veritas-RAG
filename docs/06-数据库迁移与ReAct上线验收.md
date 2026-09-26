# 数据库迁移与 ReAct 上线验收

> 当前在线 RAG 已统一切换到受控 ReAct。本文记录数据库落地、上线检查和旧 Graph 清理条件。最后核对：2026-09-26。

## 1. 当前状态

- `backend/agent/runtime.py` 是唯一在线 Runtime；
- 所有请求都执行 `backend.agent.graph.react_graph`；
- `get_runtime_mode()` 固定返回 `react`，旧值或非法值不会使 API 回到旧图；
- `backend/graph/` 不再被在线 Runtime 导入；
- 新工具层、Evidence Ledger、Context Builder 和三层记忆均位于 ReAct 主链；
- LangGraph 已接入 SQLite 持久化 Checkpointer，可按同一 `request_id` 从最近安全节点恢复；
- 目标 MySQL 仍需执行正式迁移；
- 生产认证、限流、可观测性和恢复能力仍需继续完善。

## 2. 数据库迁移

正式脚本：

```text
backend/db/mysql/migrations/20260906_v15_acceptance.sql
backend/db/mysql/migrations/20260926_react_runtime.sql
backend/db/mysql/migrations/20260926_react_default.sql
backend/db/mysql/migrations/20260926_independent_branches_and_checkpoints.sql
```

迁移职责：

- `20260906_v15_acceptance.sql`：V1.5 记忆、会话和 Trace 基础字段；
- `20260926_react_runtime.sql`：ReAct Run 字段、长期记忆治理、Tool Call 与 Observation 表；
- `20260926_react_default.sql`：将 `rag_runs.runtime_mode` 的数据库默认值改为 `react`。
- `20260926_independent_branches_and_checkpoints.sql`：分支改为独立 Session 血缘，删除 Branch Summary 字段并增加唯一索引。Checkpoint 本体位于独立 SQLite 文件，不写入 MySQL。

历史 Run 的 `runtime_mode` 保持原值，以保证审计准确性；只修改新记录的默认值。

不能只依赖 SQLAlchemy `create_all` 或开发期 `_sync_legacy_schema` 代替生产迁移。建议：

1. 备份目标数据库；
2. 在数据库副本执行全部迁移；
3. 校验字段、索引和唯一约束；
4. 验证 `rag_runs.runtime_mode` 默认值为 `react`；
5. 使用测试用户执行知识库、Web、无答案和多轮对话冒烟测试；
6. 核对 Run、Span、Tool Call、Observation 和记忆审计。

## 3. Runtime 配置

```text
AGENT_RUNTIME_MODE=react
AGENT_REACT_MAX_ITERATIONS
AGENT_MAX_TOOL_CALLS
AGENT_MAX_KB_CALLS
AGENT_MAX_WEB_CALLS
AGENT_RUN_DEADLINE_MS
AGENT_CONTEXT_PROFILE
AGENT_CONTEXT_INPUT_TOKEN_LIMIT
AGENT_OUTPUT_TOKEN_RESERVE
AGENT_CHECKPOINT_PATH=data/langgraph/checkpoints.sqlite3
MEMORY_LLM_SELECTION_ENABLED=false
```

`AGENT_RUNTIME_MODE` 目前是兼容性配置。在线代码不会根据它分流，旧值会被归一化为 `react`。

所有 LLM 角色统一使用 `deepseek-v4-flash`。不要在 `.env` 中用 `DEEPSEEK_PRO_MODEL` 制造隐式模型分工。

## 4. 上线验收

### 4.1 功能

- Markdown 入库、父子分块和增量索引正常；
- `knowledge_search` 与 `web_search` 只能通过 Tool Gateway 执行；
- Evidence Ledger、引用和拒答正确；
- 长对话、普通 Session 和独立分叉 Session 隔离正确；
- 长期记忆确认、冲突、过期和 Tombstone 正确；
- Memory Update 只在答案完成验证后提交。

### 4.2 安全与隔离

- 模型不能覆盖 User、Session、Branch、KB 身份参数；
- KB ACL 拒绝未授权检索；
- Web 未启用时拒绝联网调用；
- Trace API 只能读取当前用户数据；
- Prompt 注入不能改变 System Kernel 和工具策略；
- 日志、Trace 和 Tool 参数执行敏感字段脱敏。

### 4.3 质量与性能

| 类别 | 指标示例 |
| --- | --- |
| 检索 | Hit@K、Recall@K、MRR、NDCG、噪声率 |
| 生成 | Faithfulness、Answer Relevance、完整性 |
| 引用 | Citation Precision、Coverage、无效 ID 率 |
| 拒答 | 无答案拒答率、可答问题误拒率 |
| Agent | 平均迭代数、工具成功/空结果/重试率、无进展停止率 |
| 记忆 | 选择准确率、错误写入率、冲突发现率、撤销率 |
| 隔离 | 跨 User/Session/独立分叉/KB 泄漏必须为 0 |
| 性能 | P50/P95 延迟、Token、单位请求成本 |
| 稳定性 | Run 成功率、Provider 错误率、Deadline 超时率 |

## 5. 回滚策略

旧 Graph 不再是可通过环境变量自动启用的在线模式。若 ReAct 出现严重故障，应优先：

1. 关闭受影响的 Web 或外部 Provider；
2. 降低迭代、Tool 和上下文预算；
3. 回滚到上一稳定提交或发布制品；
4. 保留失败 Run 与 Tool 审计用于复盘。

如确有必要恢复旧 Graph，应通过独立代码回滚和完整回归测试完成，不能仅修改 `AGENT_RUNTIME_MODE`。

## 6. 旧 Graph 清理

当前仍有部分离线评测脚本直接导入 `backend.graph.graph.run_agentic_rag`。清理顺序：

1. 将评测脚本统一迁移到 `backend.agent.runtime.run_rag_runtime`；
2. 补齐 ReAct 专用评测数据提取；
3. 确认仓库中没有生产或测试入口导入旧 Graph；
4. 冻结需要保留的历史指标和报告；
5. 单独提交删除 `backend/graph/`。

不要在迁移评测脚本之前直接删除旧目录。

## 7. 发布前检查单

- [ ] 三个迁移脚本已在目标数据库执行并核验；
- [ ] 根 `.env` 使用 `AGENT_RUNTIME_MODE=react`；
- [ ] 在线 Runtime 不导入 `backend.graph`；
- [ ] 即使传入 `legacy` 或 `react_shadow`，测试仍进入 ReAct；
- [ ] Session、Branch、KB ACL 自动化测试通过；
- [ ] 相同 request_id 可恢复未完成 Checkpoint，已完成 Run 不重复执行；
- [ ] 前端可在可恢复失败消息上复用原 request_id 继续执行；
- [ ] 生产多实例部署已将 SQLite Checkpointer 迁移到 Postgres/MySQL Saver；
- [ ] Tool Call 幂等唯一约束生效；
- [ ] Trace API 只能读取当前用户数据；
- [ ] SSE 事件与前端兼容；
- [ ] 评测报告已归档到 `docs/评测与优化/`；
- [ ] 生产密钥、CORS、认证、限流和日志脱敏已配置。

## 8. 与 Checkpoint 的关系

当前已实现单实例 SQLite Checkpointer。恢复会重跑失败节点本身，但已持久化 Tool Observation 可复用。未来加入写操作、MCP、多 Agent 或长任务前，还必须补齐非幂等工具保护、人工确认、补偿动作和多实例 Checkpoint 后端。详细边界见《03-状态、追踪与失败恢复》。
