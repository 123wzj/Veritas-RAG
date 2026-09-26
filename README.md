# Veritas RAG

Veritas RAG 是一个面向个人与私有知识库的问答和记忆系统。当前在线 RAG 已统一使用受控 ReAct Runtime，并实现知识库与联网工具、Evidence Ledger、三层记忆、动态上下文和运行追踪。旧固定 Graph 不再参与 API 请求，仅暂存于仓库供历史评测迁移参考。

## 当前架构

```text
FastAPI / SSE
  → ReAct Runtime
      → hydrate → decide → act → observe → decide → verify
  → Tool Gateway
      ├─ knowledge_search
      └─ web_search
  → Evidence Ledger / Verification
  → Session + Long-term Memory
```

关键边界：

- 所有在线问答均进入 `backend.agent.graph.react_graph`；旧配置值也会被归一化为 `react`。
- 当前 Trace 保存 Run、Span、Tool Call 和 Observation；尚未保存每个节点后的完整 State，因此不能从失败节点续跑。
- 旧 `backend/graph/` 不再被在线 Runtime 导入，待历史评测脚本迁移后可删除。
- 上传 API 当前只接受 Markdown（`.md`）；其他格式属于后续规划。
- 后端模型角色统一使用 `deepseek-v4-flash`。

完整架构与准确边界见 [文档中心](docs/README.md)。

## 核心能力

### 文档入库

```text
Markdown
  → 结构解析
  → Parent Chunk
  → Child Chunk
  → Dense 向量 + BM25 语料
  → MySQL + Chroma
```

Child 用于召回，Parent 用于补充生成上下文。MySQL 保存原文、层级关系和业务元数据，Chroma 保存 Child Dense 向量。

### 检索工具

`knowledge_search` 复用现有 Dense + BM25 + Weighted RRF + Reranker + 多样性控制链路；`web_search` 复用现有联网 Provider。所有工具必须经过 Registry、Policy 和 Gateway，模型不能伪造 User、Session、Branch 或 KB 范围。

### ReAct 与证据

Controller 输出结构化 `AgentDecision`，只能选择调用工具、最终回答、澄清或拒答。工具结果被规范化为 `ToolObservation`，进入 Evidence Ledger 并分配 `E#`。最终答案只能引用实际存在的 Evidence，Verification 不通过时继续检索或拒答。

### 三层记忆

- 工作记忆：当前 Run 的目标、回答槽位、约束、动作和未解决项；
- 短期记忆：Session Summary 与近期消息，支持会话和分支隔离；
- 长期记忆：跨会话的用户/项目事实，具备时效、确认、冲突、替换和删除 Tombstone。

每一轮 `decide` 都重新构造上下文，而不是无限追加历史字符串。不同 Profile 支持 16k、32k、96k、192k 和 272k 输入预算。

## 快速开始

后端使用 Conda 环境 `cook-rag-1`：

```powershell
conda activate cook-rag-1
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

前端：

```powershell
Set-Location frontend
npm install
npm run dev
```

环境变量示例位于 `backend/.env.example`。不要提交真实 `.env`。

## 验证

```powershell
conda run -n cook-rag-1 pytest -q
Set-Location frontend
npm run build
```

## 文档导航

- [项目现状与阅读指南](docs/01-项目现状与阅读指南.md)
- [系统架构与运行流程](docs/02-系统架构与运行流程.md)
- [状态、追踪与失败恢复](docs/03-状态、追踪与失败恢复.md)
- [工具系统与检索](docs/04-工具系统与检索.md)
- [上下文工程与记忆机制](docs/05-上下文工程与记忆机制.md)
- [数据库迁移与 ReAct 上线验收](docs/06-数据库迁移与ReAct上线验收.md)

## 当前优先级

1. 在目标 MySQL 执行 ReAct Runtime 与默认值迁移并验证；
2. 为新 ReAct 主链建立持续质量、延迟和成本评测；
3. 完善生产认证、权限、限流和日志脱敏；
4. 将历史评测脚本迁移到 `run_rag_runtime`，随后删除旧 Graph；
5. 实现节点 Checkpoint 与 Resume，再扩展有副作用工具、MCP 和多 Agent。

许可证：MIT。
