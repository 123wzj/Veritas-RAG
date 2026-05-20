# Agentic RAG 开发文档

## 1. 文档目标

本文档用于指导一个基于 `LangChain + LangGraph + Chroma` 的 `Agentic RAG` 问答系统设计与开发。

系统目标：

- 支持中英混合文档问答
- 支持文档与图片等多模态数据
- 支持高质量模式，单次问答延迟可接受 `8~15 秒`
- 支持自我纠错、反思、多步推理
- 支持用户显式开启“联网增强”
- 支持统一流式输出
- 支持多用户逻辑隔离
- 支持用户记忆模块
- 支持可观测、可评测、可扩展

---

## 2. 已确认的约束与设计决策

### 2.1 业务与部署边界

- 部署模式：`单租户`
- 用户隔离：`逻辑隔离`
- 数据隔离粒度：`user_id / kb_id / acl_tags`

### 2.2 数据类型

- 语言：`中英混合`
- 主要数据类型：`文档`
- 兼容数据类型：`图片、扫描 PDF、图文混排文档、表格`

### 2.3 交互模式

- 响应模式：`高质量优先`
- 目标时延：`8~15 秒`
- 联网模式：`用户显式开启“联网增强”`

### 2.4 模型策略

- 优先使用 `API 模型`
- 模型层采用可插拔设计，不与某一家厂商强耦合

---

## 3. 系统设计目标

## 3.1 功能目标

系统需要具备以下能力：

1. 文档与图片统一入库
2. 多模态检索
3. Dense + Sparse 混合检索
4. 检索后重排
5. 多步推理与问题拆解
6. 检索失败时的反思与自我纠错
7. 联网补充证据
8. 引用溯源
9. 用户记忆注入
10. 统一流式返回

## 3.2 非功能目标

1. 查询流程可观测
2. 结果可复现
3. 用户之间逻辑隔离
4. 数据结构可扩展
5. 后续支持多知识库、多模型、多工具扩展

---

## 4. 推荐总体架构

系统拆分为 7 层：

1. 接入层
2. 编排层
3. 文档处理层
4. 检索层
5. 推理生成层
6. 记忆层
7. 治理与运维层

### 4.1 接入层

职责：

- 提供 Web / App / OpenAPI 接口
- 负责认证、会话、用户上下文
- 提供 SSE 或 WebSocket 流式响应

建议接口：

- `POST /v1/rag/query`
- `POST /v1/rag/query/stream`
- `POST /v1/knowledge/upload`
- `POST /v1/knowledge/reindex`
- `GET /v1/session/:session_id/history`

### 4.2 编排层

技术选型：

- `LangGraph` 负责主流程状态机
- `LangChain` 负责模型、retriever、prompt、tool 封装

职责：

- 管理 Agentic RAG 状态
- 控制反思、自纠错、多步推理
- 控制联网工具和检索工具
- 统一事件流输出

### 4.3 文档处理层

职责：

- 文档解析
- OCR / 图片理解
- 分块
- 元数据提取
- 向量化
- 入 Chroma

### 4.4 检索层

技术选型：

- Chroma

职责：

- Dense 检索
- Sparse 检索
- Hybrid Search
- 元数据过滤
- 逻辑隔离过滤
- 多模态检索

### 4.5 推理生成层

职责：

- 多步问题拆解
- 检索结果重排
- 答案生成
- 证据验证
- 自我反思

### 4.6 记忆层

职责：

- 用户画像
- 偏好记忆
- 会话摘要
- 长期主题记忆

### 4.7 治理与运维层

职责：

- 日志
- tracing
- 指标监控
- prompt 版本管理
- 评测
- A/B 实验
- 成本监控

---

## 5. 技术选型建议

## 5.1 后端框架

- Python 3.11+
- FastAPI
- LangChain
- LangGraph
- Pydantic
- SQLAlchemy

## 5.2 数据存储

- 向量数据库：`Chroma`
- 关系型数据库：`MySQL`
- 缓存：`Redis`
- 对象存储：`S3 / MinIO`

## 5.3 模型层

采用统一 Model Gateway 设计，支持：

- Chat LLM
- Embedding 模型
- OCR / Vision 模型
- Reranker 模型

建议通过统一配置切换不同 API 厂商，而不是在业务代码里写死。

---

## 6. 核心数据流

整个系统分为两条主链路：

1. 离线知识入库链路
2. 在线问答链路

---

## 7. 离线知识入库链路设计

## 7.1 数据源范围

支持：

- PDF
- DOCX
- PPTX
- Markdown
- HTML
- TXT
- 图片
- 扫描文档

## 7.2 入库流程

1. 文件上传
2. 文件类型识别
3. 文档解析
4. OCR 与图片描述生成
5. 文档结构抽取
6. 分块
7. 元数据打标
8. 向量化
9. Milvus 入库
10. 原始文件和结构化内容写对象存储

## 7.3 文档解析策略

### 纯文本类文档

- 按标题层级、段落、列表、表格做结构化解析

### 扫描 PDF / 图片

- OCR 提取文本
- Vision 模型生成图片摘要或 caption
- 将图片说明与邻近文本绑定

### 表格

- 表格优先结构化提取
- 生成表格文字描述
- 必要时保留原始表格索引片段

---

## 8. 分块策略设计

## 8.1 总体原则

分块目标不是“切得均匀”，而是“检索时既能找到，又不丢上下文”。

推荐三层结构：

1. Document
2. Parent Chunk
3. Child Chunk

## 8.2 Parent Chunk

用途：

- 作为生成上下文的主承载块

建议大小：

- `800 ~ 1500 tokens`

切分规则：

- 按标题
- 按段落
- 按语义边界
- 保留章节路径

## 8.3 Child Chunk

用途：

- 作为检索召回主单位

建议大小：

- `200 ~ 400 tokens`

重叠：

- `50 ~ 80 tokens`

切分规则：

- 从 parent chunk 再细切
- 每个 child 记录 parent_id

## 8.4 特殊内容处理

### 图片

- 图片单独作为一类 chunk
- 绑定：
  - 图片 caption
  - 所在页码
  - 所在章节
  - 邻近文本

### 表格

- 表格按完整语义块处理
- 不建议简单按字符硬切

### FAQ / 短问答文档

- 一问一答作为一个 parent
- 再根据长度决定是否拆 child

---

## 9. 向量化策略设计

## 9.1 总体原则

向量化层采用“多路嵌入 + 可插拔模型”设计：

1. 文本 dense embedding
2. 文本 sparse 表示
3. 图片 embedding

## 9.2 文本 embedding 选择标准

考虑因素：

- 中英混合效果
- 领域迁移能力
- 长文本鲁棒性
- 延迟
- 成本
- 维度大小

建议：

- 主文本 embedding 使用一个高质量 API embedding 模型
- 后续如果成本压力大，再加本地 embedding 备选

## 9.3 Sparse 检索设计

建议：

- 优先使用 Milvus 支持的 `BM25 / sparse` 方案

原因：

- 中英混合文档中，关键词精确匹配依然非常重要
- Dense 解决语义召回，Sparse 解决精确词命中

## 9.4 图片向量化

建议：

- 图片单独生成 image embedding
- 若需要图文统一检索，则引入跨模态 embedding

---

## 10. Milvus Schema 设计

建议单 collection 方案，逻辑隔离用 filter 控制。

### 10.1 核心字段

- `pk`
- `user_id`
- `kb_id`
- `doc_id`
- `parent_id`
- `chunk_id`
- `modality`
- `lang`
- `title`
- `section_path`
- `page_no`
- `content`
- `caption`
- `source_uri`
- `version`
- `acl_tags`
- `dense_vector`
- `sparse_vector`
- `image_vector`
- `created_at`

### 10.2 检索过滤条件

默认所有查询都带：

- `user_id`
- `kb_id`

可选：

- `acl_tags`
- `doc_id`
- `lang`
- `modality`

### 10.3 逻辑隔离设计

因为是单租户，所以不做租户物理分库。

隔离方式：

- 用户上传知识库按 `user_id + kb_id` 过滤
- 用户会话记忆按 `user_id` 过滤
- 管理员知识库可通过 `acl_tags` 授权

---

## 11. 混合检索设计

## 11.1 推荐检索链路

1. 问题改写
2. Dense 检索
3. Sparse 检索
4. Hybrid 融合
5. Metadata 过滤
6. Rerank
7. Parent 回补
8. 上下文打包

## 11.2 Dense + Sparse 融合方案

推荐默认：

- Dense topK：`40`
- Sparse topK：`40`
- 融合方式：`RRF`
- 融合后候选：`20`

这样设计的优点：

- Dense 负责语义召回
- Sparse 负责关键词精确匹配
- RRF 工程实现简单，鲁棒性较好

## 11.3 Parent 回补策略

召回单位是 child chunk，但生成时优先补回 parent chunk：

- 避免上下文过碎
- 保证答案有完整段落支撑

## 11.4 多模态检索策略

当问题中涉及图片、图表、示意图时：

- 触发文本检索 + 图片检索并行
- 再做结果融合

---

## 12. 重排机制设计

## 12.1 推荐两级重排

### 第一层：轻量重排

- 目标：快速筛掉噪音
- 方案：cross-encoder reranker 或轻量 rerank API

### 第二层：高价值问题增强重排

触发条件：

- 多跳问题
- 复杂问答
- 用户开启联网增强
- 首轮证据冲突

方案：

- 强 reranker
- 或 LLM-based rerank

## 12.2 进入重排的候选数

建议：

- 融合后 `20`
- 重排后保留 `6 ~ 8`

---

## 13. Agentic RAG 主流程设计

技术选型：

- 用 `LangGraph` 定义状态机

## 13.1 Graph State 设计

建议状态字段：

- `request_id`
- `session_id`
- `user_id`
- `kb_id`
- `query`
- `query_rewritten`
- `sub_questions`
- `web_enabled`
- `retrieved_docs`
- `reranked_docs`
- `selected_evidence`
- `reasoning_trace_summary`
- `reflection_notes`
- `draft_answer`
- `final_answer`
- `citations`
- `confidence`
- `memory_context`
- `events`
- `error`

## 13.2 节点设计

### 1. `route_request`

职责：

- 判断是否普通问答、多跳问答、多模态问答、是否联网增强

### 2. `load_user_memory`

职责：

- 加载用户画像、偏好、近期会话摘要

### 3. `rewrite_query`

职责：

- 问题标准化
- 中英混合 query 改写
- 补充检索关键词

### 4. `decompose_query`

职责：

- 对复杂问题拆成子问题

### 5. `retrieve_hybrid`

职责：

- 调用 Milvus 混合检索

### 6. `rerank_candidates`

职责：

- 对候选文档做相关性重排

### 7. `pack_evidence`

职责：

- 回补 parent
- 去重
- 合并同文档相邻 chunk

### 8. `judge_evidence`

职责：

- 判断证据是否足够
- 判断是否有冲突

### 9. `reflection`

职责：

- 检查是否需要修正检索词
- 检查是否需要增加一跳推理
- 检查是否需要联网增强

### 10. `web_search`

触发条件：

- 用户显式开启联网增强
- 且内部证据不足或过旧

职责：

- 联网获取补充证据
- 外网结果单独标注来源

### 11. `generate_answer`

职责：

- 基于最终证据生成答案
- 强制引用溯源

### 12. `verify_answer`

职责：

- 校验答案与证据一致性
- 检查是否遗漏关键子问题

### 13. `write_memory`

职责：

- 把用户偏好、任务结果摘要写回记忆模块

### 14. `stream_finalize`

职责：

- 统一输出最终事件

---

## 13.3 LangGraph 流程控制机制详解

### 13.3.1 LangGraph 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph 组成部分                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │    State     │───→│     Node     │───→│     Edge     │          │
│  │   (状态)     │    │    (节点)    │    │     (边)     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│        ▲                    ▲                                       │
│        │                    │                                       │
│   全局共享数据          处理逻辑函数                                │
│   TypedDict             接收 state, 返回更新                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**LangGraph 的核心思想**：通过状态机（State Machine）来编排复杂的 AI 应用流程，而不是传统的线性代码。

#### State（状态）- 全局数据总线

State 是节点间传递的唯一数据载体，是一个 TypedDict：

```python
class RAGState(TypedDict):
    """状态是节点间传递的唯一数据载体"""
    # 基础信息
    request_id: str
    user_id: int
    query: str
    kb_id: Optional[int]

    # 检索结果 - 由不同节点写入
    retrieved_docs: List[Dict[str, Any]]    # 检索节点写入
    reranked_docs: List[Dict[str, Any]]     # 重排节点写入
    selected_evidence: List[Dict[str, Any]]  # 打包节点写入

    # 最终输出
    final_answer: Optional[str]              # 生成节点写入
    confidence: float                        # 验证节点写入

    # 控制标志 - 决定路由
    evidence_sufficient: bool                # 决定下一步路由
    need_reflection: bool                    # 决定是否反思
    need_web_search: bool                    # 决定是否联网

    # 累加字段
    step_count: Annotated[int, operator.add]  # 每次访问自动 +1
    events: List[Dict[str, Any]]             # 事件流
```

**关键特性**：
- 节点之间**不直接调用**，通过 State 传递数据
- 每个节点返回 `Dict[str, Any]`，自动合并到 State
- 使用 `Annotated[int, operator.add]` 实现累加字段

#### Node（节点）- 状态转换函数

每个节点是一个异步函数，接收完整的 State，返回需要更新的字段：

```python
async def retrieve_hybrid(state: RAGState) -> Dict[str, Any]:
    """
    节点函数签名：
    - 输入：完整的 RAGState
    - 输出：需要更新的字段（返回 Dict）
    """
    # 1. 从 State 读取数据
    query = state.get("query_rewritten") or state.get("query", "")
    user_id = state.get("user_id", 0)
    kb_id = state.get("kb_id")

    # 2. 执行业务逻辑
    results = await hybrid_retriever.retrieve_async(
        query=query,
        user_id=user_id,
        kb_id=kb_id,
    )

    # 3. 返回状态更新（只返回需要修改的字段）
    return {
        "retrieved_docs": results,           # 新增字段
        "step_count": state.get("step_count", 0) + 1,  # 递增计数
        "events": state.get("events", []) + [  # 追加事件
            {"event": "retrieval.completed", "data": {"count": len(results)}}
        ],
    }
```

**节点执行模式**（LangGraph 内部逻辑）：
```python
# LangGraph 内部执行逻辑（伪代码）
current_state = initial_state
for node in node_sequence:
    # 调用节点函数
    updates = await node(current_state)
    # 合并更新到状态
    current_state = {**current_state, **updates}
```

#### Edge（边）- 节点连接关系

**普通边**：无条件执行
```python
# A → B，无条件执行
workflow.add_edge("retrieve", "rerank")
```

**条件边**（核心机制）：根据状态动态决定下一个节点
```python
# 根据状态动态决定下一个节点
workflow.add_conditional_edges(
    "judge_evidence",              # 源节点
    route_evidence,                # 路由函数
    {
        "generate": "generate_answer",   # 返回 "generate" → 去 generate_answer
        "reflect": "reflect",            # 返回 "reflect" → 去 reflect
        "web_search": "web_search",      # 返回 "web_search" → 去 web_search
    }
)
```

**路由函数实现**：
```python
async def route_evidence(state: RAGState) -> str:
    """
    路由函数：读取状态，返回目标节点名称
    """
    evidence_sufficient = state.get("evidence_sufficient", False)
    need_reflection = state.get("need_reflection", False)
    need_web_search = state.get("need_web_search", False)

    if evidence_sufficient:
        return "generate"         # 证据足够 → 生成答案
    if need_reflection:
        return "reflect"          # 需要反思 → 重新检索
    if need_web_search:
        return "web_search"       # 允许联网 → 联网搜索
    return "generate"             # 默认 → 生成答案
```

### 13.3.2 完整 RAG 流程图

```
                    RAGState (初始状态)
                    ─────────────────
                    query = "Python 如何装饰器?"
                    user_id = 1
                    kb_id = 123
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        load_memory                              │
    │  从数据库读取用户偏好、历史会话                                   │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: memory_context = {...}                                   │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       rewrite_query                             │
    │  用 LLM 改写查询，使其更清晰、易检索                              │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: query_rewritten = "Python 装饰器的语法和用法"            │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      decompose_query                            │
    │  拆解复杂问题为多个子问题                                         │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: sub_questions = ["装饰器语法", "装饰器用法", "装饰器示例"] │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       │     条件边 (need_retrieval)  │
                       └──────────────┬──────────────┘
                        True │                │ False
                             ▼                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        retrieve                                 │
    │  执行混合检索 (Dense + Sparse + RRF + Parent 回补)              │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: retrieved_docs = [doc1, doc2, ...]                       │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         rerank                                  │
    │  用 Reranker API 精排                                            │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: reranked_docs = [doc1, doc2, ...] (排序后)              │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       pack_evidence                             │
    │  打包成可引用的证据结构，带 evidence_id                           │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: selected_evidence = [{evidence_id: "E1", ...}, ...]     │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      judge_evidence                              │
    │  用 LLM 评估证据质量、覆盖度、可回答性                            │
    │  ────────────────────────────────────────────────────────────  │
    │  更新: evidence_sufficient, need_reflection, need_web_search    │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │ 条件边 (route_evidence)                        │
              └───────────────────────┼───────────────────────┘
                evidence_sufficient=T   │   need_reflection=T   │  need_web_search=T
                        ▼               ▼               ▼
        ┌───────────────────┐   ┌──────────────┐   ┌──────────────┐
        │  generate_answer  │   │   reflect    │   │  web_search  │
        │  生成带引用的答案  │   │ 反思后重检索  │   │  联网搜索    │
        └─────────┬─────────┘   └──────┬───────┘   └──────┬───────┘
                  │                     │                   │
                  ▼                     ▼                   │
        ┌───────────────────┐           │                   │
        │  verify_answer    │           │                   │
        │  验证答案质量      │           │                   │
        └─────────┬─────────┘           │                   │
                  │                     │                   │
        ┌─────────┴─────────┐           │                   │
        │ 条件边 (route_verification)    │                   │
        └─────────┬─────────┘           │                   │
          need_reflect=T │ F             │                   │
                ▼        │              │                   │
          ┌──────┐       │              │                   │
          │reflect│       │              │                   │
          └──┬───┘       │              │                   │
             │           │              │                   │
             └───────────┴──────────────┴───────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   write_memory   │
                    │   保存对话记忆    │
                    └────────┬─────────┘
                             │
                             ▼
                           END
```

### 13.3.3 关键控制机制

#### 1. 状态累加器（Accumulator）

```python
class RAGState(TypedDict):
    step_count: Annotated[int, operator.add]  # 每次访问自动 +1
```

使用场景：
```python
# 节点 A
return {"step_count": 1}

# 节点 B
return {"step_count": 1}

# 合并后状态
# step_count = 2  (自动累加)
```

#### 2. 反思循环控制

```python
async def route_reflection(state: RAGState) -> str:
    """
    防止无限循环的关键
    """
    need_reflection = state.get("need_reflection", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 2)  # 最多 2 次

    if need_reflection and reflection_count <= max_reflections:
        return "reflect"    # 继续反思
    return "proceed"        # 停止反思，继续生成
```

反思流程：
```
第 1 次: reflection_count = 0 → 反思 → reflection_count = 1
第 2 次: reflection_count = 1 → 反思 → reflection_count = 2
第 3 次: reflection_count = 2 → 超过 max_reflections → proceed
```

#### 3. 事件流输出

```python
async def run_agentic_rag(...):
    initial_state = create_initial_state(...)

    if stream_events:
        # 流式输出：每个节点执行完就发送事件
        async for event in agentic_rag_graph.astream(initial_state):
            # event 格式: {'node_name': {state_updates}}
            yield event
```

事件流示例：
```
{'load_memory': {'memory_context': {...}}}
{'rewrite_query': {'query_rewritten': 'Python 装饰器语法'}}
{'retrieve': {'retrieved_docs': [...], 'events': [...]}}
{'rerank': {'reranked_docs': [...]}}
...
```

### 13.3.4 智能决策示例

#### 场景 1：证据不足 → 反思重检

```
用户问题: "Golang 和 Python 性能对比"

1. judge_evidence:
   - coverage_score = 0.4 (信息不全)
   - evidence_sufficient = False
   - need_reflection = True

2. route_evidence → return "reflect"

3. reflection 节点:
   - 分析缺失信息
   - 生成新查询: ["Golang 性能基准", "Python 性能基准", "Go vs Python benchmark"]

4. 返回 retrieve → 用新查询重检
```

#### 场景 2：答案验证失败 → 重新生成

```
1. generate_answer: 生成答案

2. verify_answer:
   - grounded = False (答案不被证据支撑)
   - need_reflection = True

3. route_verification → return "reflect"

4. reflection: 分析问题，改进检索策略
```

### 13.3.5 LangGraph 控制流程总结

LangGraph 控制流程的核心在于：

| 机制 | 作用 | 代码体现 |
|------|------|----------|
| **State** | 全局数据总线，节点间通信 | `RAGState(TypedDict)` |
| **Node** | 状态转换，业务逻辑执行 | `async def node(state) -> Dict` |
| **Conditional Edge** | 动态路由，智能决策 | `add_conditional_edges(source, router, mapping)` |
| **Router Function** | 读取状态，返回目标节点 | `def router(state) -> str: return "next_node"` |
| **Accumulator** | 累加字段，防止无限循环 | `Annotated[int, operator.add]` |
| **Stream** | 实时事件输出 | `async for event in graph.astream(state)` |

这种设计让 RAG 系统变成了**自适应的决策系统**，而不是固定的流水线。

---

## 14. 自我纠错与反思机制设计

## 14.1 设计原则

反思不能做成无上限自由循环，必须做成受控节点。

建议：

- 最大反思轮数：`2`
- 最大工具步数：`8`
- 超过阈值直接输出保守答案

## 14.2 反思触发条件

1. 检索结果为空
2. 检索结果相关性低
3. 检索结果互相冲突
4. 多步问题只覆盖部分子问题
5. 生成结果缺少证据支撑

## 14.3 反思动作

1. 改写 query
2. 放宽或调整 filter
3. 切换检索模式
4. 增加子问题
5. 触发联网增强
6. 重新组织证据顺序

## 14.4 自我纠错策略

建议采用“先检索纠错，再生成纠错”的双层策略：

- 第一层：检索纠错
- 第二层：答案纠错

这样成本可控，且更容易定位问题。

---

## 15. Prompt 设计

## 15.1 Prompt 结构

建议统一采用以下拼接结构：

1. `system role`
2. `user memory context`
3. `task plan / sub-questions`
4. `retrieved evidence`
5. `web evidence`
6. `output contract`

## 15.2 System Prompt 要求

必须明确：

- 只能基于证据回答
- 证据不足时允许保守表达
- 内部知识和联网知识要分开引用
- 不得编造来源
- 输出必须遵守统一格式

## 15.3 证据拼接规则

每条证据建议统一格式：

- `evidence_id`
- `source_type`
- `doc_title`
- `page_no`
- `section_path`
- `snippet`
- `score`

---

## 16. 统一流式输出格式

## 16.1 输出协议建议

推荐使用 SSE。

事件类型建议统一为：

- `run.started`
- `memory.loaded`
- `query.rewritten`
- `query.decomposed`
- `retrieval.started`
- `retrieval.completed`
- `rerank.completed`
- `reflection.started`
- `reflection.completed`
- `websearch.started`
- `websearch.completed`
- `answer.delta`
- `citation.delta`
- `answer.completed`
- `memory.updated`
- `run.failed`

## 16.2 统一事件结构

```json
{
  "event": "answer.delta",
  "request_id": "req_xxx",
  "session_id": "sess_xxx",
  "timestamp": 1730000000,
  "data": {
    "text": "本系统采用混合检索..."
  }
}
```

## 16.3 最终答案对象

建议最终完整输出：

```json
{
  "answer": "最终答案文本",
  "citations": [
    {
      "source_type": "knowledge_base",
      "doc_id": "doc_1",
      "title": "RAG设计文档",
      "page_no": 12,
      "snippet": "......"
    }
  ],
  "web_citations": [],
  "confidence": 0.82,
  "reasoning_summary": "系统先进行了问题拆解，再做混合检索与重排。",
  "used_web_enhancement": false
}
```

---

## 17. 用户记忆模块设计

## 17.1 记忆分类

### 1. Profile Memory

保存：

- 用户偏好语言
- 输出偏好
- 常问主题
- 角色信息

### 2. Session Memory

保存：

- 最近会话摘要
- 上下文任务状态
- 未完成问题

### 3. Long-term Semantic Memory

保存：

- 用户长期关注主题
- 高价值历史问答摘要

## 17.2 不建议做的事情

- 不要把所有对话原文都塞进 prompt
- 不要把用户记忆和知识库事实混在一个 Milvus collection 里

## 17.3 推荐存储

- Profile Memory：`PostgreSQL`
- Session Summary：`PostgreSQL + Redis`
- Long-term Semantic Memory：`Milvus 或 PGVector`

---

## 18. 联网增强设计

## 18.1 触发方式

只在以下条件下触发：

1. 用户显式开启 `web_enabled=true`
2. 内部知识不足或可能过时

## 18.2 联网工具职责

- 搜索
- 网页抓取
- 内容抽取
- 去广告
- 来源标记

## 18.3 联网结果使用规则

- 联网结果不得覆盖内部知识库结果，只能作为补充证据
- 联网引用必须单独标注
- 对时效性信息优先展示时间戳

---

## 19. 多用户逻辑隔离设计

虽然是单租户，但系统必须避免用户数据混淆。

## 19.1 隔离原则

- 所有知识库数据带 `user_id`
- 所有检索请求强制过滤 `user_id`
- 管理员公共知识库通过 `acl_tags` 控制访问

## 19.2 风险控制

- 不允许跨用户检索
- 不允许记忆串用
- 会话上下文必须绑定 `session_id + user_id`

---

## 20. 可观测性与治理

## 20.1 日志

必须记录：

- 原始 query
- rewrite 结果
- 子问题
- 检索候选
- rerank 结果
- 反思结果
- 最终答案
- 引用
- 总耗时

## 20.2 Tracing

建议接入 LangSmith 或自建 tracing：

- 节点级耗时
- 模型调用耗时
- 工具调用耗时
- 检索耗时

## 20.3 关键指标

- 检索命中率
- rerank 后有效率
- 答案引用率
- 幻觉率
- 用户满意度
- 平均时延
- 平均 token 成本
- 联网触发率

---

## 21. 评测体系

## 21.1 离线评测

建议构建评测集：

- 单跳问答
- 多跳问答
- 图文问答
- 时效性问答
- 无答案问答

评测指标：

- Recall@k
- MRR
- nDCG
- Faithfulness
- Answer Relevance
- Citation Accuracy

## 21.2 在线评测

- 用户反馈
- 点赞/踩
- 问后追问率
- 联网增强使用率
- 人工复核样本

---

## 22. 推荐开发阶段划分

## Phase 1：最小可用版本

包含：

- 文档上传
- 文本解析
- 分块
- Milvus dense + sparse 检索
- 基础 RAG
- SSE 流式输出

## Phase 2：增强检索版本

包含：

- 重排
- parent-child chunk
- 用户记忆
- 多用户逻辑隔离

## Phase 3：Agentic RAG 版本

包含：

- LangGraph 状态机
- 多步问题拆解
- 自我纠错
- 反思节点
- 答案验证

## Phase 4：多模态 + 联网增强版本

包含：

- 图片 OCR / caption
- 图片向量
- 联网工具
- 外网证据补充

---

## 23. 风险与注意事项

1. 不要一开始就把反思做得太复杂
2. 不要把所有数据都塞进一个统一 embedding 流程
3. 不要只做 dense 检索
4. 不要让联网结果和内部知识混淆
5. 不要让记忆模块替代知识库
6. 不要把逻辑隔离只放在前端，后端和检索层必须强制过滤

---

## 24. 最终推荐方案总结

本系统推荐的最终形态是：

- `LangGraph` 负责 Agentic RAG 状态编排
- `LangChain` 负责模型、Prompt、Retriever、Tool 封装
- `Milvus` 负责 Dense + Sparse + 多模态检索
- `PostgreSQL` 负责会话、用户、记忆、配置
- `Redis` 负责缓存和短期状态
- `SSE` 负责统一流式输出

核心能力组合为：

- `Hybrid Retrieval`
- `Reranking`
- `Parent-child Chunk`
- `Reflection + Self-correction`
- `Optional Web Enhancement`
- `User Memory Injection`
- `Structured Streaming Output`

这套设计兼顾了：

- 中英混合文档检索效果
- 高质量问答
- Agentic 推理能力
- 后续扩展性

---

## 25. 参考依据

本方案参考以下官方能力边界：

- [LangChain Retrieval Chain](https://docs.langchain.com/oss/python/integrations/vectorstores/aperturedb)
- [LangChain Streaming Retrieval](https://docs.langchain.com/oss/python/integrations/retrievers/ragatouille)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [Milvus Hybrid Search with Dense and Sparse Retrieval](https://milvus.io/docs/full_text_search_with_langchain)
- [Milvus Hybrid Search and Reranking Example](https://milvus.io/docs/contextual_retrieval_with_milvus)
- [Milvus Multimodal / Hybrid Search](https://milvus.io/docs/hybridsearch)
