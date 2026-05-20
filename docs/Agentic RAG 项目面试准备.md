# Agentic RAG 项目面试准备

## 1. 一分钟项目介绍

我做的是一个面向私域知识库的 Agentic RAG 智能问答系统。项目主要解决普通 RAG 在长文档场景下检索不准、上下文不完整、问题路由单一、答案容易幻觉和引用不可追溯的问题。

系统的核心架构是“子块向量检索 + 父块关系库回补 + Agent 控制流”。文档入库时，我把子块向量和检索元数据写入 Chroma，把父块和子块正文写入 MySQL；默认用本地 BGE-M3 做 embedding、用本地 BGE reranker 做精排，LLM 通过 OpenAI-compatible 接口接入。用户提问后，系统会先做 Query Rewrite、Query Decomposition 和 Route Planning，再根据问题类型走 `chat / web_search / knowledge_base / hybrid` 不同路由。知识库检索时，会用 Dense + Independent Sparse 双路召回，再通过 Weighted RRF 融合、MySQL 父块回补、Reranker 精排、证据评估、反思和验证生成可追溯答案。

---

## 2. 高频问题

### Q1：这个项目和普通 RAG 最大区别是什么？

普通 RAG 更像固定链路：检索一次、拼上下文、直接生成。  
这个项目是 Agentic RAG，会先判断问题意图和路由，再根据证据质量决定下一步是生成、反思重检索还是联网补充。

可以这样回答：

> 普通 RAG 是固定流水线，我这个项目更像一个带状态机的问答 Agent。它会先做 query rewrite、decomposition 和 route planning，然后走知识库、联网或闲聊路由；检索后还会做 evidence grading，如果证据不足就 reflection 重检索，生成后再做 groundedness/usefulness 验证。

### Q2：为什么要做 Parent-Child Chunking？

因为小块适合检索，大块适合生成。

可以这样回答：

> Parent-Child 的核心是把“召回粒度”和“生成上下文”拆开。child chunk 用来做高精度召回，parent chunk 用来回补完整上下文。这样既能保证召回准确，也能避免生成时只看到孤立片段。

### Q3：为什么当前架构是 Chroma + MySQL？

因为两者职责不同。

可以这样回答：

> Chroma 负责子块向量召回和 metadata filter，MySQL 负责知识库、文档、父子块正文、会话和用户记忆。这样向量库只做检索，关系库负责正文和业务数据，结构更清晰，也方便后续扩展权限、审计和评测。

### Q4：为什么只把子块向量写入 Chroma？

因为 first-stage recall 更适合用子块。

可以这样回答：

> 子块更适合向量检索，父块更适合生成上下文。如果父块也都放进向量召回，会让召回粒度变粗。当前实现里，Chroma 只存 child 向量和检索元数据，命中 child 后再根据 parent_id 到 MySQL 回补 parent 正文。

### Q5：为什么要做 Dense + Sparse？

Dense 和 Sparse 解决的是两类不同问题。

可以这样回答：

> Dense 擅长语义相似，比如用户问法和文档表达不完全一样；Sparse 擅长关键词和术语命中，比如字段名、API 名、版本号、命令。知识库问答里这两类问题都会出现，所以我做了 Dense + Independent Sparse 双路召回。

### Q6：Independent Sparse 和普通 lexical re-score 有什么区别？

区别在于是否能补召回。

可以这样回答：

> 如果 sparse 只在 dense 候选里打分，它只能改排序，不能补召回。当前实现里 sparse 是独立 first-stage recall，会从 MySQL 的 child corpus 单独召回一批候选，再和 dense 结果做 RRF 融合，这样它能把 dense 没召回的术语型片段补回来。

### Q7：为什么用 RRF？

因为不同检索器分数不可直接比较。

可以这样回答：

> Dense 和 Sparse 的分数尺度不一样，直接相加不稳定。RRF 基于排名做融合，不强依赖原始分数归一化，适合多检索器、多 query 的工程融合。

### Q8：为什么还需要 Reranker？

召回和精排目标不同。

可以这样回答：

> 召回阶段目标是尽量别漏，所以候选会比较宽；Reranker 负责在候选里把最能回答问题的证据排到前面。当前实现里 Reranker 优先看 parent_content，同时保留 child snippet，这样更接近最终生成所需的证据形式。

### Q9：Route Planning 是做什么的？

它决定问题该走哪条工具链路。

可以这样回答：

> 我加了前置路由规划，不再让所有问题都强制走知识库检索。比如问候类问题走 chat，实时问题且允许联网时走 web_search，私域知识问题走 knowledge_base，知识库优先但可能需要外部补充时走 hybrid。这样系统更像智能体，而不是固定检索器。

### Q10：Agent 怎么决定继续检索还是直接回答？

靠证据评估和验证闭环。

可以这样回答：

> 检索后会做 Evidence Grading，判断覆盖度和可回答性。证据够就生成；证据不够但还有优化空间，就 reflection 改写 query 后重检索；如果需要外部信息且允许联网，就走 web search。生成后还会做 groundedness 和 usefulness 验证，不通过且没超过反思次数就继续回流。

### Q11：Reflection 会不会太慢？

会增加成本，所以需要限制。

可以这样回答：

> 会增加一定延迟，所以我设置了 `MAX_REFLECTION_ROUNDS=3` 和最大步骤数。它不是每个问题都强制反思，而是在证据不足或验证失败时才触发，属于质量兜底机制。

### Q12：如何控制幻觉？

三层控制。

可以这样回答：

> 第一层是检索前和检索中，通过 rewrite、decomposition、Dense/Sparse、RRF 和 Rerank 提高证据质量；第二层是生成时要求模型基于 evidence 输出并带 `[E#]` 引用；第三层是生成后做 groundedness/usefulness 验证，不通过就回到 reflection。

---

## 3. 项目难点

可以这样表达：

> 这个项目难点不是把 RAG 跑起来，而是平衡召回精度、上下文完整性和答案可靠性。我用 Parent-Child 解决检索粒度和上下文矛盾，用 Dense + Sparse + RRF 解决语义召回和关键词命中矛盾，用 Evidence Grading + Reflection + Verification 解决证据不足和生成幻觉问题。

---

## 4. 面试时建议重点强调

- 我不是只接了一个 embedding 接口，而是做了从文档入库到答案验证的完整闭环。
- 我把向量库和关系库职责拆开，Chroma 只负责子块向量检索，MySQL 存父子块正文和业务数据。
- 我把 sparse 做成独立召回，而不是 dense 候选里的二次打分。
- 我用 LangGraph 做了前置路由、证据判断、反思和验证，让系统具备动态决策能力。
- 我用 `[E#]` 引用把答案和证据绑定，提高了可追溯性。

---

## 5. 可背诵总结

一句话版：

> 我做的是一个 Agentic RAG 知识库问答系统，核心是通过 Parent-Child 分块、Dense + Independent Sparse 混合召回、RRF 融合、MySQL 父块回补、Rerank 精排、证据评估和反思验证闭环，让系统先判断该怎么找证据、证据够不够，再决定是否回答。

三句话版：

> 第一，入库侧我把 child chunk 向量和检索元数据写入 Chroma，把 parent/child 正文写入 MySQL，实现子块召回、父块回补。  
> 第二，检索侧我做了 Query Rewrite、Decomposition、Dense + Independent Sparse、Weighted RRF 和 Reranker，提高复杂问题和术语型问题的召回质量。  
> 第三，控制侧我用 LangGraph 做 Route Planning、Evidence Grading、Reflection 和 Verification，并通过 `[E#]` 引用机制提升答案可靠性和可追溯性。
