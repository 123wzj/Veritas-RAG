# Agentic RAG 项目面试准备

## 1. 一分钟项目介绍

我做的是一个面向私域知识库的 Agentic RAG 智能问答系统。项目主要解决普通 RAG 在长文档场景下检索不准、上下文不完整、问题路由单一、答案容易幻觉和引用不可追溯的问题。

系统的核心架构是“子块向量检索 + 父块关系库回补 + Agent 控制流”。文档入库时，我把子块向量和检索元数据写入 Chroma，把父块和子块正文写入 MySQL；默认用本地 BGE-M3 做 embedding、用本地 BGE reranker 做精排，LLM 通过 OpenAI-compatible 接口接入。用户提问后，系统会先做 Query Rewrite、Query Decomposition 和 Route Planning，再根据问题类型走 `chat / web_search / knowledge_base / hybrid` 不同路由。知识库检索时，会用 Dense + Independent Sparse 双路召回，再通过 Weighted RRF 融合、MySQL 父块回补、Reranker 精排、证据评估、反思和验证生成可追溯答案。

---

## 2. 高频问题

### Q1：请解释RAG的工作原理。与直接对LLM进行微调相比，RAG主要解决了什么问题？

RAG（Retrieval-Augmented Generation)的核心是“检索+生成”两阶段：先从外部知识库中检索与用户问题相关的文档片段，再将这些片段作为上下文拼接到prompt，让LLM基于检索到的信息生成回答。相比微调，RAG的优势在于：

1. 知识可以实时更新，不需要重新训练模型
2. 可追溯来源，减少幻觉且便于审计
3. 成本低，不需要GPU资源做训练
4. 领域迁移够方便，换一套知识库即可适配新场景。微调的优势则在于风格适配和深层知识内化，实际中常将两者结合：微调让模型学会“如何利用检索结果”，RAG提供最新知识。

### Q2：为什么要做 Parent-Child Chunking？

因为小块适合检索，大块适合生成。建议用 RAGAS 等工具做 A/B 测试，用检索召回率和最终回答质量来选最优参数。

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

**Dense 检索解决语义匹配。**
 它把 query 和 chunk 编码成向量，按向量相似度找内容。优点是用户不一定说出原文关键词，也能找到语义相关内容。比如用户问“怎么防止模型胡说”，dense 可能召回到“幻觉抑制、证据约束、引用生成”相关内容。

**Sparse 检索解决精确匹配。**
 它基于关键词、术语、编号、专有名词、英文缩写、API 名称进行匹配。比如用户问 `MAX_REFLECTION_ROUNDS`、`bge-reranker-v2-m3`、`session_id`，dense 未必稳定，但 sparse 很容易命中。

所以双路检索不是重复，而是互补：

> | 检索方式 | 擅长                                 | 容易失败的场景                     |
> | -------- | ------------------------------------ | ---------------------------------- |
> | Dense    | 语义相似、同义表达、用户自然语言问题 | 专有名词、编号、字段名、代码变量名 |
> | Sparse   | 精确词、术语、代码符号、标题、编号   | 用户换一种说法、没有出现原文关键词 |
>
> 更合理的讲法是：
>
> 第一阶段检索采用 Dense + Sparse 的双路召回。Dense 负责语义泛化，Sparse 负责关键词和结构化术语命中。两路结果不直接比较原始分数，因为向量相似度和 BM25 分数分布不同，所以系统使用 Weighted RRF 做排名融合。这样可以避免某一路分数天然偏高导致另一种证据被压掉。

你当前的参数逻辑可以这样解释：

用户最终需要的证据数量是 `top_k`，默认 6，最大 12。但这 6 条不是直接从向量库取出来的，而是经历了：

```
用户问题
  ↓
query rewrite / decompose / route
  ↓
Dense 检索召回一批候选
Sparse 检索召回一批候选
  ↓
Weighted RRF 融合
  ↓
Reranker 精排
  ↓
Evidence packing 控制数量和 token
  ↓
生成模型使用最终证据
```

默认情况下：

```
最终证据 top_k = 6
retrieve_hybrid 传入 top_k = max(6*3, 12) = 18
retriever 内部 desired_top_k = 18
Dense 路召回约 90 条
Sparse 路召回约 90 条
融合后进入 rerank
rerank 后保留约 12 条
最终 prompt 里放 6 条左右
```

这背后的设计依据是：

**第一阶段宁可多召回，不能漏召回。**
 因为如果第一阶段没把相关文档召回来，后面的 reranker 和 LLM 都没有机会修复。

**第二阶段再靠 reranker 精排。**
 Dense 和 Sparse 只是粗筛，它们不能真正理解“这个文档是否能回答当前问题”。reranker 才是判断 query-document 匹配程度的关键。

**第三阶段 evidence packing 控制成本。**
 最终进入 prompt 的证据不能太多，否则会增加 token 成本，也会把无关信息带进生成模型，引发回答跑偏。

可以用一句话总结：

> 检索层采用“宽召回、精排序、窄注入”的策略：前面扩大候选集合保证召回率，中间用 reranker 提高相关性，最后按 token budget 和引用需求压缩进入 prompt 的证据数量。



### Q6：Independent Sparse 和普通 lexical re-score 有什么区别？

区别在于是否能补召回。

可以这样回答：

> 如果 sparse 只在 dense 候选里打分，它只能改排序，不能补召回。当前实现里 sparse 是独立 first-stage recall，会从 MySQL 的 child corpus 单独召回一批候选，再和 dense 结果做 RRF 融合，这样它能把 dense 没召回的术语型片段补回来。

### Q7：为什么用 RRF？

Dense 的分数可能是余弦相似度，例如 0.73、0.81；Sparse 的分数可能是 BM25，例如 8.5、15.2。它们的数值范围完全不同，直接相加不公平。

所以用 RRF，核心思想是：

```
不看原始分数，看每个候选在各路检索中的排名。
排名越靠前，贡献越大。
如果一个文档在 Dense 和 Sparse 里都靠前，它的融合分就会更高。
```

可以这样写：

```
score(d) = dense_weight / (k + dense_rank)
         + sparse_weight / (k + sparse_rank)
```

如果一个文档只在 Dense 中出现，它也有分；只在 Sparse 中出现，也有分；如果两路都出现，就会被加强。

你当前 dense 权重 1.0，sparse 权重 0.9，可以解释为：

> Dense 作为主召回通道，权重略高；Sparse 作为术语和关键词补偿通道，权重接近 Dense。这样既保证自然语言语义召回，又不会丢失专有名词、代码字段和标题编号类信息。

### Q8：为什么还需要 Reranker？

Reranker 和 Dense 检索最大的区别在于：

**Dense 检索是双塔模型。**
 query 编码一次，document 编码一次，然后比较两个向量。速度快，适合大规模召回，但它对 query 和 document 的细粒度交互理解有限。

**Reranker 是 cross-encoder / pair scorer。**
 它把 `(query, document)` 放在一起输入模型，让模型直接判断“这个 document 对 query 有多相关”。它更慢，但判断更准，所以只适合对候选集精排。

可以这样讲：

> Dense 检索解决“从大量文档中快速找出可能相关的候选”，reranker 解决“在候选中判断哪些真的适合回答当前问题”。因此 reranker 不是替代检索，而是检索后的精排层。

你当前 rerank 输入不是单纯 child chunk，而是：

```
parent_content
+ title
+ section_path
+ matched child chunk
```

这个设计很关键，要展开讲。

原因是 child chunk 很短，可能只有局部片段。单看 child chunk，reranker 可能误判；拼上 parent 内容、标题和章节路径后，reranker 能看到更完整的语义上下文。

可以这样解释：

> 系统没有直接把 child chunk 丢给 reranker，而是构造 parent-aware rerank 文本。这样做是因为 child chunk 适合精确定位，但信息可能不完整；parent_content 提供上下文，title 和 section_path 提供文档结构，matched child 片段提供命中依据。reranker 判断的是“这个证据块整体是否能回答 query”，而不是只判断某一句是否相似。

### Rerank 后为什么保留 `top_k * 2`，而不是直接保留最终 top_k

你可以这样答：

> Rerank 后不会立刻裁成最终 top_k，因为后面还有去重、父块回补、证据覆盖检查、引用构造和 token packing。如果精排后直接只留 6 条，一旦里面有重复父块、证据覆盖不均、引用不完整，就没有补救空间。因此系统先保留 `max(top_k*2, 8)`，给后续 evidence packing 留余量，最终再根据子问题覆盖、token budget 和引用需求压缩成真正进入 prompt 的证据。

换句话说：

```
rerank_top = 候选池里最相关的一批
final_evidence = 真正放进 prompt 的少量证据
```

它们不是同一个概念。

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



### Q13：一个完整的RAG流水线包含哪些关键步骤？

完整流程分为离线索引和在线查询两部分。

**离线索引**：1.数据采集（PDF/网页/数据库等多源数据）----》2.文档解析（表格、图片用专门工具如LIamaParse处理）---》3.文本切块（按语义/固定长度切分，设置重叠）---》4.向量化（用Embedding模型如BGE/M3E生成向量）---》5.存入向量数据库（Milvus/Chroma/Weaviate）。

**在线查询**：1.Query改写（扩展、纠错、多语言统一）---》2.检索（向量检索+BM25混合）---》3.重排序（用Cross-Encoder 如BGE-Reranker精排）---》4.Prompt组装（将top-k文档拼入system prompt）---》5.LLM生成 ---》6.后处理（格式化、来源引用标注）。



### Q14:如何选择一个合适的嵌入模型？评估一个Embedding模型的好坏有哪些指标？

选型考虑：1.语言支持（中文场景优先BGE、M3E、GTE，英文可以用OpenAI text-embedding-3）；2.维度和性能（高精度好但存储和检索成本高）；3.最大输入长度（影响切块策略）；4.推理速度和部署成本。

评估指标：在MTEB（Massive Text Embedding Benchmark）排行榜上对比，核心指标包——Recall@K（检索召回率）、MRR（平均倒数排名）、NDCG（归一化折损累积增益）。实际中要用自己的领域数据构造测试集评估，因为通用排行榜上的冠军在你的垂直领域未必最优。还要考虑对称vs非对称检索——query短，document长时需要非对称模型（如E5系列）效果更好。



### Q15：除了基础的向量检索，你还知道哪些可以提升RAG检索质量的技术？

**混合检索**：向量检索+BM25互补，用RRF融合排序

**query改写**：HyDE（让LLM先生成假设性答案再用答案去检索）、Query Expansion（阔同义词）、Multi-Query（将一个问题拆成多个子问题分别检索再合并）。

**重排序**：检索后用Cross-Encoder（如BGE-Reranker、Cohere Rerank）对候选文档精排，大幅提升精度。

**知识图谱增强**：对实体关系密集的场景（如医疗、金融），用图检索补充结构化知识。

**上下文压缩：**对检索到的长文档做摘要或提取关键句，减少噪声。

**递归检索**：先粗检索大范围文档，再在文档内精检索具体段落。



### Q16：请解释“Lost in the Middle”问题。

“Lost in the Middle”是Stanford 2023年研究发现：当多个检索文档放入LLM的上下文时，模型倾向于关注开头和结尾的文档，而忽略中间部分的信息。即使正确答案在中间位置，回答准确率也会显著下降。

**缓解方法**：1.将最相关的文档放在开头或者结尾（按相关性交替排列而非顺序排列）2.减少输入文档数量，只保留top-3而非top-103.使用Map-Reduce策略（先让LLM对每个文档单独提取信息，在合并）4.使用长上下文优化过的模型5.对检索结果做上下文压缩，缩短每个文档的长度。



### Q17: 在什么场景下，你会选择使用图数据库或知识图谱来增强或替代传统的向量数据库检索？

**适合知识图谱的场景**：①需要多跳推理（如"张三的老板的母校是哪？"需要跨两个关系跳转）；②实体关系密集的领域（医学：药物-疾病-症状、金融：公司-人物-持股）；③需要精确的结构化查询（"列出所有市值超过100亿且在北京的公司"）；④数据有明确的图结构（组织架构、社交网络）。

**实践方案**：GraphRAG（微软开源）先对文档做实体抽取和关系建图，查询时用图遍历+向量检索混合。也可以用 Neo4j 存知识图谱，Cypher 查询结构化关系，向量库查语义相关性，两路结果融合。知识图谱的劣势是构建成本高、实体抽取质量依赖模型能力、难以覆盖非结构化长文本。

### Q18: 如何全面地评估一个 RAG 系统的性能？

分两阶段评估。

**检索阶段**：Recall@K（top-K 中包含正确文档的比例）、Precision@K（top-K 中正确文档的占比）、MRR（正确文档首次出现的排名倒数）、NDCG（考虑排名位置的加权指标）。

**生成阶段**：Faithfulness（回答是否忠实于检索到的文档，不编造）、Answer Relevancy（回答是否切题）、Context Relevancy（检索文档是否与问题相关）。

**端到端指标**：任务完成率、人工评分。

**工具**：RAGAS 框架可以自动化评估 Faithfulness/Relevancy/Context Precision 等，LangSmith 可做链路追踪和质量监控。建议构建 golden test set（100-200 条带标注的 QA 对），每次迭代后回归测试。

### Q19: 你是否了解一些更复杂的 RAG 范式，比如自适应检索？

**Adaptive RAG**：根据查询复杂度动态决定是否需要检索。简单问题（如"法国首都是哪"）LLM 直接回答，复杂问题才触发检索。可用分类器或 LLM 自身判断。

**Iterative RAG（迭代式）**：ITER-RETGEN 等方法在生成过程中发现信息不足时再次检索，多轮迭代直到信息充分。

**Self-RAG**：模型自己判断是否需要检索、检索到的文档是否有用、生成的回答是否被检索内容支持，通过特殊 token 实现自省。

**Corrective RAG (CRAG)**：对检索结果做可信度评估，不可信时回退到 web 搜索或直接拒答。

**Agentic RAG**：将 RAG 作为 Agent 的一个工具，Agent 自主决定何时检索、检索什么、检索几次，结合规划和反思能力。实际中 Agentic RAG 是当前的主流趋势。

### Q20: RAG 系统在实际部署中可能面临哪些挑战？

**数据层**：文档格式多样（PDF 表格、扫描件、图片）解析困难；数据更新频率高时增量索引成本大；多语言混合场景下 Embedding 效果差。

**检索层**：长尾 query 召回率低；同义词/近义词匹配不准；检索延迟在高并发下增长。

**生成层**：幻觉（检索到了但没用或编造了不在文档中的信息）；上下文窗口不够放所有相关文档；生成结果不可控（格式/长度）。

**工程层**：向量数据库选型和运维成本；Embedding 模型更新后需要全量重建索引；缓存策略设计；监控和评估体系搭建。

**安全层**：知识库中的敏感信息可能被检索出来返回给无权限用户，需要做文档级别的权限控制。

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
