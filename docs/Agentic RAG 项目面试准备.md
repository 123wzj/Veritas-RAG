# Agentic RAG 项目面试准备

上级导航：[`docs/README.md`](README.md) · 所属模块：表达与面试材料（以代码和现状文档为准）

## 1. 一分钟项目介绍

我做的是一个面向私域知识库的 Agentic RAG 智能问答系统。项目主要解决普通 RAG 在长文档场景下检索不准、上下文不完整、问题路由单一、答案容易幻觉和引用不可追溯的问题。

系统的核心架构是“子块向量检索 + 父块关系库回补 + Agent 控制流”。文档入库时，我把子块向量和检索元数据写入 Chroma，把父块和子块正文写入 MySQL；默认用本地 BGE-M3 做 embedding、用本地 BGE reranker 做精排，LLM 通过 `DeepSeek` 接口接入。用户提问后，系统会先做 Query Rewrite、Query Decomposition 和 Route Planning，再根据问题类型走 `chat / web_search / knowledge_base / hybrid` 不同路由。知识库检索时，会用 Dense + Sparse 双路召回，再通过 Weighted RRF 融合、MySQL 父块回补、Reranker 精排、证据评估、反思和验证生成可追溯答案。

---

## 2. 高频问题

### Q1：请解释RAG的工作原理。与直接对LLM进行微调相比，RAG主要解决了什么问题？

RAG（Retrieval-Augmented Generation)的核心是“检索+生成”两阶段：先从外部知识库中检索与用户问题相关的文档片段，再将这些片段作为上下文拼接到prompt，让LLM基于检索到的信息生成回答。相比微调，RAG的优势在于：

1. 知识可以实时更新，不需要重新训练模型
2. 可追溯来源，减少幻觉且便于审计
3. 成本低，不需要GPU资源做训练
4. 领域迁移够方便，换一套知识库即可适配新场景。微调的优势则在于风格适配和深层知识内化，实际中常将两者结合：微调让模型学会“如何利用检索结果”，RAG提供最新知识。

### Q2：当前项目对原始文档是怎么分块的？为什么这样分？

当前链路是：

```text
原始文件
  -> parser.py 按文件格式解析
  -> chunker.py 先生成 Parent Chunk
  -> 每个 Parent 再按句子生成 Child Chunk
  -> Child 做召回，命中后通过 parent_id 回补 Parent
```

不同格式的 Parent 划分方式不同：

| 文件类型 | 当前 Parent 划分方式 | 主要原因 |
|---|---|---|
| PDF | 默认一页一个 Parent，超过约 1500 个估算 token 再按句子切 | 页码天然适合引用和回溯 |
| DOCX | 按段落顺序合并，接近 1000 个估算 token 时切分，并记录标题章节 | 尽量保留标题和段落结构 |
| PPTX | 一张幻灯片一个 Parent | 一页幻灯片通常表达一个主题 |
| TXT | 按空行分段，再合并到约 1000 个估算 token | TXT 没有可靠标题结构 |
| Markdown | 按标题章节生成 Parent，过大时继续按句子切 | Markdown 标题是较强的语义边界 |
| HTML | 当前先抽取整页文本，再对超长内容按句子切 | 当前实现只做了基础文本化 |
| 独立图片 | OCR 文本整体作为 Parent，再生成 Child | 先把图片转换成可检索文本 |

Child 的处理方式是：

1. 在每个 Parent 内按中英文句号、问号、感叹号等切句。
2. 逐句合并到约 `300` 个估算 token。
3. 相邻 Child 保留约 `50` 的重叠，减少边界处信息丢失。
4. 每个 Child 记录 `parent_id`、标题、章节和页码。

当前代码里的实际默认值是：

```text
Parent 目标大小：1000
Child 目标大小：300
Child overlap：50
Parent 超过 1500 左右时继续切分
```

需要注意两个实现细节：

- 当前 `estimate_tokens()` 用“中文字符数 + 英文单词数”估算，不是真实 tokenizer 计数。
- `config.py` 虽然也声明了分块参数，但全局 `document_chunker` 目前仍按构造函数默认值创建，配置没有真正注入；`parent_chunk_overlap` 也只是声明了，当前没有生效。

为什么使用 Parent-Child：

> 小块更容易准确命中用户问题，但容易丢上下文；大块上下文完整，但向量语义容易被很多无关内容稀释。项目把“召回粒度”和“生成粒度”分开：用 Child 定位，用 Parent 补上下文，再交给 Reranker 和生成模型。

### Q2.1：Parent 划分会不会造成语义割裂？当前怎么处理？

会。Parent-Child 只能缓解语义割裂，不能自动消除。

当前已经做的处理：

- 优先使用页、段落、标题、幻灯片等文档结构作为 Parent 边界。
- Parent 太大时尽量按完整句子继续切，不直接按固定字符硬切。
- Child 有 overlap，命中后还会回补 Parent。
- Reranker 使用 `标题 + 章节 + Parent 正文 + 命中 Child`，不是只看孤立 Child。

当前仍存在的风险：

- PDF 按页切时，跨页段落、跨页表格会被分开。
- DOCX 新标题附近可能出现上一段正文和下一章节元数据混在一起的问题。
- Markdown 一个章节可能很长，二次按句切后仍可能丢失章节内的局部关系。
- Parent 没有启用 overlap，恰好落在 Parent 边界上的信息只能依赖 Child overlap 补偿。
- 超长单句、代码、表格缺少句号时，可能生成超过目标大小的块。

更稳妥的生产方案：

1. 先把文档解析为 `heading / paragraph / table / image / list / code` 等结构化 block。
2. 优先在标题、段落、列表结束处切分，不在表格行、代码块和一句话中间切。
3. 使用 BGE tokenizer 做真实 token 计数，并对超长单句做二级硬切。
4. 给相邻 Parent 增加少量结构化 overlap，或在回补时同时取前后相邻 Parent。
5. 用业务评测集对 `Parent 800/1000/1500`、`Child 200/300/500` 做 A/B 测试，而不是凭经验固定参数。

面试时可以这样总结：

> 当前实现是结构优先、长度兜底：先按页、标题、段落等自然边界生成 Parent，过长时再按句子切；Child 负责精确召回，Parent 回补上下文。它能降低语义割裂，但 PDF 跨页、长表格和超长单句仍是已知边界，生产版需要升级成 layout-aware 的结构化分块。

### Q2.2：表格会不会被切到不同子块？如果出现怎么办？

会，而且当前项目对表格的支持还不完整。

当前真实情况：

- DOCX 解析器已经能提取 `tables`，但 `chunker.py` 目前只消费 `paragraphs`，表格还没有进入索引。
- Markdown 表格会跟着章节文本进入 Parent，但 Child 按句子切，不能保证表头和数据行始终在同一块。
- PDF 表格由 `pypdf` 当普通文本抽取，行列结构可能被打乱；跨页表格还会被拆到不同 Parent。
- PPTX 只读取 shape 文本，没有专门恢复表格结构。

因此不能在面试中说“当前已经完整支持表格 RAG”，更准确的说法是“已支持基础文本解析，但表格结构化入库仍需增强”。

生产级处理方式：

```text
表格
  -> 保留 table_id、表名、页码、表头、行列和 bbox
  -> 小表整体作为一个结构化块
  -> 大表按行组切分，每个子块重复表名和表头
  -> 同时生成 Markdown/HTML 表格文本和一段表格摘要
  -> 检索命中任一行组后，按 table_id 回补相邻行组或完整小表
```

切表原则：

- 不在一行中间切分。
- 每个块重复表头，否则单独一行没有列含义。
- 数字、单位和列名必须一起保留。
- 大表按行组切，不按字符数盲切。
- 表格摘要用于语义召回，原始行列用于精确回答和引用。

### Q2.3：为什么选择 BGE-M3？能处理多大的块？向量是多少维？

选择 BGE-M3 主要不是因为“模型名字新”，而是它比较适合当前中文私域知识库：

- 中英文和多语言能力较好，适合中文文档夹杂英文术语、代码字段和产品型号。
- 支持本地部署，私域文档不必发送给外部 embedding 服务。
- 官方模型支持 Dense、Learned Sparse 和 Multi-Vector 三种检索表示，后续有统一升级空间。
- 官方最大输入长度为 `8192 tokens`，Dense 向量维度为 `1024`。

但“模型最大支持 8192”不等于项目应该把 8192 token 全塞进一个 Child。

当前项目的真实设计是：

```text
BGE-M3 官方能力：最大 8192 tokens，1024 维
项目 EMBEDDING_MAX_LENGTH：512
项目 Child 目标大小：约 300 个估算 token
```

为什么项目只用 300 左右：

- Child 越大，主题越容易混杂，召回定位越粗。
- 300 左右通常能容纳一到数个完整段落，又能给 overlap 留空间。
- 512 的编码上限可以控制显存、吞吐和批处理延迟。
- 完整上下文由 Parent 回补，不需要让 Child 承担全部上下文。

当前风险是：分块使用估算 token，且超长单句不会被强制切开，所以个别 Child 仍可能超过 512，编码时后半部分被截断。后续应使用模型 tokenizer 做硬校验。

还要注意：

> 当前项目 Dense 使用 BGE-M3，但独立 Sparse 路实际使用的是自实现 BM25，不是 BGE-M3 learned sparse。`bge_m3.py` 虽然封装了 sparse 输出能力，当前入库和检索主链路没有使用它。

### Q2.4：当前项目整体是怎么设计的？为什么这么设计？

入库侧：

```text
文档解析
  -> Parent-Child 分块
  -> Child 生成 BGE-M3 Dense 向量
  -> Child 生成 BM25 词频 Sparse 表示
  -> Child Dense 向量和检索元数据写入 Chroma
  -> Parent/Child 正文、结构元数据和 Sparse 表示写入 MySQL
```

查询侧：

```text
原问题 + 改写问题 + 子问题
  -> Dense 独立召回
  -> BM25 Sparse 独立召回
  -> Weighted RRF 融合
  -> doc/parent/chunk 多样性限制
  -> parent_id 回补 Parent
  -> BGE Reranker 精排
  -> Evidence Packing
  -> Evidence Grading
  -> Reflection / Web Search
  -> 生成、引用和验证
```

这样设计的原因：

- Chroma 专注向量近邻搜索，不承担长正文和复杂业务关系。
- MySQL 保存正文、父子关系、ACL 和业务数据，便于事务、审计和回补。
- Dense 解决语义改写，BM25 解决术语、编号、变量名和数字的精确命中。
- RRF 按排名融合，避免 Dense 和 BM25 原始分数尺度不同。
- 多样性限制避免同一文档、同一 Parent 的相似 Child 占满 Top-K。
- Reranker 只处理有限候选，用更高计算成本换取更准确的最终排序。
- Evidence Gate 把“相关”与“足以回答”分开，避免只看检索分数就强行回答。

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

当前生效的主链路不是简单判断“相似度大于多少”，而是对每个子问题做严格证据分类：

| 证据状态 | 含义 | 下一步 |
|---|---|---|
| `direct_support` | 证据明确写出了问题要求的实体和属性 | 正常回答 |
| `partial_support` | 只覆盖了复合问题的一部分 | 部分回答，并指出缺失内容 |
| `background_only` | 主题相关，但没有直接回答所问事实 | 联网、反思或拒答 |
| `no_support` | 没有可用证据 | 联网、反思或拒答 |
| `conflict` | 多条证据对同一事实给出冲突值 | 尝试权威来源校验，否则说明冲突并拒绝下结论 |

每个复杂问题先拆成多个 answer slot，必须逐个检查。只要仍有未覆盖子问题，系统就不会把整体问题直接判为“证据充分”。

路由优先级大致是：

```text
需要联网且用户允许
  -> Web Search
否则证据完整
  -> Generate
否则未超过反思上限且还能重检索
  -> Reflection
否则
  -> 带着证据状态进入受控生成，执行部分回答或拒答
```

生成后还会再检查 `grounded` 和 `useful`。如果答案没有被证据支撑或没有回答问题，并且反思次数未用完，就重新进入 Reflection。

### Q11：Reflection 会不会太慢？

会，所以 Reflection 被设计成按需触发的质量兜底，而不是每次请求都执行。

当前 Reflection 输入包括：

- 当前未覆盖的子问题。
- 当前检索 query 和已有候选。
- Evidence Grade 中的缺失点。
- 已选择证据和检索分数。
- 生成后 Verification 的失败原因。
- 用户是否允许联网、是否存在知识库。

Reflection 要输出：

```json
{
  "should_retry_retrieval": true,
  "new_query": "更具体的查询",
  "additional_queries": ["补充检索词"],
  "need_web_search": false,
  "focus": "下一轮重点补什么",
  "reason": "为什么要这样改"
}
```

之后系统会：

1. 对新旧 query 去重。
2. 只清空证据不足子问题的检索结果，不重跑已经覆盖的子问题。
3. 重新执行 Dense、BM25、RRF、父块回补、Rerank 和证据评审。
4. `reflection_count + 1`。
5. 达到 `MAX_REFLECTION_ROUNDS=3` 或最大步骤数后停止循环。

为什么说它“基本合理”：

- 只有 Evidence Gate 或 Verification 失败时才触发。
- 反思结果还要经过下一轮真实检索和证据评审，不能直接当答案。
- 联网需要用户开关允许。
- 有轮数和步骤数保护，避免无限循环。
- 已经充分的子问题不会被重复反思。

但当前仍有可改进点：

- 只校验 JSON 结构，没有强制检查新 query 是否真的覆盖 `missing_aspects`。
- 没有比较反思前后的新证据数量、覆盖率和排序提升。
- 没有“连续两轮没有新 chunk 就提前停止”的无收益检测。
- Reflection 和 Evidence Grading 都依赖 LLM，可能出现同一个模型自我确认。

生产版可以增加确定性约束：

```text
query 必须与上一轮有足够差异
且必须覆盖至少一个 missing_aspect
且新一轮至少出现新 chunk / 新来源 / 更高 slot coverage
否则停止反思并进入部分回答或拒答
```

面试时可以这样回答：

> Reflection 不是让模型自由地“再想一遍”，而是让它根据缺失证据生成下一轮检索计划。计划本身不算正确，只有重新检索后真的带来新证据，并通过 Evidence Gate，才算反思有效。

### Q12：如何控制幻觉？

三层控制。

可以这样回答：

> 第一层是检索前和检索中，通过 rewrite、decomposition、Dense/Sparse、RRF 和 Rerank 提高证据质量；第二层是生成时要求模型基于 evidence 输出并带 `[E#]` 引用；第三层是生成后做 groundedness/usefulness 验证，不通过就回到 reflection。

### Q12.1：拒答、部分回答和正常回答是怎么设计的？阈值怎么定？

当前回答模式由证据类别决定，不是由一个统一相似度阈值决定。

| Evidence Grade | Generation Mode | 行为 |
|---|---|---|
| `direct_support` | `normal_answer` | 只使用明确支持的证据生成答案 |
| `partial_support` | `partial_answer` | 输出已确认部分，并明确列出未覆盖部分 |
| `background_only` / `no_support` | `need_web` 或 `refusal` | 允许联网则先补证据，否则拒答 |
| `conflict` 且有可靠来源可消解 | `conflict_answer` | 说明存在冲突，再引用更可靠来源 |
| `conflict` 且无法消解 | `conflict_answer` | 不给确定结论，明确说明冲突 |

几个关键保护：

- `direct_support` 如果没有 `direct_evidence_ids`，会被自动降级为 `background_only`。
- `direct_support` 如果还有 `missing_aspects`，会被降级为 `partial_support`。
- 正常生成只接收 `allowed_citation_ids` 对应的证据，背景材料不能偷偷进入生成上下文。
- 拒答和受控部分回答由代码模板直接生成，不再调用生成 LLM 自由发挥。
- 如果证据评审 LLM 调用失败，当前严格回退为 `no_support`，优先少答而不是误答。

阈值方面要区分“当前生效”和“历史遗留”：

- 当前图使用 `judge_evidence_slots()`，核心是上述类别门控，没有使用统一数值阈值。
- 同一文件里旧的 `judge_evidence()` 仍保留 `coverage_score >= 0.72` 且 `answerability_score >= 0.72` 的规则，但当前 `graph.py` 没有调用它。
- 旧逻辑在评审失败时还会使用“至少 2 条证据且平均分不低于 0.18”的启发式，这也不是当前主链路。
- `confidence` 会被限制在 0 到 1，但当前主要用于展示和验证调整，不负责决定正常回答还是拒答。

为什么不用固定 rerank score 直接决定回答：

> Reranker 分数表示相关性，不等于事实正确概率。不同问题类型、文档长度和模型版本的分数分布会变化。一个片段可以高度相关，却没有明确回答用户要求的属性，所以回答门控更适合基于“是否存在直接证据”来做。

企业落地时，阈值应该这样校准：

1. 准备正常回答、部分可答、无答案、证据冲突四类验证集。
2. 把“错误回答”的成本设得高于“保守拒答”。
3. 分别统计回答准确率、拒答精确率、拒答召回率、部分回答正确率和覆盖率。
4. 根据业务风险选择阈值，例如法律、医疗、财务场景应更保守。
5. 每次更换 embedding、reranker、LLM 或分块参数后重新标定。



### Q13：一个完整的RAG流水线包含哪些关键步骤？

完整流程分为离线索引和在线查询两部分。

**当前项目离线索引**：

```text
文件上传与哈希去重
  -> PDF/DOCX/PPTX/Markdown/HTML/TXT/图片解析
  -> Parent-Child 分块
  -> Child BGE-M3 Dense 向量 + BM25 Sparse 表示
  -> Child 向量写入 Chroma
  -> Parent/Child 正文和结构元数据写入 MySQL
```

**当前项目在线查询**：

```text
加载记忆
  -> Query Rewrite
  -> Query Decomposition
  -> 子问题级 Route Planning
  -> Dense + BM25 独立召回
  -> Weighted RRF
  -> Parent 回补
  -> BGE Reranker
  -> Evidence Packing / Grading
  -> Reflection / Web Search
  -> 子问题回答与聚合
  -> 引用、Verification、写入记忆
```

当前项目比基础流水线多了三层：

- 子问题级 Route Planning，决定闲聊、知识库、联网或混合路线。
- Evidence Grading + Reflection，判断证据是否真的能回答，而不只是是否相关。
- Citation + Verification，把答案和 `[E#]` 证据绑定，并在生成后检查 grounded/useful。

### Q13.1：当前项目可以处理图片吗？包含图片的文档应该怎么设计？

当前只能算“有限图片支持”，还不是完整多模态 RAG。

已经具备的能力：

- 上传独立的 `PNG/JPG/JPEG` 文件。
- 通过 Qwen Vision 或 PaddleOCR 提取图片文字。
- OCR 结果按普通文本进行分块、向量化和检索。
- Qwen Vision 的提示词会要求表格尽量转换为 Markdown。

当前没有完整实现的能力：

- PDF、DOCX、PPTX 中的内嵌图片没有被单独提取和建立图片块。
- 扫描 PDF 虽然会尝试判断 `is_scanned`，但没有自动把每页渲染成图片后送入 OCR。
- 图表、流程图、示意图和照片中的视觉语义不能只靠 OCR 完整理解。
- `MultimodalEmbeddingService` 已有辅助代码，但没有接入当前 ingestion 和 retrieval 主链路。
- 图片 OCR 后的 Chunk 和 Document 目前仍被标记为 `TEXT`，没有真正形成多模态索引和路由。

所以面试时应该说：

> 当前项目支持独立图片 OCR 后进入文本 RAG，但尚未完成文档内嵌图片、图表理解和图文联合检索。完整多模态能力属于下一阶段设计。

更完整的多模态入库设计：

```text
文档解析
  -> 生成带阅读顺序的 block
     - text block
     - table block
     - image block
     - caption block
  -> 每个 block 保存 doc_id、page_no、block_id、bbox、parent_section
  -> 图片同时生成 OCR 文本、VLM 描述和图片向量
  -> 文本向量、图片向量分别索引
  -> 查询时判断走 text / image / hybrid retrieval
  -> 命中后回补同页文字、图片、标题和图注
```

如何保证图文一致：

1. **位置关联**：通过页码和 `bbox` 把图片与附近标题、图注、段落关联。
2. **身份关联**：图片、OCR、VLM 描述和周边文字共享同一个 `image_id/block_id`。
3. **双表示保留**：同时保存原图、OCR 原文和 VLM 描述，不能只保留一段摘要。
4. **一致性校验**：让校验模型检查 OCR 数字、VLM 描述和图注是否冲突，冲突时降低置信度。
5. **可追溯引用**：答案引用到具体页码、图片编号和区域，而不是只引用整个文档。
6. **评测覆盖**：单独构造图表问答、OCR 数字、跨图文推理和错误图注测试集。

图文一致不代表让 VLM “相信附近文字”，而是要求所有视觉描述都能回到同一文档位置和原始图片进行核验。


### Q14:如何选择一个合适的嵌入模型？评估一个Embedding模型的好坏有哪些指标？

模型选型至少看六件事：

1. **语言和领域**：中文、英文、代码、法律或医疗术语是否覆盖。
2. **输入长度**：模型上限和实际编码上限是否匹配分块策略。
3. **向量维度**：维度越高通常表达能力更强，但索引内存和检索成本也更高。
4. **检索形式**：是否支持 Dense、Sparse、Multi-Vector，是否方便做混合检索。
5. **部署成本**：显存、吞吐、批大小、量化和离线部署能力。
6. **领域实测**：通用排行榜只用于初筛，最终必须用真实业务 query 和文档评测。

当前选择 BGE-M3 的理由是：中文和多语言表现、1024 维 Dense 表示、本地部署、8192 token 官方上限，以及未来升级 learned sparse/multi-vector 的空间。

评估 Embedding 不只看一个总分：

- `Recall@K`：正确证据能不能被召回。
- `Hit@K`：至少能不能找到一条正确证据。
- `MRR@K`：第一条正确证据出现得是否足够早。
- `NDCG@K`：多条相关证据的整体排序是否合理。
- 推理吞吐、P95 延迟、GPU 显存和向量库占用。
- 按普通问法、同义改写、术语、数字、跨语言等 query 类型分别统计。

更可靠的选型方式是固定同一份 chunk 和 gold qrels，对候选模型执行离线 A/B 测试，再观察 Reranker 后的端到端提升，而不是直接照搬 MTEB 排名。



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

不能只评最终答案。完整评估至少要分六层：

| 层级 | 重点检查 | 建议指标 |
|---|---|---|
| 解析与分块 | 文本、标题、表格、图片有没有丢；边界是否合理 | 解析成功率、结构保留率、表格单元格准确率、OCR 准确率、超长块比例 |
| 检索 | 正确证据能否召回并排在前面 | Hit@K、Recall@K、MRR@K、NDCG@K、Precision@K、重复率 |
| 证据 | Top-K 是否干净、完整、无冲突 | Context Precision/Recall、slot coverage、冲突识别率、引用支撑率 |
| 生成 | 答案是否正确、忠实、切题 | Faithfulness、Answer Relevancy、Answer Correctness、Citation Quality |
| Agent 决策 | 路由和反思是否真正有收益 | Route Accuracy、Tool Precision/Recall、Reflection Recovery Rate、无收益循环率 |
| 工程与安全 | 延迟、成本、稳定性和权限是否达标 | P50/P95 延迟、token/请求、错误率、ACL 泄漏率、Prompt Injection 成功率 |

#### 当前检索指标是怎么实现的？

当前 `backend/evaluation/run_t2retrieval_eval.py` 在文档级计算四个指标：

**1. Hit@K：Top-K 里有没有至少一个正确文档**

```text
有至少一个相关文档 = 1
一个都没有 = 0
```

它回答“系统有没有基本找到答案来源”，适合单证据问题，但不能说明是否找全。

**2. Recall@K：全部正确文档找回了多少**

```text
Recall@K = Top-K 中去重后的相关文档数 / 全部相关文档数
```

它回答“证据覆盖是否完整”。复杂问题和多跳问题通常比 Hit@K 更依赖 Recall。

**3. MRR@K：第一条正确文档出现得多早**

```text
RR = 1 / 第一条相关文档排名
MRR = 所有 query 的 RR 平均值
```

第一条正确结果排第 1 得 1，排第 5 得 0.2。它适合看用户最快能否拿到有效证据，也适合衡量精排效果。

**4. NDCG@K：多条相关文档的整体顺序是否合理**

```text
NDCG@K = DCG@K / IDCG@K
```

相关性越高的文档排得越靠前，得分越高；代码会跳过重复 `doc_id`，避免同一文档重复计算收益。它适合 qrels 带多级相关性的场景。

为什么四个都要看：

```text
Hit 高、Recall 低：能找到一条，但其他关键证据漏了
Recall 高、MRR 低：能找全，但正确证据排得太后
MRR 高、NDCG 低：第一条不错，后面的整体排序仍然混乱
四项都高：既能找到、又能找全、还排得靠前
```

当前脚本没有计算 `Precision@K`，因为当前评测重点是召回和排序；但企业落地仍应补上 Precision、Unique Doc Ratio、同 Parent 重复率和无关证据率，防止上下文被噪声占满。

#### 当前 300 条 T2Retrieval 结果怎么看？

最新本地报告：

```text
data/evaluation/reports/t2retrieval_eval_kb8_20260705_110946.json
```

评测参数是 300 条 query、`top_k=10`、Dense/BM25 各召回 50、Rerank 候选 50。

| 阶段 | Hit@10 | MRR@10 | NDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| Dense | 0.9700 | 0.9526 | 0.8463 | 0.8432 |
| BM25 | 0.9233 | 0.8844 | 0.7245 | 0.7121 |
| RRF | 0.9633 | 0.9305 | 0.8337 | 0.8447 |
| Rerank | 0.9800 | 0.9527 | 0.9121 | 0.9126 |

结论不是“RRF 一定比 Dense 所有指标都高”，而是：

- BM25 能补术语型召回，但单独使用弱于 Dense。
- RRF 的 Recall 略高于 Dense，说明双路融合补回了一部分相关文档。
- Reranker + diversity control 明显提高了整体排序和证据覆盖。
- 指标仍然来自公开通用检索集，不能代替企业真实业务集。

#### RAGAS 是什么？当前项目怎么做？

RAGAS 是面向 RAG 的自动化评估框架。它通常用评估 LLM 和 Embedding 对“问题、检索上下文、系统回答、标准答案”进行评分。

当前入口：

```text
backend/evaluation/run_ragas_answer_eval.py
```

默认指标：

| 指标 | 回答的问题 |
|---|---|
| `faithfulness` | 回答中的事实是否都能被检索上下文支持 |
| `answer_relevancy` | 回答是否直接回应用户问题 |
| `context_precision` | 检索上下文是否相关且排序靠前 |
| `context_recall` | 标准答案需要的信息是否被检索完整 |
| `answer_correctness` | 最终回答与标准答案在事实层面是否一致 |

当前脚本的执行过程：

```text
评测样本 question + reference
  -> 调用真实 Agentic RAG 图
  -> 收集 final_answer、selected_evidence、citations、route、verification
  -> 整理为 RAGAS SingleTurnSample
  -> 使用评估 LLM + 项目 BGE Embedding 打分
  -> 输出逐条结果和平均分 JSON
```

数据至少要有：

```json
{
  "question": "用户问题",
  "reference": "人工确认的标准答案"
}
```

最好再增加：

```json
{
  "gold_doc_ids": ["正确文档 ID"],
  "question_type": "normal|multi_hop|partial|unanswerable|conflict|table|image"
}
```

当前实现有一个重要规则：只有所有样本都有 `reference` 时，才会启用 `context_precision`、`context_recall` 和 `answer_correctness` 等依赖参考答案的指标；否则会自动跳过。

#### RAGAS 评估应该怎么做才完整？

1. **先做 Golden Set**
   不要只用随机问题。至少覆盖普通可答、多跳、部分可答、无答案、冲突证据、数字日期、表格和图片问题。

2. **先跑 collect-only**
   先确认真实 RAG 图可以稳定产出答案、证据和引用，再调用 RAGAS，避免把系统错误误判为评估器错误。

3. **再跑 RAGAS**
   低温度、低并发、重试，并保存逐条结果，不能只看平均分。

4. **按问题类型分桶**
   平均分可能掩盖无答案和冲突问题。要分别看正常问答、拒答、部分回答和多跳问题。

5. **加入人工抽检**
   RAGAS 也是 LLM Judge，会受评估模型、Prompt 和参考答案质量影响。每次版本回归都要人工抽查低分样本和随机样本。

6. **同时看延迟和成本**
   Agentic RAG 不能只追求分数。Reflection Recovery Rate 提升很小但 P95 延迟翻倍时，可能不值得上线。

T2Retrieval 适合检索评估，不适合直接拿来做完整 RAGAS 生成评估，因为它主要提供 query、corpus 和 qrels，不一定有自然语言标准答案。生成评估应优先使用人工业务 QA，或 HotpotQA、RGB 等带答案或行为标签的数据。

面试时可以这样总结：

> 我把评估拆成解析、检索、证据、生成、Agent 决策和工程安全六层。T2Retrieval 用来定位 Dense、BM25、RRF、Rerank 哪一层有问题；RAGAS 用来评 Faithfulness、Relevancy、Context Precision/Recall 和 Answer Correctness；无答案、冲突、路由、反思和 ACL 则需要单独指标与人工抽检，不能只靠一个 RAGAS 平均分。

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

面试时不要说错：

- 当前 Sparse 是独立 BM25，不是 BGE-M3 learned sparse。
- BGE-M3 官方支持 8192 tokens，但项目当前编码上限配置是 512，Child 目标大小约 300。
- 当前支持独立图片 OCR，不等于已经支持 PDF/DOCX/PPTX 内嵌图片的完整多模态检索。
- DOCX 表格虽然已经解析，但还没有进入当前分块和索引主链路。
- 当前回答门控按证据类别判断，不是用 rerank score 或 `0.72` 单一阈值决定。
- `PARENT_CHUNK_OVERLAP` 当前没有真正生效，配置里的分块参数也还没有注入全局分块器。

---

## 5. 可背诵总结

一句话版：

> 我做的是一个 Agentic RAG 知识库问答系统，核心是通过 Parent-Child 分块、Dense + Independent Sparse 混合召回、RRF 融合、MySQL 父块回补、Rerank 精排、证据评估和反思验证闭环，让系统先判断该怎么找证据、证据够不够，再决定是否回答。

三句话版：

> 第一，入库侧我把 child chunk 向量和检索元数据写入 Chroma，把 parent/child 正文写入 MySQL，实现子块召回、父块回补。
> 第二，检索侧我做了 Query Rewrite、Decomposition、Dense + Independent Sparse、Weighted RRF 和 Reranker，提高复杂问题和术语型问题的召回质量。
> 第三，控制侧我用 LangGraph 做 Route Planning、Evidence Grading、Reflection 和 Verification，并通过 `[E#]` 引用机制提升答案可靠性和可追溯性。

---

## 6. 代码和资料锚点

当前项目代码：

- 文档解析：`backend/services/ingestion/parser.py`
- Parent-Child 分块：`backend/services/ingestion/chunker.py`
- 入库与存储：`backend/services/ingestion/ingestion.py`
- BGE-M3：`backend/embeddings/bge_m3.py`
- BM25 Sparse：`backend/embeddings/sparse.py`
- 混合检索：`backend/services/retrieval/hybrid.py`
- Reranker：`backend/services/retrieval/reranker.py`
- 证据评审：`backend/graph/nodes/retrieval_nodes.py`
- Reflection：`backend/graph/nodes/reflection_nodes.py`
- 拒答、部分回答和验证：`backend/graph/nodes/generation_nodes.py`
- T2Retrieval：`backend/evaluation/run_t2retrieval_eval.py`
- RAGAS：`backend/evaluation/run_ragas_answer_eval.py`

官方资料：

- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
