# T2Retrieval 检索评估扩容与去重优化记录

上级导航：[`docs/README.md`](README.md) · 所属模块：检索评测与优化记录

## 1. 这次做了什么

之前 retrieval 评估样本太少，只有 20 条左右，后面在 `kb5` 上最多也只能评估到 136 条有效 query。

这次新建了一个评估知识库：

```text
kb_id: 8
kb_name: public_eval_t2retrieval_300
dataset: mteb/T2Retrieval
eval_queries: 300
documents: 1504
parent_chunks: 1815
child_chunks: 4476
```

然后在同一批 300 条 query 上做了两轮评估：

- 优化前：`data/evaluation/reports/t2retrieval_eval_kb8_20260521_155934.json`
- 优化后：`data/evaluation/reports/t2retrieval_eval_kb8_20260521_160949.json`

评估命令保持一致，只改 retrieval 代码，方便前后对比。

---

## 2. 优化前的问题

优化前最大的问题不是“完全搜不到”，而是：

```text
同一篇文档的多个 child chunk 反复进入 top10，
把本来应该留给其他相关文档的位置挤掉了。
```

项目当前是 Parent-Child 分块：

- 一个文档会被拆成多个父块 parent chunk
- 每个父块又会拆成多个子块 child chunk
- dense / sparse 检索时召回的是 child chunk
- 后面再做 RRF 融合、父块回补、rerank 精排

所以如果某篇文档和 query 很相关，它下面的多个 child chunk 都可能被召回。
这在召回阶段是正常的，因为子块粒度更细，命中机会更多。

但问题出在后处理：

```text
融合和精排后，没有按 doc_id / parent_id 做去重。
```

结果就是 top10 里面可能有 4、5 条都来自同一篇文档。
看起来 top10 填满了，但实际上证据来源很单一，其他 gold document 被挤出去了。

优化前重复情况：

| 阶段 | 有重复文档的 query 数 | 平均重复 doc_id 数 |
|---|---:|---:|
| Dense | 281 / 300 | 3.347 |
| BM25 / Sparse | 271 / 300 | 2.930 |
| RRF | 285 / 300 | 3.340 |
| Rerank | 290 / 300 | 4.353 |

最明显的是 Rerank 阶段：
300 条 query 里有 290 条都出现了重复文档，平均每条 query 重复 4.353 个 doc_id。

这会带来两个问题：

- 评估上：`recall@10` 和 `ndcg@10` 被拉低，因为 top10 没有覆盖足够多的正确文档。
- 真实问答上：Evidence Packing 拿到的证据不够多样，后续生成答案容易只围绕同一篇文档打转。

---

## 3. 指标是什么意思

这次主要看 4 个指标：

| 指标 | 大白话解释 |
|---|---|
| `hit@10` | top10 里有没有至少一个正确文档。有一个就算命中。 |
| `mrr@10` | 第一个正确文档排得有多靠前。越靠前越好。 |
| `recall@10` | top10 找回了多少正确文档。适合看证据覆盖够不够。 |
| `ndcg@10` | 正确文档是否排得靠前，并且整体排序是否合理。 |

简单理解：

- `hit@10` 高，说明“基本能搜到”。
- `mrr@10` 高，说明“第一个正确结果来得早”。
- `recall@10` 高，说明“正确证据覆盖得多”。
- `ndcg@10` 高，说明“正确证据不仅找到了，而且排得好”。

本次问题的关键不是 `hit@10`，因为大部分 query 已经能搜到至少一个正确文档。
真正需要提升的是 `recall@10` 和 `ndcg@10`，也就是证据覆盖和排序质量。

---

## 4. 优化是怎么做的

这轮评估当时没有改 embedding、BM25、RRF 权重，也没有换 reranker 模型。

当时只做了一件事：

```text
对排序后的候选结果做去重，让 top10 尽量来自不同文档。
```

去重优先级：

```text
doc_id -> parent_id -> chunk_id
```

意思是：

1. 如果有 `doc_id`，优先按文档去重。
2. 如果没有 `doc_id`，再按 `parent_id` 去重。
3. 如果都没有，再按 `chunk_id` 去重。
4. 重复时保留排名最高的那条。

改动位置：

| 文件 | 优化点 |
|---|---|
| `backend/services/retrieval/hybrid.py` | 评估时曾在 RRF 融合后、父块回补后做硬去重；当前实现已调整为 rerank 前 soft cap。 |
| `backend/services/retrieval/reranker.py` | 评估时曾在 BGE reranker 和 Simple reranker 排序后硬去重；当前实现已调整为可配置 diversity selection。 |

具体到每一步，是这样处理的：

### 4.1 Dense / Sparse 原始召回阶段

原始召回阶段不急着做文档级去重。

这里不是 “Dense 召回 1 条、Sparse 召回 1 条”。

实际链路里，Dense 和 Sparse 都会各自召回一批 child chunk；如果 query rewrite / decomposition 产生了多条 retrieval query，每条 query 又会分别走 Dense 和 Sparse。也就是说，候选数量大致来自：

```text
query 数量 × 2 路检索 × 每路 topN child chunks
```

Dense 和 Sparse 都是先召回 child chunk：

- Dense 从 Chroma 里按向量相似度召回 `is_parent=false` 的子块。
- Sparse 从 MySQL 里加载当前知识库的 child 正文，按 lexical sparse 分数召回子块。
- 每一路都会返回多个 child，所以同一篇文档、同一个 parent 下面的多个 child 可能同时被召回。

原因是第一阶段目标是“尽量别漏”。如果在召回阶段过早按 `doc_id` 去重，可能会把同一文档里更匹配的后续 child 提前删掉，影响后面的 RRF 融合和 rerank 判断。

重复来源主要有两类：

1. 同一篇文档被切成多个 child，多个 child 都和 query 相关。
2. 多 query / Dense / Sparse 交叉命中，同一个 child 或同一篇文档反复出现。

### 4.2 RRF 融合内部先按 `chunk_id` 聚合

RRF 融合时，第一层处理是按 `chunk_id` 合并 Dense / Sparse / 多 query 的同一个 child chunk。

也就是说，如果同一个 `chunk_id` 同时被 Dense 和 Sparse 命中，或者被多个 query variant 命中，不会生成多条候选，而是把它们的 RRF 分数累加到同一个候选上。

这一层可以理解为“子块级合并”，解决的是“同一个 child chunk 被多路检索重复命中”的问题。

例如：

```text
query A 的 Dense 第 3 名命中 chunk_001
query A 的 Sparse 第 8 名也命中 chunk_001
query B 的 Dense 第 5 名又命中 chunk_001
```

RRF 不会保留 3 条 `chunk_001`，而是只保留 1 条 `chunk_001`，并把三次命中的 RRF 贡献加到这个 chunk 上。这样做的含义不是简单删除重复子块，而是认为“同一个子块被多路、多 query 反复命中，说明它更稳定相关”，所以给它更高融合分。

但它还不是文档级去重。因为同一篇文档下面可能有多个不同 `chunk_id`，它们在这一层仍然会同时存在。

例如同一个 `chunk_id`：

```
Dense 第 3 名命中
Sparse 第 8 名命中
Query variant B 的 Dense 第 5 名命中
```

RRF 会变成：

```
rrf_score =
1 / (k + 3)
+ 1 / (k + 8)
+ 1 / (k + 5)
```

如果有权重，也可以是：

```
rrf_score =
dense_weight  * 1 / (k + dense_rank)
+ sparse_weight * 1 / (k + sparse_rank)
+ query_weight  * 1 / (k + variant_rank)
```

所以含义是：**同一个 chunk 被多路、多 query 反复命中，说明它稳定相关，应该排名更靠前。**

### 4.3 RRF 排序后不再直接做 doc 级硬去重

RRF 分数算完并排序后，不应该立刻每个 `doc_id` 只保留 1 条。

原因是 RRF 仍然处在 rerank 前的候选准备阶段。如果这一步过早按文档硬去重，可能会删掉同一篇文档里更适合回答的其他 parent / child，尤其是长文档、多章节文档和多跳问题。

当前推荐做法是 soft cap：

```text
每个 doc 最多保留 3～5 条
每个 parent 最多保留 1～2 条
总候选控制在 50～100 条进入 rerank
```

也就是说，RRF 后不是不控重复，而是不做“一篇文档只留一条”的硬去重。

当前代码中使用：

```python
cap_by_group(fused, max_per_doc=5, max_per_parent=2, max_per_chunk=1)
```

含义是：

1. 同一个 `chunk_id` 仍然只保留 1 条。
2. 同一个 `parent_id` 最多保留 2 条，避免一个父块下多个相似 child 占太多位置。
3. 同一个 `doc_id` 最多保留 5 条，保留长文档里多个相关父块进入 rerank 的机会。

### 4.4 父块回补后继续 soft cap

RRF 后拿到的仍然是 child 候选。父块回补做的是：

```text
根据 child.parent_id 到 MySQL 查 parent chunk
把 parent_content / parent_title / parent_section_path 等补回候选
```

父块回补不是“重新检索父块”，也不是“拿完整文档去重”。它只是把子块命中的上层上下文补回来，供 reranker 和 evidence packing 使用。

回补后仍然使用 soft cap，而不是 doc 级硬去重：

```python
cap_by_group(backfilled, max_per_doc=5, max_per_parent=2, max_per_chunk=1)
```

这一步是第二道候选控制：

- 如果多个 child 指向同一个 parent，不让它们无限占位。
- 如果同一篇文档多个 parent 都相关，允许保留多个 parent 进入 rerank。
- 控制进入 rerank 的候选规模，避免 reranker 成本过高。

### 4.5 Rerank 排序后再做最终多样性选择

Reranker 会对候选重新打分，然后按 `rerank_score` 降序排序。

最终多样性选择应该放在 rerank 后，因为这时模型已经判断过“哪条 parent / child 更适合当前 query”。

当前不再保留“文档级评估每个 doc 只留 1 条”的配置。

原因是这个配置主要服务于文档级评估指标，会让 top10 看起来更“覆盖不同文档”，但真实 QA 场景里不一定合理。长文档、多章节、多跳问题经常需要同一篇文档里的多个 parent 共同提供证据。

当前 RAG 问答节点和 T2Retrieval 重新评估都统一使用普通 QA 策略：

```python
select_diverse_results(reranked, top_k=top_k, max_per_doc=3, max_per_parent=1)
```

为了避免 cap 太严格导致 topK 填不满，`select_diverse_results` 现在是三段式 fallback：

```text
第一轮：按 max_per_doc / max_per_parent 严格选择
第二轮：如果不足 top_k，放宽 parent cap
第三轮：如果还不足 top_k，再放宽 doc cap
```

这样正常情况下能控制同一文档、同一父块的冗余；极端 query 下也不会因为 diversity 规则过严导致结果数量不足。

评估脚本也显式暴露并记录这两个参数：

```text
--max-per-doc 3
--max-per-parent 1
```

后面重新跑检索指标时，就按这个真实 QA 口径评估，不再为了文档级指标强行每篇文档只保留一条。

注意：这里所谓“文档多样性控制”不是拿完整文档内容去重，而是使用 chunk 元数据里的 `doc_id` 做来源文档约束。系统检索和 rerank 的对象仍然是 child / parent chunk。

### 4.6 Evidence Packing 单独处理

进入问答链路时，`pack_evidence` 还会做一层证据级去重：

```python
select_diverse_results(reranked_docs, top_k=top_k, max_per_doc=3, max_per_parent=1)
```

这层和前面 rerank 后的 diversity selection 目标不完全一样。

RRF / rerank 阶段关注候选质量和多样性；Evidence Packing 阶段关注的是：

- 减少 prompt 冗余
- 控制 token budget
- 保证证据覆盖

所以 packing 阶段不应该走“每 doc 只留 1 条”。普通 QA 更适合每个 parent 最多 1 条、每个 doc 最多 2～3 条。

所以完整链路可以理解为：

```text
Dense/Sparse 原始召回：允许多个 child 命中，保证召回率
RRF 内部聚合：同一个 chunk_id 的多路命中合并加分
RRF 排序后 soft cap：每 doc 3～5 条，每 parent 1～2 条
父块回补：按 child.parent_id 补 parent 内容
父块回补后 soft cap：继续控制候选规模
Rerank：对 parent-enriched candidates 精排
Rerank 后 final diversity selection：按场景控制 doc / parent 数量
Evidence Packing：按 token budget 和 parent/doc 多样性选证据
```

为什么在这些位置做？

- RRF 后 soft cap：避免过早删掉好 chunk，同时控制同一文档/父块过度占位。
- 父块回补后 soft cap：补齐 parent 上下文后，再控制进入 rerank 的候选规模。
- Rerank 后多样性选择：先让 reranker 判断相关性，再按场景控制最终 topK 多样性。

这属于低风险优化：
不改变召回模型，也不改变精排模型，只是把重复候选清理掉，让 top10 更有信息量。

---

## 5. 优化前后对比

本节分两组结果：

1. 第一组是早期“doc 级硬去重”评估结果，用来证明重复文档确实是主要问题。
2. 第二组是当前最新策略的重评结果，也就是 `RRF 后 soft cap + rerank 后 QA diversity selection`。

当前代码已经进一步调整为：

```text
RRF 后 soft cap
Rerank 后按场景做 final diversity selection
```

当前最新评估使用普通 QA 口径：

```text
--max-per-doc 3
--max-per-parent 1
```

也就是允许同一篇长文档最多保留 3 条不同父块证据，但同一个 parent 默认只保留 1 条，避免同一章节重复占位。

### 5.1 早期 doc 级硬去重结果

这组结果来自早期策略：

```text
RRF / Rerank 后每个 doc 只保留 1 条
```

它能显著提升 T2Retrieval 这类文档级评估指标，但对真实 QA 不一定最合理，因为长文档、多章节问题可能需要同一文档里的多个 parent 共同提供证据。

#### RRF 对比

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| `hit@10` | 0.9600 | 0.9667 | +0.0067 |
| `mrr@10` | 0.9301 | 0.9323 | +0.0022 |
| `ndcg@10` | 0.8159 | 0.8947 | +0.0788 |
| `recall@10` | 0.8180 | 0.8975 | +0.0794 |
| 有重复文档的 query 数 | 285 | 0 | -285 |
| recall=1 的 query 数 | 152 | 208 | +56 |

RRF 的提升主要来自去重后 top10 能放进更多不同文档。

#### Rerank 对比

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| `hit@10` | 0.9667 | 0.9767 | +0.0100 |
| `mrr@10` | 0.9483 | 0.9518 | +0.0035 |
| `ndcg@10` | 0.7431 | 0.9242 | +0.1811 |
| `recall@10` | 0.7624 | 0.9234 | +0.1610 |
| 有重复文档的 query 数 | 290 | 0 | -290 |
| recall=1 的 query 数 | 140 | 228 | +88 |

Rerank 提升最明显。
优化前 rerank 容易把同一文档的相似 child chunk 一起排到前面；去重后，top10 覆盖了更多不同文档，所以 `recall@10` 和 `ndcg@10` 都明显提升。

### 5.2 当前 soft cap + QA diversity 重评结果

这组是当前代码策略的重新评估结果：

```text
Dense / Sparse 原始召回
-> RRF 内部按 chunk_id 聚合
-> RRF 后 soft cap：max_per_doc=5, max_per_parent=2
-> Parent backfill / hydration
-> backfill 后继续 soft cap
-> Rerank
-> Rerank 后 final diversity：max_per_doc=3, max_per_parent=1
```

评估命令：

```powershell
cd E:\PythonProject\Veritas-RAG\backend
D:\Software\anaconda3\envs\cook-rag-1\python.exe evaluation\run_t2retrieval_eval.py --kb-id 8 --limit-queries 300 --top-k 10 --dense-top-k 50 --bm25-top-k 50 --rerank-candidates 50 --max-per-doc 3 --max-per-parent 1 --seed 42
```

报告文件：

```text
data/evaluation/reports/t2retrieval_eval_kb8_20260705_110946.json
```

#### 当前最新指标

| 阶段 | `hit@10` | `mrr@10` | `ndcg@10` | `recall@10` |
|---|---:|---:|---:|---:|
| Dense | 0.9700 | 0.9526 | 0.8463 | 0.8432 |
| BM25 / Sparse | 0.9233 | 0.8844 | 0.7245 | 0.7121 |
| RRF | 0.9633 | 0.9305 | 0.8337 | 0.8447 |
| Rerank | 0.9800 | 0.9527 | 0.9121 | 0.9126 |

#### 和早期 doc 级硬去重的差异

| 阶段 | 指标 | doc 级硬去重 | 当前 QA diversity | 变化 |
|---|---|---:|---:|---:|
| RRF | `hit@10` | 0.9667 | 0.9633 | -0.0034 |
| RRF | `mrr@10` | 0.9323 | 0.9305 | -0.0018 |
| RRF | `ndcg@10` | 0.8947 | 0.8337 | -0.0610 |
| RRF | `recall@10` | 0.8975 | 0.8447 | -0.0528 |
| Rerank | `hit@10` | 0.9767 | 0.9800 | +0.0033 |
| Rerank | `mrr@10` | 0.9518 | 0.9527 | +0.0009 |
| Rerank | `ndcg@10` | 0.9242 | 0.9121 | -0.0121 |
| Rerank | `recall@10` | 0.9234 | 0.9126 | -0.0108 |

这个结果符合预期：

- doc 级硬去重对文档级评估最友好，所以 RRF 阶段的 `recall@10 / ndcg@10` 更高。
- 当前 QA diversity 不再强制“每篇文档只留 1 条”，因此文档级覆盖指标会略低一些。
- 但经过 rerank 后，当前策略的 `hit@10 / mrr@10` 反而略高，`recall@10 / ndcg@10` 只小幅低于硬去重。
- 当前策略更贴近真实 QA，因为它允许同一篇长文档的多个不同 parent 进入最终证据候选。

因此，后续默认建议使用当前 QA diversity 口径作为回归评估口径，而不是继续用 `max_per_doc=1` 的文档级硬去重口径。

---

## 6. 当前结论

当前 retrieval 底座的主要问题不是“完全搜不到”。
Dense 本身已经比较强，真正暴露出来的是：

```text
Parent-Child 分块后，同一文档 / 同一父块下的多个 child 很容易同时命中；
如果后处理不控制多样性，融合和精排阶段会被重复候选占位。
```

早期 doc 级硬去重实验说明了问题确实存在：

- RRF 重复文档 query 数从 285 降到 0。
- Rerank 重复文档 query 数从 290 降到 0。
- Rerank `recall@10` 从 0.7624 提升到 0.9234。
- Rerank `ndcg@10` 从 0.7431 提升到 0.9242。

但这个实验口径不应该作为真实 QA 的默认策略，因为 `max_per_doc=1` 会过早限制长文档、多章节问题的证据覆盖。

当前更合理的默认方案是：

```text
RRF 内部继续按 chunk_id 聚合；
RRF / 父块回补后用 soft cap 控制候选冗余；
Rerank 后按 QA 口径做 final diversity selection；
Evidence Packing 再按 token budget 做证据选择。
```

当前 QA diversity 重评结果为：

| 阶段 | `hit@10` | `mrr@10` | `ndcg@10` | `recall@10` |
|---|---:|---:|---:|---:|
| RRF | 0.9633 | 0.9305 | 0.8337 | 0.8447 |
| Rerank | 0.9800 | 0.9527 | 0.9121 | 0.9126 |

和早期 doc 级硬去重相比，当前策略在文档级 `recall@10 / ndcg@10` 上略低，但更贴近真实问答：

- 不再为了评估指标强制每篇文档只保留 1 条。
- 允许同一篇长文档最多保留 3 条不同父块证据。
- 同一 parent 默认只保留 1 条，避免同一章节重复占满上下文。
- 通过 fallback 机制避免 cap 过严导致 topK 填不满。

因此，后续检索层回归评估建议统一使用当前 QA diversity 口径：

```text
--max-per-doc 3
--max-per-parent 1
```

这次优化的核心价值不是单纯把评估指标刷到最高，而是把检索结果从“召回很多相似 chunk”调整为“保留足够相关、足够多样、适合进入问答上下文的证据候选”。

---

## 7. 后续优化方向

后续不用急着增加更复杂的 Agent 节点，优先把 retrieval 的稳定性继续打磨好。

建议按这个顺序做：

1. **保留去重策略**
   当前 doc / parent / chunk 去重收益很明显，应该作为默认策略保留。

2. **增加 evidence 多样性控制**
   比如限制每个 parent 最多进入 1 到 2 条证据，避免一个章节占满上下文。

3. **优化 rerank 输入**
   对比几种输入方式：只给 child、给 parent、给标题路径 + child、给标题路径 + parent + child，看哪种更稳定。

4. **微调 RRF 权重**
   当前 dense 明显强于 sparse，可以测试 sparse 权重 0.3、0.5、0.7、0.9，找到更合适的融合比例。

5. **补真实业务 query 评估**
   T2Retrieval 是公开集，只能说明通用中文检索能力。企业落地还要补真实业务问题，尤其是术语型、表格型、长文档定位型、多跳问题。

6. **持续记录回归指标**
   后面每次改 chunking、embedding、BM25、RRF、rerank，都固定跑同一批 query，看 `hit@10 / recall@10 / mrr@10 / ndcg@10 / 重复率` 是否退化。
