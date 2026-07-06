# RAGAS 答案生成评估说明

当前项目原来已有 `mteb/T2Retrieval` 检索阶段评估，现在新增 `backend/evaluation/run_ragas_answer_eval.py`，用于对当前系统的端到端 RAG 答案生成阶段做 RAGAS 评估。

简单说：

```text
T2Retrieval 看“证据找没找对”
RAGAS 看“最后答案答得靠不靠谱”
```

这两个评估不要混在一起看。检索好，不代表答案一定好；答案差，也不一定都是检索的问题。



## 1. 评测数据需要哪些字段？

最常见的一行评测数据包含：

| 字段                                       | 含义                                 |
| ------------------------------------------ | ------------------------------------ |
| `question` / `user_input`                  | 用户问题                             |
| `reference` / `ground_truth`               | 标准答案                             |
| `retrieved_contexts` / `contexts`          | RAG 实际检索出来的 chunk             |
| `response` / `answer`                      | LLM 最终回答                         |
| `reference_contexts` / `reference_doc_ids` | 可选，人工标注的正确 chunk 或文档 ID |

Ragas 的 `evaluate()` 接收 dataset 和 metrics，官方示例中 dataset 可包含 `question`、`ground_truth`、`answer`、`contexts`，然后输出如 `context_precision`、`faithfulness`、`answer_relevancy` 等结果。



## 指标：

| 指标                                      | 主要评估什么         | 含义                                                  | 低分说明                                                     |
| ----------------------------------------- | -------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| **Context Precision**                     | 检索排序质量         | 检索出来的 chunk 里，相关内容是否占比高、是否排在前面 | 检索噪声大、top-k 里无关 chunk 太多、需要 rerank             |
| **Context Recall**                        | 检索覆盖率           | 标准答案所需的信息，有多少被检索出来了                | 知识库缺内容、chunk 切分差、召回 top-k 太小、embedding 不合适 |
| **Faithfulness**                          | 回答是否忠实于上下文 | 回答里的事实 claim 是否都能被 retrieved contexts 支持 | 模型幻觉、回答编造、prompt 约束不够                          |
| **Response Relevancy / Answer Relevancy** | 回答是否对题         | 回答是否直接回应用户问题，不看事实对错                | 答非所问、回答不完整、废话多、格式不符合需求                 |

RAGAS 官方当前把 RAG 场景的指标列为：Context Precision、Context Recall、Context Entities Recall、Noise Sensitivity、Response Relevancy、Faithfulness，以及多模态相关指标。



### 1. Context Precision：检索得准不准

它看的是：**检索出来的内容里，相关 chunk 的比例和排序是否好**。官方定义里，Context Precision 衡量 `retrieved_contexts` 中相关 chunk 的比例，并按 precision@k 计算，越高越好。

例子：

```
问题：企业版是否支持 SSO？
Top-3 检索：
1. 企业版支持 SAML SSO     ✅
2. 企业版价格说明           ✅/部分相关
3. 用户头像上传规则         ❌
```

这个指标回答的是：**检索结果里有没有太多垃圾内容？正确内容是不是排在前面？**

### 2. Context Recall：该找的有没有找全

它看的是：**标准答案需要的关键信息，有多少被检索到了**。官方说明 Context Recall 关注“不漏掉重要结果”，通常需要 `reference` 或 `reference_contexts` 来对照；其思想是看 reference 中的 claims 有多少能被 retrieved context 支持。

例子：

```
标准答案需要：
- 支持 SAML SSO
- 只在企业版开放
- 需要管理员配置

检索结果只找到了：
- 支持 SAML SSO
```

这种情况下 Context Recall 就不高，因为重要信息没找全。

### 3. Faithfulness：回答有没有基于资料

它看的是：**模型最终回答是否忠实于检索到的上下文**。官方定义是：如果回答中的所有 claim 都能被 retrieved context 支持，就认为回答 faithful；分数是“被上下文支持的回答 claim 数 / 回答总 claim 数”。

例子：

```
检索内容：退款通常 3-7 个工作日到账。
回答：退款通常 3-7 个工作日到账。 ✅ Faithfulness 高

回答：退款通常 1 个工作日到账。 ❌ Faithfulness 低
```

这个指标主要用来抓 **幻觉**。
 注意：Faithfulness 高不一定代表答案完全正确，它只代表“回答是否被检索上下文支持”。如果检索到的上下文本身是错的，Faithfulness 仍可能高。

### 4. Response Relevancy / Answer Relevancy：有没有答到问题

它看的是：**回答和用户问题是否相关**，不评估事实正确性。官方说明 Answer Relevancy 衡量 response 与 user input 的相关程度，惩罚不完整或包含多余信息的回答，但不判断 factual accuracy。

例子：

```
问题：如何申请退款？

回答 A：进入订单页，点击申请退款。 ✅ 相关
回答 B：我们的会员体系分为普通版和企业版。 ❌ 不相关
回答 C：退款可以申请，另外我们公司成立于 2018 年…… ⚠️ 有冗余
```

### 5. Factual Correctness：和标准答案比，事实对不对

如果你有 `reference` / `ground_truth`，这个指标很有用。官方定义里，Factual Correctness 会比较 generated `response` 和 `reference` 的事实一致性，通常会把两边拆成 claims，再用 precision、recall、F1 衡量事实重合程度。

它适合回答：

```
生成答案和标准答案相比，对了多少？
有没有漏掉标准答案里的关键点？
有没有多说错误事实？
```

如果你是做学校项目、企业知识库、客服 QA，我建议把它加入评估，因为它比单纯看 Faithfulness 更接近“最终答案对不对”。

### 6. Context Entities Recall：关键实体有没有召回

这个指标看的是：**reference 中的实体，有多少出现在 retrieved contexts 里**。官方说明它适合事实型、实体密集型场景，比如旅游问答、历史 QA 等。

适合这些知识库：

```
产品型号
人名
地名
合同编号
药品名称
政策条款
公司名
时间日期
```

如果你的问题经常依赖实体，比如“某产品 A-203 的保修期是多少？”，这个指标很有价值。

### 7. Noise Sensitivity：抗干扰能力

它看的是：**当检索结果里混入相关或不相关文档时，系统是否容易答错**。官方说明 Noise Sensitivity 衡量系统在使用相关或无关检索文档时产生错误回答的频率，分数 0 到 1，越低越好。

这个指标适合检查：

```
检索结果里有噪声时，模型会不会被带偏？
多个 chunk 内容冲突时，模型会不会乱答？
上下文里夹杂无关内容时，模型是否还能回答正确？
```
