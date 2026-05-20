# RAGAS 答案生成评估使用说明

当前项目原来已有 `mteb/T2Retrieval` 检索阶段评估，现在新增 `backend/evaluation/run_ragas_answer_eval.py`，用于对当前系统的端到端 RAG 答案生成阶段做 RAGAS 评估。

## 1. 评估脚本做什么

脚本会按下面流程执行：

```text
评估样本 question
  -> 调用当前项目 run_agentic_rag
  -> 获取 final_answer / selected_evidence / citations / verification
  -> 转换成 RAGAS 所需字段
  -> 计算 faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness
  -> 输出 JSON 报告
```

它不是绕过系统直接评估文本，而是使用当前项目真实的 LangGraph、混合检索、RRF、reranker、证据打包和答案生成链路。

## 2. 评估样本格式

支持 JSONL 或 JSON。推荐 JSONL，每行一条：

```json
{"id":"recipe_001","question":"红烧肉怎么做才不柴？","ground_truth":"选择带皮五花肉，先焯水去腥，再用小火慢炖到软烂；不要长时间大火煮，否则肉质容易变柴。","gold_doc_ids":["doc_hongshaorou"]}
```

字段说明：

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `id` | 否 | 样本 ID |
| `question` | 是 | 用户问题 |
| `ground_truth` | 建议填 | 标准答案，RAGAS 的 context recall、answer correctness 等指标需要它 |
| `reference` | 否 | `ground_truth` 的别名 |
| `expected_answer` | 否 | `ground_truth` 的别名 |
| `gold_doc_ids` | 否 | 标准证据文档 ID，当前脚本会保存进报告，方便人工排查 |

示例文件已放在：

```text
data/evaluation/ragas_answer_samples.example.jsonl
```

实际项目中建议复制一份自己的样本文件，例如：

```text
data/evaluation/ragas_answer_samples.jsonl
```

## 3. 安装依赖

已在 `backend/requirements.txt` 增加：

```text
ragas>=0.2.14,<0.4.0
datasets>=2.16.0
pandas>=2.0.0
pyarrow>=14.0.0
```

安装：

```bash
cd backend
pip install -r requirements.txt
```

RAGAS 会调用评估 LLM。当前脚本复用项目里的 `graph.llm_factory.get_llm()`，也就是 `.env` / `core.config.py` 中的 OpenAI-compatible 配置。

## 4. 运行评估

先确保你已经有一个可用知识库 `kb_id`，其中已导入业务文档或 T2Retrieval 文档。

运行：

```bash
cd backend
python evaluation/run_ragas_answer_eval.py --input ../data/evaluation/ragas_answer_samples.jsonl --kb-id 你的KB_ID --user-id 1 --top-k 6
```

如果想只收集当前 RAG 生成结果，不调用 RAGAS：

```bash
cd backend
python evaluation/run_ragas_answer_eval.py --input ../data/evaluation/ragas_answer_samples.jsonl --kb-id 你的KB_ID --collect-only
```

如果要启用联网增强：

```bash
cd backend
python evaluation/run_ragas_answer_eval.py --input ../data/evaluation/ragas_answer_samples.jsonl --kb-id 你的KB_ID --web-enabled
```

限制样本数量：

```bash
cd backend
python evaluation/run_ragas_answer_eval.py --input ../data/evaluation/ragas_answer_samples.jsonl --kb-id 你的KB_ID --limit 20
```

指定指标：

```bash
cd backend
python evaluation/run_ragas_answer_eval.py --input ../data/evaluation/ragas_answer_samples.jsonl --kb-id 你的KB_ID --metrics faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness
```

## 5. 输出报告

报告默认输出到：

```text
data/evaluation/reports/ragas_answer_eval_kb{kb_id}_{timestamp}.json
```

报告结构：

```json
{
  "config": {},
  "seconds": 123.45,
  "sample_count": 10,
  "success_count": 10,
  "ragas": {
    "summary": {
      "faithfulness": 0.82,
      "answer_relevancy": 0.79,
      "context_precision": 0.75,
      "context_recall": 0.68,
      "answer_correctness": 0.71
    },
    "rows": []
  },
  "samples": []
}
```

`samples` 会保存每条问题的：

- `question`
- `answer`
- `contexts`
- `reference`
- `citations`
- `selected_evidence`
- `confidence`
- `verification`
- `route_type`
- `evidence_grade`
- `latency_ms`

这些字段用来定位失败原因。比如 RAGAS 的 faithfulness 低，可以直接看 `answer` 是否有证据外扩写；context_recall 低，可以看 `contexts` 是否漏了标准答案所需信息。

## 6. 指标怎么看

| 指标 | 解释 | 低分优先排查 |
| --- | --- | --- |
| `faithfulness` | 答案是否被上下文支持 | 生成 prompt、引用约束、证据不足仍硬答 |
| `answer_relevancy` | 答案是否回答了问题 | query rewrite、问题拆解、生成指令 |
| `context_precision` | 给模型的上下文是否干净 | rerank、证据去重、pack_evidence |
| `context_recall` | 标准答案所需信息是否进入上下文 | 检索召回、chunk、parent backfill |
| `answer_correctness` | 答案与标准答案是否一致 | 检索 + 生成综合问题 |

如果样本没有 `ground_truth/reference/expected_answer`，脚本会跳过需要参考答案的指标，只保留不依赖标准答案的指标。

## 7. 建议的落地方式

先用 20-50 条高质量手写样本跑通流程，再扩大到 200 条左右。每条样本最好包含：

- 一个真实用户会问的问题。
- 一个人工写的标准答案。
- 可选的标准文档 ID 或 chunk ID。
- 问题类型，例如精确菜名、食材组合、步骤技巧、替代方案、证据不足。

上线前建议固定一份 baseline 报告。后续改 chunk、embedding、reranker、prompt、反思阈值时，都用同一批样本重跑，比较 RAGAS 分数、延迟和失败样本。

## 8. 使用专门的生成评估数据集 HotpotQA

如果不想用自建样本，也不要继续拿纯检索数据集硬凑答案评估，可以使用 HotpotQA。HotpotQA 每条样本包含：

- `question`：问题。
- `answer`：标准答案。
- `context`：候选上下文段落。
- `supporting_facts`：支撑答案的证据标题和句子位置。

这类数据更适合评估答案生成阶段，尤其是：

- 多跳问答是否答对。
- 答案是否忠实于证据。
- 检索到的上下文是否足够支持答案。
- 生成阶段是否把多个证据串起来。

项目已新增导入脚本：

```text
backend/evaluation/import_hotpotqa_dataset.py
```

安装/确认依赖后，用 `cook-rag-1` 环境导入少量样本：

```bash
D:\Software\anaconda3\envs\cook-rag-1\python.exe backend/evaluation/import_hotpotqa_dataset.py --limit-examples 20 --user-id 1 --clean-existing
```

脚本会：

- 下载 `hotpotqa/hotpot_qa` 的 `distractor/validation` split。
- 创建知识库 `public_eval_hotpotqa_generation`。
- 把 HotpotQA context 段落导入 MySQL + Chroma。
- 生成 RAGAS 样本文件：

```text
data/evaluation/hotpotqa_ragas_samples.jsonl
```

导入完成后会输出 `kb_id`，例如：

```text
kb_id=6
sample_output=data/evaluation/hotpotqa_ragas_samples.jsonl
```

然后运行 RAGAS：

```bash
D:\Software\anaconda3\envs\cook-rag-1\python.exe backend/evaluation/run_ragas_answer_eval.py --input data/evaluation/hotpotqa_ragas_samples.jsonl --kb-id 6 --user-id 1 --top-k 6
```

注意：RAGAS 和当前 RAG 生成链路都会调用项目配置里的 LLM。当前项目默认指向：

```text
OPENAI_BASE_URL=http://127.0.0.1:8317/v1
OPENAI_MODEL=gpt-5.4
```

如果这个本地 OpenAI-compatible 服务没有启动，报告会生成，但 RAGAS 指标会是 `nan`，日志里会出现 `APIConnectionError: Connection error`。这时需要先启动本地模型服务，或把 `.env` 里的 `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL` 改成可访问的评估模型。
