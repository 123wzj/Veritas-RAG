# Markdown 文档多模态解析与检索增强 — PRD

| 项目 | 内容 |
|---|---|
| 文档 | Markdown 文档多模态解析与检索增强 PRD |
| 版本 | v2.0 |
| 日期 | 2026-08-12 |
| 状态 | 实施中（阶段 1 已完成，阶段 2 进行中） |
| 关联仓库 | Veritas-RAG（本文档相关代码：`backend/`） |

---

## 1. 背景与问题

### 1.1 现状

Markdown 是私域知识库中最常见的文档格式之一，其中大量承载关键信息的**表格、图片、代码块**在当前链路中处理不当：

- **解析层**（`backend/services/ingestion/parser.py`）：Markdown 仅作为纯文本读取，表格结构、代码块语义、图片引用全部丢失。
- **分块层**（`backend/services/ingestion/chunker.py`）：所有 chunk 硬编码 `modality=ModalityType.TEXT`，对全文统一按句子切分——**表格被切碎、代码块被腰斩**。
- **入库层**（`backend/services/ingestion/ingestion.py`）：图片块无 OCR/描述文本，`caption`/`source_uri` 从未写入，图片内容无法被检索。
- **检索层**（`hybrid.py`）：检索结果不透传 `modality`/`caption`/`source_uri`，即使命中图片/表格块，结果中也无法识别其模态。
- **证据/生成层**（`retrieval_nodes.py` / `generation_nodes.py`）：证据项与引用无 `modality`/`image_url`，图片/表格证据不能进入 prompt、不能展示、不能引用。
- **API/前端**：上传响应无模态统计；前端 `Citation` 不渲染图片/表格。

### 1.2 影响

表格（数据、结构化结论）、图片（图表、架构、截图）、代码（算法、配置）承载了文档中大量关键信息，当前方案**无法召回、无法引用、无法展示**，直接限制系统对 MD 类文档的实用价值。

### 1.3 目标

Markdown 端到端多模态支持：**解析 → 分块 → 入库 → 检索 → 证据 → 展示** 全链路正确处理 text / table / code / image 四种模态。

### 1.4 范围界定

**本期（本 PRD）只处理 Markdown 文档**。

- 处理：Markdown（`.md`）+ 现有 TXT 文本（保留现状）。
- **不在本期**：PDF / DOCX / XLSX / HTML / 图片单文件 / PPTX；MinerU 云端解析；扫描件 OCR 增强；外部数据源同步。
- **后续扩展（按需评估引入）**：
  - **MinerU 云端解析**（PDF/DOCX/XLSX/图片）：见 §3.4，付费 API，本期不实施，留作后续扩展方向。
  - **HTML 结构化解析**：与 MD 共用 blocks 机制（见 §3.1），改动小。
  - **图像向量**（qwen3-vl-embedding）：见 §4.8。

---

## 2. 相关代码锚点（现状）

| 模块 | 文件 | 说明 |
|---|---|---|
| 上传入口 | `backend/api/v1/endpoints/knowledge.py` | `POST /{kb_id}/upload`，校验扩展名→保存→调 `ingest_document` |
| 解析 | `backend/services/ingestion/parser.py` | `parse_markdown()` → 结构化 blocks；`_parse_markdown_blocks` 单遍状态机 |
| 分块 | `backend/services/ingestion/chunker.py` | `_create_parent_chunks_from_blocks` 按 type 分流；`create_child_chunks` 非 text 块不按句切 |
| 入库 | `backend/services/ingestion/ingestion.py` | `_describe_image_chunks`（OCR+VLM 描述）；`_save_metadata` 写 modality/caption/source_uri |
| 向量 | `backend/embeddings/embeddings.py`、`bge_m3.py`、`sparse.py` | BGE-M3 本地 dense/sparse |
| 图像能力 | `backend/services/vision/ocr.py` | `QwenVisionService`（OCR+描述）；`MultimodalEmbeddingService.embed_image`（图像向量，未接线） |
| 检索 | `backend/services/retrieval/hybrid.py`、`reranker.py` | 文本 dense+sparse+RRF+精排；**不透传 modality/caption/source_uri** |
| 证据 | `backend/graph/nodes/retrieval_nodes.py` | `pack_evidence` 打包证据；**无 modality/image_url** |
| 生成 | `backend/graph/nodes/generation_nodes.py` | `_build_citations` 构建引用；**无 modality/image_url** |
| 数据模型 | `backend/models/schemas/knowledge.py` | `ModalityType`（TEXT/IMAGE/TABLE/CODE/MIXED） |
| 配置 | `backend/core/config.py` | `VISION_*`（Qwen-VL，云端）；`EMBEDDING_*`（BGE-M3，本地） |

---

## 3. 方案概述

### 3.1 MD 本地结构化解析（blocks 统一结构）

不依赖云端，`parse_markdown()` 本地单遍扫描产出结构化 blocks：

```
# 标题章节
文本...
| 季度 | 收入 |
| Q1   | 100  |
```python
def f(): ...
```
![架构图](assets/arch.png)

→ 扫描产出 ->
[
  {"type": "text",  "content": "...标题章节..."},
  {"type": "table", "content": "| 季度 | ... |"},
  {"type": "code",  "content": "def f(): ...", "language": "python"},
  {"type": "image", "content": "![架构图](...)", "caption": "架构图", "source_uri": "...", "local": true},
]
```

| 源语法 | 识别方式 | 产出块 |
|---|---|---|
| GFM 表格 `\| a \| b \|` | 分隔行回溯收集整表 | `table`（转 Markdown 表格） |
| 代码围栏 ```` ```lang ```` | 围栏区间整块提取 | `code`（保留语言 + 整块） |
| 图片 `![alt](path)` | 正则提取 | `image`（alt 作 caption） |
| 其余 | 按段落累积 | `text` |

图片处理：
- **本地路径**（相对/绝对）→ 复制到 `images/` 子目录，`source_uri` 指向落盘文件。
- **外链 URL**（http/https）→ 仅保留 `alt` + `source_uri`，**不下载**（避免外网依赖与成本）。

### 3.4 MinerU 云端解析（后续扩展方向，本期不实施）

- **思路**：MinerU 云端 Open API（`mineru-open-sdk`）可将 PDF/DOCX/XLSX/图片解析为结构化 `content_list`（text/table/image/equation），与 MD 本地 blocks 机制同构，复用同一套分块/入库/检索链路。
- **现状**：`.env` 已配置 `MINERU_API_KEY`；`backend/core/config.py` **尚无 `MINERU_API_KEY` 字段**（需补充）；SDK 未安装。
- **触发条件**：后续需要支持 PDF/DOCX 等格式时再评估——验证 `MINERU_API_KEY` 有效性、云端费用、内容质量。
- **依赖**：`pip install mineru-open-sdk`；远程调用按量计费。
- **降级**：若 API 不可用或未配置 Key，回退到现有 `DocumentParser`（pypdf 文本流）。

### 3.2 端到端新流程

```
上传 .md 文件
  → parse_markdown → blocks(text/table/code/image)      ✅ 已完成
  → chunker 按 type 分流                                  ✅ 已完成
      text   → Parent-Child 文本切分（modality=TEXT）
      table  → 整表独立块，不按句切（modality=TABLE）
      code   → 整块保留语言（modality=CODE）
      image  → 独立块（modality=IMAGE），存图片 + OCR/描述文本
  → 图片块 OCR+VLM 描述拼接入库（content="[图片] OCR+描述"） ✅ 已完成
  → 落库 MySQL + Chroma（modality/caption/source_uri）      ✅ 已完成
  → 检索结果透传 modality/caption/source_uri                ⬅ 阶段2
  → 证据打包携带 modality/image_url/caption                 ⬅ 阶段2
  → 引用 [E#] 透传多模态字段，前端可展示                     ⬅ 阶段2
```

### 3.3 多模态覆盖矩阵

| 模态 | 是否支持 | 处理方式 |
|---|---|---|
| 文本 | ✅ | Parent-Child 切分，`modality=TEXT` |
| 表格 | ✅ | 转 Markdown 表格整表独立 chunk，`modality=TABLE` |
| 图片 | ✅ | 存图片文件 + OCR/描述文本，`modality=IMAGE` |
| 代码 | ✅ | 围栏整块保留语言，`modality=CODE` |
| 公式 | ⬜ 后续 | MD 中公式较少，暂以代码块/文本处理 |

---

## 4. 详细需求

> 优先级：P0 = 必须（阶段1，已完成）；P1 = 重要（阶段2，当前目标）；P2 = 期望（阶段3，远期）。

### 4.1 MD 结构化解析（P0，✅ 已完成）

- `parse_markdown()` 按 §3.1 产出 blocks；`_parse_markdown_blocks` 单遍状态机处理表格/代码/图片/文本。
- 图片：本地复制到 `images/`，外链仅保留 uri + alt。
- 回归：`tests/test_markdown_multimodal.py` 6 用例覆盖。

### 4.2 按类型分块（P0，✅ 已完成）

- `chunker.py` `_create_parent_chunks_from_blocks` 按 blocks type 分流：text→整体 Parent；table/code/image→独立块。
- `_create_chunk_from_text` 增加 `modality/caption/source_uri` 透传。
- 非 text 块**不按句子切分**（`create_child_chunks`/`_split_large_chunk` 保持块级完整）。

### 4.3 图片语义化落库（P0，✅ 已完成）

- `ingestion.py` `_describe_image_chunks`：图片块通过 **Qwen-VL（`VISION_MODEL`，当前 `qwen3-vl-flash`）** 生成 OCR 文字 + VLM 语义描述，拼接为 `[图片] OCR文字+描述` 作为 chunk 内容。
- 图片块因此可被**文本检索**（BGE-M3 dense + BM25 sparse）召回。
- `_save_metadata` 写入 `modality`/`caption`/`source_uri`；Chroma 元数据带 `modality`/`source_uri`/`caption`。
- `metadata` 中 `[图片]` 前缀为阶段3 图像向量回填预留钩子。

### 4.4 检索结果多模态字段透传（P1，阶段2 进行中）

**改造 `hybrid.py`**

- `_format_chroma_query_results`（dense）：补 `modality`/`caption`/`source_uri`。
- `_load_sparse_corpus_sync`（sparse）：补 `modality`/`caption`/`source_uri`。
- `_hydrate_child_results_sync`：child 为 IMAGE/TABLE/CODE 时透传相应字段。
- `_parent_backfill_sync`：parent 回补时透传 parent 的 modality/caption/source_uri。
- 检索结果自始至终携带模态信息，供上游证据/引用使用。

> 图片块内容已是文本（OCR+描述），天然参与 dense/sparse 检索，无需额外检索路径。

### 4.5 多模态证据与引用（P1，阶段2 进行中）

**改造 `retrieval_nodes.py` / `generation_nodes.py`**

- `pack_evidence` 证据项新增 `modality`、`image_url`（图片块 `source_uri`）、`caption`。
- `_build_citations` 引用透传 `modality`/`image_url`/`caption`，供前端展示。
- 图片证据以 `caption`/OCR 文本进入 prompt（DeepSeek 为文本模型，不做多模态输入）。

### 4.6 API 响应扩展（P1，阶段2 进行中）

**改造 `api/v1/endpoints/knowledge.py`**

- `POST /{kb_id}/upload` 响应新增 `modality`（文档级）与各模态块统计（text/table/code/image 数量）。
- `GET /{kb_id}/documents/{doc_id}` 返回文档级 `modality` 与各模态块数量。

### 4.7 前端展示（P1，阶段2 进行中）

**改造 `frontend/`**

- `types/index.ts`：`ModalityType` 增加 `"code"`；`Citation` 增加 `modality`/`image_url`/`caption`。
- `chat-message.tsx`：证据卡片按 `modality` 渲染——图片显示缩略图（`image_url`）、表格渲染 Markdown、代码显示代码块。
- `knowledge.tsx`：文档列表展示模态类型标签。

### 4.8 图像向量（P2，阶段3 远期）

- `MultimodalEmbeddingService.embed_image`（qwen3-vl-embedding）已实现端点修复，但**未接线**。
- 价值：图片直接向量化，弥补"VLM 描述文本未覆盖的精确数据级查询"盲区。
- 决策：**云端付费**，待阶段2 闭环跑通后评估是否需要。

---

## 5. 非功能需求

| 项 | 要求 |
|---|---|
| 性能 | MD 本地解析无云端调用，入库延迟主要来自图片 OCR/VLM（云端，按图计费）；进度回调沿用 `task_callback`。 |
| 可用性 | Qwen-VL 不可用时图片块降级为 `[图片]` + OCR 失败空文本，不阻断入库。 |
| 成本 | 仅图片语义化消耗云端 Qwen-VL 配额（`qwen3-vl-flash`，输入 ¥0.15/百万token）；文本/表格/代码零成本。 |
| 安全 | `VISION_API_KEY` 不落日志；上传文件路径沿用 `_safe_upload_path` 校验。 |
| 兼容 | 不破坏现有文本检索、证据判断、生成、记忆链路；旧文档数据无需迁移。 |

---

## 6. 验收标准

1. 上传含表格/图片/代码的 Markdown，入库后：
   - 文档级 `modality` 为 `mixed`（含图片/表格时）。
   - `chunks` 表存在 `modality=IMAGE`/`TABLE`/`CODE` 块，`caption`/`source_uri` 有值。
   - 图片块 content 为 `[图片] OCR+描述`，本地图片已复制到 `images/`。
2. **检索命中**：查询"Q2 收入"等含图片/表格内容的查询，检索结果能召回图片/表格块，且结果带 `modality` 与 `source_uri`。
3. **证据携带**：证据项带 `modality`/`image_url`/`caption`；引用 `[E#]` 透传多模态字段。
4. **前端展示**：图片证据显示缩略图，表格证据渲染 Markdown。
5. **上传响应**：返回文档级 `modality` 与各模态块统计。
6. 表格块不被句子切碎；代码块不被句子切碎；外链图片仅存 `source_uri` + alt，不下载。
7. 现有测试不回归（`tests/` 全部通过）。

---

## 7. 里程碑与任务拆解

| 阶段 | 交付物 | 状态 |
|---|---|---|
| **阶段 1（P0）** | ① `parse_markdown` 结构化 blocks ② `chunker` 按 type 分流 ③ 图片 OCR+VLM 描述落库 ④ `ModalityType.CODE` ⑤ 回归测试 | ✅ 已完成（44 测试通过） |
| **阶段 2（P1）** | ⑥ `hybrid.py` 检索透传 modality/caption/source_uri ⑦ `pack_evidence`/`_build_citations` 多模态证据 ⑧ 上传/文档 API 模态统计 ⑨ 前端证据展示 | ⬅ 当前目标 |
| **阶段 3（P2）** | ⑩ 图像向量（qwen3-vl-embedding）接线 ⑪ 生成模型多模态输入 | 远期，按需 |

---

## 8. 风险与待确认

| 项 | 说明 | 决策 |
|---|---|---|
| 云端费用 | 图片语义化消耗 Qwen-VL 配额 | 已切 `qwen3-vl-flash`（输入 ¥0.15/百万token），成本可控；图片少时几乎无感 |
| 图片召回盲区 | 图片靠描述文本召回，精确数据级查询可能漏网 | §4.8 图像向量阶段3 补齐；`[图片]` 前缀 + `source_uri` 钩子已预留 |
| 外链图片 | 本期不下载，仅存 uri + alt | 避免外网依赖与成本 |
| 范围收敛 | 本期聚焦 MD；PDF/DOCX/HTML/MinerU 不在本期 | 已排除，避免范围蔓延；HTML 复用 blocks 机制，后续改动小 |
| **MinerU 接入评估** | MinerU 云端解析（PDF/DOCX/XLSX/图片）为付费 API | 本期不实施；**留作后续扩展方向**，待验证 `MINERU_API_KEY` 有效性、成本、内容质量后再决定是否接入（见 §3.4） |
| 生成模型 | DeepSeek 为文本模型，图片证据靠 VLM 描述文本 | 阶段3 引入多模态模型或保留文本表示 |

---

## 9. 附录：图片语义化实现说明（Qwen-VL，已实测）

### 9.1 方案

图片块入库时，用 Qwen-VL 视觉模型生成两段文本，拼接到 chunk 内容：

```
[图片]
<OCR 文字：识别图片中的全部文字>
<VLM 语义描述：图片整体语义、场景、数据含义、检索关键词建议>
```

- OCR（`extract_text`）：识别图片文字，表格自动转 Markdown。
- VLM（`describe_image`）：描述图片语义，便于按主题/关键词检索到图片。

### 9.2 模型与成本（2026-08 实测）

| 模型 | 输入 ¥/百万token | 输出 ¥/百万token | 实测 |
|---|---|---|---|
| ~~qwen-vl-max~~（原默认） | 3 | 9 | — 已弃用（贵） |
| **qwen3-vl-flash（当前）** | **0.15** | **1.5** | OCR + 描述均准确 ✅ |

- 单张图 OCR+描述约消耗数百 token，成本约 0.001 元级。
- 配置：`.env` `VISION_MODEL=qwen3-vl-flash`、`VISION_PROVIDER=qwen`。

### 9.3 相关文件

- `backend/services/vision/ocr.py`：`QwenVisionService`（`_resolve_url` 走 `api/v1/services/aigc/multimodal-generation/generation`）。
- `backend/services/ingestion/ingestion.py`：`_describe_image_chunks`。