# -*- coding: utf-8 -*-
"""
文档分块模块
实现 Parent-Child Chunk 策略
1. 什么是 Parent-Child Chunk？

  文档
  ├── Parent Chunk 1 (约1000 tokens)
  │   ├── Child Chunk 1.1 (约300 tokens)
  │   ├── Child Chunk 1.2 (约300 tokens)
  │   └── Child Chunk 1.3 (约300 tokens)
  ├── Parent Chunk 2 (约1000 tokens)
  │   ├── Child Chunk 2.1 (约300 tokens)
  │   └── Child Chunk 2.2 (约300 tokens)
  ...

  - Parent Chunk：较大的块，保留完整的上下文信息
  - Child Chunk：从 Parent 进一步切分的小块，记录 parent_id

  ---
  2. 为什么要使用父子块？

  ┌────────────┬─────────────────────┬────────────────────────────────┐
  │    维度     │    单层分块问题       │    Parent-Child 解决方案        │
  ├────────────┼─────────────────────┼────────────────────────────────┤
  │ 召回精度    │ 大块匹配模糊           │ 小块提高语义匹配精度              │
  ├────────────┼─────────────────────┼────────────────────────────────┤
  │ 上下文完整   │ 小块信息割裂          │ 返回时带上 Parent 完整内容        │
  ├────────────┼─────────────────────┼────────────────────────────────┤
  │ token 成本  │ 大块 embedding 昂贵  │ 只对小块做 embedding，节省成本     │
  └────────────┴─────────────────────┴────────────────────────────────┘

  检索流程：
  1. 查询向量与 Child Chunk 向量匹配（粒度细，匹配准）
  2. 返回时带上对应的 Parent Chunk 内容（上下文完整）

  ---
  3. 分块是如何切割的？

  不同文件类型的 Parent Chunk 切割策略：

  ┌──────────┬──────────────────────────────────────────────────┐
  │ 文件类型 │                     切割方式                     │
  ├──────────┼──────────────────────────────────────────────────┤
  │ PDF      │ 按页切分 (pages)                                 │
  ├──────────┼──────────────────────────────────────────────────┤
  │ DOCX     │ 按段落合并，达到 parent_chunk_size (1000) 时切分 │
  ├──────────┼──────────────────────────────────────────────────┤
  │ PPTX     │ 按幻灯片切分                                     │
  ├──────────┼──────────────────────────────────────────────────┤
  │ TXT      │ 按空行分段，合并小段                             │
  ├──────────┼──────────────────────────────────────────────────┤
  │ Markdown │ 按章节 (sections) 切分                           │
  └──────────┴──────────────────────────────────────────────────┘

  Child Chunk 切割 (第244-299行)：

  1. 将 Parent Chunk 按句子分割（split_by_sentences）
  2. 逐句累积，超过 child_chunk_size (300) 时切分
  3. 加入 overlap（重叠内容）避免边界信息丢失

  # overlap 逻辑 (第400-412行)
  if language == "zh":
      # 中文：取末尾约 1/3 的字符
      overlap_chars = min(child_chunk_overlap, len(current_content) // 3)
  else:
      # 英文：取末尾约 50 个单词
      overlap_words = min(child_chunk_overlap // 4, 8)

  Token 估算 (第44-49行)：

  # 简单估算：中文字符=1token，英文单词=1token
  chinese_chars + english_words

  ---
  关键参数

  ┌──────────────────────┬────────┬────────────────────────┐
  │         参数         │ 默认值 │          说明          │
  ├──────────────────────┼────────┼────────────────────────┤
  │ parent_chunk_size    │ 1000   │ 父块目标大小           │
  ├──────────────────────┼────────┼────────────────────────┤
  │ child_chunk_size     │ 300    │ 子块目标大小           │
  ├──────────────────────┼────────┼────────────────────────┤
  │ parent_chunk_overlap │ 100    │ 父块重叠（目前未启用） │
  ├──────────────────────┼────────┼────────────────────────┤
  │ child_chunk_overlap  │ 50     │ 子块重叠               │
  └──────────────────────┴────────┴────────────────────────┘
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.models.schemas.knowledge import ModalityType

# 代码块拆分阈值（tokens）：超过该值的代码块按逻辑边界拆分，否则保持块级完整
code_split_threshold = 600

# 表格拆分阈值（tokens）：超过该值的表格按数据行分组拆分，否则保持块级完整
table_split_threshold = 600


@dataclass
class Chunk:
    """数据块"""
    chunk_id: str
    parent_id: Optional[str]
    doc_id: str
    content: str
    modality: ModalityType
    language: str
    title: Optional[str]
    section_path: Optional[str]
    page_no: Optional[int]
    token_count: int
    metadata: Dict[str, Any]
    code_language: Optional[str] = None


class DocumentChunker:
    """文档分块器"""
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50,
        code_split_threshold: int = code_split_threshold,
        table_split_threshold: int = table_split_threshold,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.code_split_threshold = code_split_threshold
        self.table_split_threshold = table_split_threshold

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量（中文约等于字符数，英文约等于词数）"""
        # 简单估算：中文字符 = 1 token，英文单词 = 1 token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        return chinese_chars + english_words

    def detect_language(self, text: str) -> str:
        """轻量语言检测，避免父子块全部被标成中文。"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text or ""))
        english_chars = len(re.findall(r'[a-zA-Z]', text or ""))
        if english_chars > chinese_chars * 2:
            return "en"
        return "zh"

    def split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 中英文句子分割
        sentence_ends = r'([。！？.!?]+)'
        sentences = re.split(sentence_ends, text)

        result = []

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        if len(sentences) % 2 == 1:
            trailing_text = sentences[-1].strip()
            if trailing_text:
                result.append(trailing_text)

        return result

    def create_parent_chunks(
        self,
        parsed_content: Dict[str, Any],
        doc_id: str,
    ) -> List[Chunk]:
        """
        创建 Parent Chunks

        根据 800-1500 tokens 的目标大小，按标题、段落、语义边界切分
        """
        file_type = parsed_content.get("type")
        parent_chunks = []
        chunk_index = 0

        if file_type == "pdf":
            # PDF 按页切分
            for page in parsed_content.get("pages", []):
                content = page["content"]
                page_no = page["page"]
                self._create_chunk_from_text(
                    content,
                    doc_id,
                    None,
                    parsed_content.get("metadata", {}).get("title"),
                    f"Page {page_no}",
                    page_no,
                    parent_chunks,
                    is_parent=True,
                )

        elif file_type == "docx":
            # 按段落切分
            paragraphs = parsed_content.get("paragraphs", [])
            current_content = ""
            current_section = None

            for para in paragraphs:
                content = para.get("content", "")
                section = para.get("section") or current_section
                if section:
                    current_section = section

                # 合并小段落
                if self.estimate_tokens(current_content + content) < self.parent_chunk_size:
                    current_content += "\n" + content
                else:
                    # 保存当前 chunk
                    if current_content.strip():
                        self._create_chunk_from_text(
                            current_content.strip(),
                            doc_id,
                            None,
                            parsed_content.get("metadata", {}).get("title"),
                            current_section,
                            None,
                            parent_chunks,
                            is_parent=True,
                        )
                    current_content = content

            # 保存最后的 chunk
            if current_content.strip():
                self._create_chunk_from_text(
                    current_content.strip(),
                    doc_id,
                    None,
                    parsed_content.get("metadata", {}).get("title"),
                    current_section,
                    None,
                    parent_chunks,
                    is_parent=True,
                )

        elif file_type == "pptx":
            slides = parsed_content.get("slides", [])
            for slide in slides:
                content = slide.get("content", "").strip()
                if not content:
                    continue
                self._create_chunk_from_text(
                    content,
                    doc_id,
                    None,
                    parsed_content.get("metadata", {}).get("title"),
                    f"Slide {slide.get('slide')}",
                    slide.get("slide"),
                    parent_chunks,
                    is_parent=True,
                )

        elif file_type == "txt":
            content = parsed_content.get("content", "")
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
            current_content = ""
            for para in paragraphs:
                merged = f"{current_content}\n{para}".strip() if current_content else para
                if self.estimate_tokens(merged) <= self.parent_chunk_size:
                    current_content = merged
                    continue

                if current_content:
                    self._create_chunk_from_text(
                        current_content,
                        doc_id,
                        None,
                        parsed_content.get("metadata", {}).get("title"),
                        None,
                        None,
                        parent_chunks,
                        is_parent=True,
                    )
                current_content = para

            if current_content:
                self._create_chunk_from_text(
                    current_content,
                    doc_id,
                    None,
                    parsed_content.get("metadata", {}).get("title"),
                    None,
                    None,
                    parent_chunks,
                    is_parent=True,
                )

        elif file_type == "markdown":
            # Markdown 按 blocks 分流（来自 parse_markdown 的结构化输出）
            blocks = parsed_content.get("blocks")
            if blocks:
                self._create_parent_chunks_from_blocks(
                    blocks,
                    doc_id,
                    parsed_content,
                    parent_chunks,
                )
            else:
                # 兼容旧结构：按章节切分
                sections = parsed_content.get("sections", [])
                for section in sections:
                    title = section.get("title", "未分类")
                    content = section.get("content", "")
                    self._create_chunk_from_text(
                        content,
                        doc_id,
                        None,
                        parsed_content.get("metadata", {}).get("title"),
                        title,
                        None,
                        parent_chunks,
                        is_parent=True,
                    )

        else:
            # 其他类型，整体作为一块
            content = parsed_content.get("content", "")
            self._create_chunk_from_text(
                content,
                doc_id,
                None,
                parsed_content.get("metadata", {}).get("title"),
                None,
                None,
                parent_chunks,
                is_parent=True,
            )

        # 如果 parent chunk 太大，进一步切分
        final_parents = []
        for chunk in parent_chunks:
            # 代码块按 code_split_threshold、表格按 table_split_threshold 触发拆分，
            # 其他块按 parent 1.5x
            split_trigger = (
                self.code_split_threshold
                if chunk.modality == ModalityType.CODE
                else self.table_split_threshold
                if chunk.modality == ModalityType.TABLE
                else self.parent_chunk_size * 1.5
            )
            if chunk.token_count > split_trigger:
                # 需要进一步切分
                sub_chunks = self._split_large_chunk(chunk)
                final_parents.extend(sub_chunks)
            else:
                final_parents.append(chunk)

        return final_parents

    def _create_parent_chunks_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        doc_id: str,
        parsed_content: Dict[str, Any],
        parent_chunks: List[Chunk],
    ) -> None:
        """按 blocks 的 type 分流创建 Parent Chunks。

        text → 文本走整体 Parent（后续 create_child_chunks 再按句切）；
        table/code/image → 作为独立块，不按句切分（保证结构完整）。
        """
        doc_title = parsed_content.get("metadata", {}).get("title")
        # 按相邻文本块累积为一个 Parent，避免过度切碎
        current_text: List[str] = []

        def _flush_text() -> None:
            if current_text:
                joined = "\n".join(current_text).strip()
                if joined:
                    self._create_chunk_from_text(
                        joined,
                        doc_id,
                        None,
                        doc_title,
                        None,
                        None,
                        parent_chunks,
                        is_parent=True,
                        modality=ModalityType.TEXT,
                    )
                current_text.clear()

        for block in blocks:
            btype = block.get("type")
            content = block.get("content") or ""
            if not content.strip():
                continue

            if btype == "text":
                current_text.append(content)
            elif btype in ("table", "code", "image"):
                # 遇到独立块时先收拢累积的文本
                _flush_text()
                modality = {
                    "table": ModalityType.TABLE,
                    "code": ModalityType.CODE,
                    "image": ModalityType.IMAGE,
                }[btype]
                self._create_chunk_from_text(
                    content,
                    doc_id,
                    None,
                    doc_title,
                    None,
                    None,
                    parent_chunks,
                    is_parent=True,
                    modality=modality,
                    caption=block.get("caption"),
                    source_uri=block.get("source_uri"),
                    code_language=block.get("language") if btype == "code" else None,
                )
            else:
                # 未知类型安全回退为文本
                current_text.append(content)

        _flush_text()

    def create_child_chunks(
        self,
        parent_chunks: List[Chunk],
        doc_id: str,
    ) -> List[Chunk]:
        """
        创建 Child Chunks

        从 parent chunk 再细切，每个 child 记录 parent_id
        """
        child_chunks = []

        for parent in parent_chunks:
            # 非 TEXT 块（表格/图片）保持块级完整，不按句切分，原样作为单个 child
            if parent.modality == ModalityType.CODE:
                # 代码块：按 child_chunk_size 切成多个 child（大代码块不原样透传）。
                # _split_code_child_chunks 内部经 _create_chunk_from_text 追加到 child_chunks，
                # 此处不再重复 extend，避免子块被写入两次。
                self._split_code_child_chunks(parent, doc_id, child_chunks)
                continue
            if parent.modality == ModalityType.TABLE:
                # 表格：按 child_chunk_size 切多个 child（大表格不原样透传），表头内嵌
                self._split_table_child_chunks(parent, doc_id, child_chunks)
                continue
            if parent.modality != ModalityType.TEXT:
                child = self._create_chunk_from_text(
                    parent.content,
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                    modality=parent.modality,
                    caption=parent.metadata.get("caption"),
                    source_uri=parent.metadata.get("source_uri"),
                )
                continue

            # 将 parent chunk 按 child_chunk_size 切分
            sentences = self.split_by_sentences(parent.content)
            current_content = ""
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = self.estimate_tokens(sentence)

                if current_tokens + sentence_tokens > self.child_chunk_size and current_content:
                    # 保存当前 child chunk
                    self._create_chunk_from_text(
                        current_content.strip(),
                        doc_id,
                        parent.chunk_id,
                        parent.title,
                        parent.section_path,
                        parent.page_no,
                        child_chunks,
                        is_parent=False,
                    )

                    # 处理 overlap
                    overlap_content = self._get_overlap_content(current_content, sentences, sentence)
                    current_content = overlap_content + sentence
                    current_tokens = self.estimate_tokens(current_content)
                else:
                    current_content += sentence if current_content else sentence
                    current_tokens += sentence_tokens

            # 保存最后的 child chunk
            if current_content.strip():
                self._create_chunk_from_text(
                    current_content.strip(),
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                )

        return child_chunks

    def _create_chunk_from_text(
        self,
        text: str,
        doc_id: str,
        parent_id: Optional[str],
        title: Optional[str],
        section_path: Optional[str],
        page_no: Optional[int],
        chunks_list: List[Chunk],
        is_parent: bool,
        modality: ModalityType = ModalityType.TEXT,
        *,  # 以下为多模态块可选元数据
        caption: Optional[str] = None,
        source_uri: Optional[str] = None,
        code_language: Optional[str] = None,
    ) -> Chunk:
        """从文本创建 chunk"""
        chunk_id = f"{doc_id}_{'parent' if is_parent else 'child'}_{len(chunks_list)}"
        token_count = self.estimate_tokens(text)
        language = self.detect_language(text)

        chunk = Chunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            doc_id=doc_id,
            content=text,
            modality=modality,
            language=language,
            title=title,
            section_path=section_path,
            page_no=page_no,
            token_count=token_count,
            code_language=code_language,
            metadata={
                "is_parent": is_parent,
                "created_at": datetime.now().isoformat(),
                "caption": caption or "",
                "source_uri": source_uri or "",
                "code_language": code_language or "",
            },
        )

        chunks_list.append(chunk)
        return chunk

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """切分过大的 chunk。

        非 TEXT 块（表格/图片）保持块级完整，按句切分会破坏结构；
        超阈值代码块按逻辑边界（空行/函数/硬切）拆成多个 Parent；
        仅对 TEXT 块按句二次切分。
        """
        if chunk.modality == ModalityType.CODE:
            # 低于阈值不拆，保持块级完整（调用方与直接调用统一行为）
            if chunk.token_count <= self.code_split_threshold:
                return [chunk]
            return self._split_code_chunk(chunk)
        if chunk.modality == ModalityType.TABLE:
            # 低于阈值不拆，保持块级完整（调用方与直接调用统一行为）
            if chunk.token_count <= self.table_split_threshold:
                return [chunk]
            return self._split_table_chunk(chunk)
        if chunk.modality != ModalityType.TEXT:
            return [chunk]

        sentences = self.split_by_sentences(chunk.content)
        sub_chunks = []
        current_content = ""
        current_tokens = 0
        sub_index = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.parent_chunk_size and current_content:
                # 保存子 chunk
                new_chunk = Chunk(
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                    parent_id=chunk.parent_id,
                    doc_id=chunk.doc_id,
                    content=current_content.strip(),
                    modality=chunk.modality,
                    language=chunk.language,
                    title=chunk.title,
                    section_path=chunk.section_path,
                    page_no=chunk.page_no,
                    token_count=current_tokens,
                    metadata={
                        **chunk.metadata,
                        "is_parent": chunk.metadata.get("is_parent", False),
                        "split_from": chunk.chunk_id,
                    },
                )
                sub_chunks.append(new_chunk)
                sub_index += 1

                # 重置
                current_content = sentence
                current_tokens = sentence_tokens
            else:
                current_content += sentence if current_content else sentence
                current_tokens += sentence_tokens

        # 保存最后的子 chunk
        if current_content.strip():
            new_chunk = Chunk(
                chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                parent_id=chunk.parent_id,
                doc_id=chunk.doc_id,
                content=current_content.strip(),
                modality=chunk.modality,
                language=chunk.language,
                title=chunk.title,
                section_path=chunk.section_path,
                page_no=chunk.page_no,
                token_count=current_tokens,
                metadata={
                    **chunk.metadata,
                    "is_parent": chunk.metadata.get("is_parent", False),
                    "split_from": chunk.chunk_id,
                },
            )
            sub_chunks.append(new_chunk)

        return sub_chunks

    def _get_overlap_content(self, current_content: str, sentences: List[str], next_sentence: str) -> str:
        """获取 overlap 内容"""
        if not current_content:
            return ""

        language = self.detect_language(current_content)
        if language == "zh":
            overlap_chars = min(self.child_chunk_overlap, max(0, len(current_content) // 3))
            return current_content[-overlap_chars:] if overlap_chars > 0 else ""

        words = current_content.split()
        overlap_words = min(max(self.child_chunk_overlap // 4, 8), len(words))
        return " ".join(words[-overlap_words:]) if overlap_words > 0 else ""

    def _split_code_chunk(self, chunk: Chunk) -> List[Chunk]:
        """将超大代码块按逻辑边界（空行 → 函数 → 行号）拆成多个 Parent。

        每个子 Parent 保留原块的元数据（code_language/source_uri/caption 等），
        chunk_id 后缀 `_part_N`，原代码块作为根（parent_id 指向原块）。
        仅当块超过 code_split_threshold 时由调用方触发。
        """
        content = chunk.content
        # 1) 按空行切成逻辑段
        segments = [s.strip() for s in re.split(r"\n\s*\n", content) if s.strip()]
        if not segments:
            return [chunk]

        # 2) 把相邻小段累积到 parent_chunk_size 再切，避免拆出过多碎块
        merged: List[str] = []
        buffer = ""
        for seg in segments:
            candidate = f"{buffer}\n\n{seg}" if buffer else seg
            if self.estimate_tokens(candidate) <= self.parent_chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = seg
        if buffer:
            merged.append(buffer)

        # 3) 单个大段仍超 1.5x → 降级为按函数/类边界拆分
        threshold = self.parent_chunk_size * 1.5
        final: List[str] = []
        for part in merged:
            if self.estimate_tokens(part) <= threshold:
                final.append(part)
                continue
            final.extend(self._split_code_by_function(part, threshold))

        return [self._make_code_part(chunk, part, idx) for idx, part in enumerate(final)]

    def _split_code_by_function(self, part: str, threshold: float) -> List[str]:
        """按函数/类定义行（def/class）切分单个超大代码段；仍超则按行硬切。"""
        lines = part.split("\n")
        pieces: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for line in lines:
            # 函数/类定义行作为新段的起点（保留上一段）
            if re.match(r"^\s*(async\s+def|def|class)\s+\w+", line) and current:
                pieces.append("\n".join(current))
                current = []
                current_tokens = 0
            current.append(line)
            current_tokens += self.estimate_tokens(line)
            if current_tokens > threshold:
                pieces.append("\n".join(current))
                current = []
                current_tokens = 0
        if current:
            pieces.append("\n".join(current))

        # 兜底：仍超阈值（无 def/class，或单函数极长）→ 按行硬切到阈值
        result: List[str] = []
        for piece in pieces:
            if self.estimate_tokens(piece) <= threshold:
                result.append(piece)
            else:
                result.extend(self._split_code_hard(piece, threshold))
        return result

    def _split_code_hard(self, text: str, threshold: float) -> List[str]:
        """按行硬切代码段，每段不超过 threshold。"""
        lines = text.split("\n")
        pieces: List[str] = []
        buffer: List[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > threshold and buffer:
                pieces.append("\n".join(buffer))
                buffer = []
                current_tokens = 0
            buffer.append(line)
            current_tokens += line_tokens
        if buffer:
            pieces.append("\n".join(buffer))
        return pieces or [text]

    def _make_code_part(self, chunk: Chunk, part: str, index: int) -> Chunk:
        """根据拆分片段生成一个代码 Parent 子块。"""
        return Chunk(
            chunk_id=f"{chunk.chunk_id}_part_{index}",
            parent_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=part,
            modality=chunk.modality,
            language=chunk.language,
            title=chunk.title,
            section_path=chunk.section_path,
            page_no=chunk.page_no,
            token_count=self.estimate_tokens(part),
            code_language=chunk.code_language or chunk.metadata.get("code_language"),
            metadata={
                **chunk.metadata,
                "is_parent": chunk.metadata.get("is_parent", False),
                "split_from": chunk.chunk_id,
                "code_language": chunk.code_language or chunk.metadata.get("code_language") or "",
            },
        )

    @staticmethod
    def _is_table_separator_line(line: str) -> bool:
        """判断是否为 GFM 表格分隔行（如 | --- | --- |），复用 parser 的判定规则。"""
        stripped = line.strip()
        if not stripped:
            return False
        stripped = stripped.strip("|")
        cells = [c.strip() for c in stripped.split("|")] if "|" in stripped else [stripped]
        if not cells:
            return False
        return all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)

    def _split_table_lines(self, content: str) -> Tuple[List[str], Optional[str], List[str]]:
        """把表格块内容拆成 (数据行, 分隔行, 表头行)。

        GFM 表格块前两行依次为表头与分隔行；不存在分隔行时退化，
        取第一行为表头、其余为数据行。
        """
        lines = [ln for ln in content.split("\n") if ln.strip()]
        if not lines:
            return [], None, []
        if len(lines) >= 2 and self._is_table_separator_line(lines[1]):
            return lines[2:], lines[1], lines[0]
        # 无分隔行：第一行为表头
        return lines[1:], None, lines[0]

    def _split_table_chunk(self, chunk: Chunk) -> List[Chunk]:
        """将超大表格按数据行分组拆成多个 Parent（每组内嵌表头）。

        仅当块超过 table_split_threshold 时由调用方触发；若数据行合并后
        只够一组（实际并不大），保持原块不拆，避免无意义的 `_part_0` 包装。
        """
        data_rows, separator, header = self._split_table_lines(chunk.content)
        if not data_rows:
            return [chunk]

        # 按 parent_chunk_size 把数据行分组，组内 content = 表头 + 分隔行 + 数据行
        groups: List[List[str]] = []
        buffer: List[str] = []
        current_tokens = 0
        for row in data_rows:
            row_tokens = self.estimate_tokens(row)
            if current_tokens + row_tokens > self.parent_chunk_size and buffer:
                groups.append(buffer)
                buffer = []
                current_tokens = 0
            buffer.append(row)
            current_tokens += row_tokens
        if buffer:
            groups.append(buffer)

        if len(groups) <= 1:
            return [chunk]

        header_block = "\n".join([header] + ([separator] if separator else []))
        parts: List[str] = []
        for group in groups:
            content = "\n".join([header_block] + group)
            parts.append(content)

        # 单组仍超 1.5x（超宽行/巨型单元格）→ 按行硬切兜底
        threshold = self.parent_chunk_size * 1.5
        final: List[str] = []
        for part in parts:
            if self.estimate_tokens(part) <= threshold:
                final.append(part)
            else:
                final.extend(self._split_table_hard(part, chunk, threshold))

        return [self._make_table_part(chunk, part, idx) for idx, part in enumerate(final)]

    def _split_table_hard(self, text: str, chunk: Chunk, threshold: float) -> List[str]:
        """按行硬切表格段（每段内嵌表头），每段不超过 threshold。"""
        data_rows, separator, header = self._split_table_lines(text)
        header_block = "\n".join([header] + ([separator] if separator else []))
        pieces: List[str] = []
        buffer: List[str] = []
        current_tokens = 0
        for row in data_rows:
            row_tokens = self.estimate_tokens(row)
            if current_tokens + row_tokens > threshold and buffer:
                pieces.append("\n".join([header_block] + buffer))
                buffer = []
                current_tokens = 0
            buffer.append(row)
            current_tokens += row_tokens
        if buffer:
            pieces.append("\n".join([header_block] + buffer))
        return pieces or [text]

    def _make_table_part(self, chunk: Chunk, part: str, index: int) -> Chunk:
        """根据拆分片段生成一个表格 Parent 子块（不带 code_language）。"""
        return Chunk(
            chunk_id=f"{chunk.chunk_id}_part_{index}",
            parent_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=part,
            modality=chunk.modality,
            language=chunk.language,
            title=chunk.title,
            section_path=chunk.section_path,
            page_no=chunk.page_no,
            token_count=self.estimate_tokens(part),
            metadata={
                **chunk.metadata,
                "is_parent": chunk.metadata.get("is_parent", False),
                "split_from": chunk.chunk_id,
            },
        )

    def _split_table_child_chunks(
        self,
        parent: Chunk,
        doc_id: str,
        child_chunks: List[Chunk],
    ) -> List[Chunk]:
        """将表格 Parent 按 child_chunk_size 切成多个 Child（表头内嵌）。

        小于阈值的表格不拆，原样作为单个 Child（保持既有行为）；
        大表格把数据行累积到 child_chunk_size 再切，child 的 parent_id 指向该表格 Parent。
        """
        if self.estimate_tokens(parent.content) <= self.table_split_threshold:
            return [
                self._create_chunk_from_text(
                    parent.content,
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                    modality=parent.modality,
                    caption=parent.metadata.get("caption"),
                    source_uri=parent.metadata.get("source_uri"),
                )
            ]

        data_rows, separator, header = self._split_table_lines(parent.content)
        header_block = "\n".join([header] + ([separator] if separator else []))
        children: List[Chunk] = []
        buffer: List[str] = []
        current_tokens = 0
        for row in data_rows:
            row_tokens = self.estimate_tokens(row)
            if current_tokens + row_tokens > self.child_chunk_size and buffer:
                children.append(
                    self._create_chunk_from_text(
                        "\n".join([header_block] + buffer),
                        doc_id,
                        parent.chunk_id,
                        parent.title,
                        parent.section_path,
                        parent.page_no,
                        child_chunks,
                        is_parent=False,
                        modality=parent.modality,
                        caption=parent.metadata.get("caption"),
                        source_uri=parent.metadata.get("source_uri"),
                    )
                )
                buffer = []
                current_tokens = 0
            buffer.append(row)
            current_tokens += row_tokens
        if buffer:
            children.append(
                self._create_chunk_from_text(
                    "\n".join([header_block] + buffer),
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                    modality=parent.modality,
                    caption=parent.metadata.get("caption"),
                    source_uri=parent.metadata.get("source_uri"),
                )
            )
        return children

    def _split_code_child_chunks(
        self,
        parent: Chunk,
        doc_id: str,
        child_chunks: List[Chunk],
    ) -> List[Chunk]:
        """将代码 Parent 按 child_chunk_size 切成多个 Child。

        小于阈值的代码块不拆，原样作为单个 Child（保持既有行为）；
        大代码块按行累积到 child_chunk_size 再切，child 的 parent_id 指向该代码 Parent。
        """
        if self.estimate_tokens(parent.content) <= self.code_split_threshold:
            return [
                self._create_chunk_from_text(
                    parent.content,
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                    modality=parent.modality,
                    caption=parent.metadata.get("caption"),
                    source_uri=parent.metadata.get("source_uri"),
                    code_language=parent.code_language or parent.metadata.get("code_language"),
                )
            ]

        lines = parent.content.split("\n")
        children: List[Chunk] = []
        buffer: List[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > self.child_chunk_size and buffer:
                children.append(
                    self._create_chunk_from_text(
                        "\n".join(buffer),
                        doc_id,
                        parent.chunk_id,
                        parent.title,
                        parent.section_path,
                        parent.page_no,
                        child_chunks,
                        is_parent=False,
                        modality=parent.modality,
                        caption=parent.metadata.get("caption"),
                        source_uri=parent.metadata.get("source_uri"),
                        code_language=parent.code_language or parent.metadata.get("code_language"),
                    )
                )
                buffer = []
                current_tokens = 0
            buffer.append(line)
            current_tokens += line_tokens
        if buffer:
            children.append(
                self._create_chunk_from_text(
                    "\n".join(buffer),
                    doc_id,
                    parent.chunk_id,
                    parent.title,
                    parent.section_path,
                    parent.page_no,
                    child_chunks,
                    is_parent=False,
                    modality=parent.modality,
                    caption=parent.metadata.get("caption"),
                    source_uri=parent.metadata.get("source_uri"),
                    code_language=parent.code_language or parent.metadata.get("code_language"),
                )
            )
        return children

    def chunk_document(
        self,
        parsed_content: Dict[str, Any],
        doc_id: str,
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        对文档进行分块

        Returns:
            (parent_chunks, child_chunks)
        """
        # 1. 创建 parent chunks
        parent_chunks = self.create_parent_chunks(parsed_content, doc_id)

        # 2. 创建 child chunks
        child_chunks = self.create_child_chunks(parent_chunks, doc_id)

        return parent_chunks, child_chunks


# 全局分块器实例
from datetime import datetime
document_chunker = DocumentChunker()
