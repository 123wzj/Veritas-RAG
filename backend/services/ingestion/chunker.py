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


class DocumentChunker:
    """文档分块器"""
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

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
            # Markdown 按章节切分
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
            if chunk.token_count > self.parent_chunk_size * 1.5:
                # 需要进一步切分
                sub_chunks = self._split_large_chunk(chunk)
                final_parents.extend(sub_chunks)
            else:
                final_parents.append(chunk)

        return final_parents

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
            modality=ModalityType.TEXT,
            language=language,
            title=title,
            section_path=section_path,
            page_no=page_no,
            token_count=token_count,
            metadata={
                "is_parent": is_parent,
                "created_at": datetime.now().isoformat(),
            },
        )

        chunks_list.append(chunk)
        return chunk

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """切分过大的 chunk"""
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
