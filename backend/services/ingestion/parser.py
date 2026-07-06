# -*- coding: utf-8 -*-
"""
文档解析模块
支持 PDF、DOCX、PPTX、Markdown、HTML、TXT 等格式
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from bs4 import BeautifulSoup

from backend.services.vision.ocr import ocr_service


class DocumentParser:
    """文档解析器"""

    @staticmethod
    def detect_file_type(filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".pptx": "pptx",
            ".md": "markdown",
            ".html": "html",
            ".htm": "html",
            ".txt": "txt",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
        }
        return type_map.get(ext, "unknown")

    @staticmethod
    def generate_doc_id(file_path: str) -> str:
        """生成文档唯一 ID"""
        # 使用文件路径和修改时间生成哈希
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """解析 PDF 文件"""
        reader = PdfReader(file_path)
        text_content = []
        metadata = {
            "page_count": len(reader.pages),
            "title": reader.metadata.title if reader.metadata else None,
            "author": reader.metadata.author if reader.metadata else None,
        }

        # 提取文本
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_content.append({
                        "page": page_num + 1,
                        "content": text,
                    })
            except Exception as e:
                print(f"Error extracting page {page_num + 1}: {e}")

        # 检测是否为扫描 PDF（文本内容太少）
        total_text = "\n".join([p["content"] for p in text_content])
        is_scanned = len(total_text.strip()) < len(text_content) * 50

        return {
            "type": "pdf",
            "is_scanned": is_scanned,
            "pages": text_content,
            "metadata": metadata,
        }

    def parse_docx(self, file_path: str) -> Dict[str, Any]:
        """解析 DOCX 文件"""
        doc = DocxDocument(file_path)
        paragraphs = []
        current_section = None

        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                current_section = para.text
            text = para.text.strip()
            if text:
                paragraphs.append({
                    "content": text,
                    "section": current_section,
                    "style": para.style.name,
                })

        # 提取表格
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)

        return {
            "type": "docx",
            "paragraphs": paragraphs,
            "tables": tables,
            "metadata": {
                "title": doc.core_properties.title or Path(file_path).stem,
            },
        }

    def parse_pptx(self, file_path: str) -> Dict[str, Any]:
        """解析 PPTX 文件"""
        prs = Presentation(file_path)
        slides_content = []

        for slide_num, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)
            slides_content.append({
                "slide": slide_num + 1,
                "content": "\n".join(slide_text),
            })

        return {
            "type": "pptx",
            "slides": slides_content,
            "metadata": {
                "title": Path(file_path).stem,
            },
        }

    def parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """解析 Markdown 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 简单的分块：按标题分割
        sections = []
        current_section = "未分类"
        current_content = []

        lines = content.split("\n")
        for line in lines:
            if line.startswith("#"):
                if current_content:
                    sections.append({
                        "title": current_section,
                        "content": "\n".join(current_content),
                    })
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_content),
            })

        return {
            "type": "markdown",
            "sections": sections,
            "raw_content": content,
            "metadata": {
                "title": Path(file_path).stem,
            },
        }

    def parse_html(self, file_path: str) -> Dict[str, Any]:
        """解析 HTML 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        # 提取文本
        text = soup.get_text()

        # 提取标题
        titles = []
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            titles.append({
                "level": int(h.name[1]),
                "text": h.get_text().strip(),
            })

        return {
            "type": "html",
            "content": text,
            "titles": titles,
            "metadata": {
                "title": soup.title.string if soup.title else Path(file_path).stem,
            },
        }

    def parse_txt(self, file_path: str) -> Dict[str, Any]:
        """解析 TXT 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "type": "txt",
            "content": content,
            "metadata": {
                "title": Path(file_path).stem,
            },
        }

    def parse_image(self, file_path: str) -> Dict[str, Any]:
        """
        解析图片文件
        使用 OCR 提取文字内容
        """

        try:
            # 使用 OCR 提取文字
            ocr_result = ocr_service.extract_text(file_path)

            return {
                "type": "image",
                "content": ocr_result["text"],
                "metadata": {
                    "title": Path(file_path).stem,
                    "ocr_model": ocr_result.get("model"),
                    "confidence": ocr_result.get("confidence"),
                },
            }
        except Exception as e:
            print(f"OCR failed for {file_path}: {e}")
            # OCR 失败时返回基本信息
            return {
                "type": "image",
                "content": f"[图片文件: {Path(file_path).name}]",
                "metadata": {
                    "title": Path(file_path).stem,
                    "ocr_error": str(e),
                },
            }

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            解析结果字典
        """
        file_type = self.detect_file_type(file_path)

        parsers = {
            "pdf": self.parse_pdf,
            "docx": self.parse_docx,
            "pptx": self.parse_pptx,
            "markdown": self.parse_markdown,
            "html": self.parse_html,
            "txt": self.parse_txt,
            "image": self.parse_image,
        }

        parser = parsers.get(file_type)
        if not parser:
            raise ValueError(f"Unsupported file type: {file_type}")

        result = parser(file_path)
        result["file_path"] = file_path
        result["file_type"] = file_type
        return result


# 全局解析器实例
document_parser = DocumentParser()
