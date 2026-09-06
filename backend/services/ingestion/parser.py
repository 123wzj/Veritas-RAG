# -*- coding: utf-8 -*-
"""
文档解析模块
支持 PDF、DOCX、PPTX、Markdown、HTML、TXT 等格式
"""

import os
import re
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
        """解析 Markdown 文件为结构化 blocks（text/table/code/image）。"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = self._parse_markdown_blocks(content, md_dir=Path(file_path).parent)

        return {
            "type": "markdown",
            "blocks": blocks,
            "raw_content": content,
            "metadata": {
                "title": Path(file_path).stem,
            },
        }

    def _parse_markdown_blocks(self, content: str, md_dir: Path) -> List[Dict[str, Any]]:
        """按行扫描 Markdown，识别表格/代码围栏/图片/文本块。

        图片处理：本地相对/绝对路径 → 下载到 images/ 子目录并返回本地 uri；
                  外链 http(s) → 仅保留 alt + source_uri，不下载。
        """
        lines = content.split("\n")
        blocks: List[Dict[str, Any]] = []
        i = 0
        n = len(lines)
        images_dir: Optional[Path] = None

        def _ensure_images_dir() -> Path:
            nonlocal images_dir
            if images_dir is None:
                images_dir = md_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
            return images_dir

        def _emit_text(text: str) -> None:
            text = text.strip("\n")
            if not text.strip():
                return
            if blocks and blocks[-1]["type"] == "text":
                blocks[-1]["content"] += "\n\n" + text
            else:
                blocks.append({"type": "text", "content": text})

        def _handle_image(match) -> None:
            alt = match.group(1) or ""
            raw_uri = (match.group(2) or "").strip()
            if not raw_uri:
                return
            if raw_uri.startswith(("http://", "https://")):
                # 外链：不下载，仅保留 alt + url
                blocks.append({
                    "type": "image",
                    "content": f"![{alt}]({raw_uri})",
                    "caption": alt,
                    "source_uri": raw_uri,
                    "local": False,
                })
            else:
                # 本地路径：另存到 images/ 子目录
                src = Path(raw_uri)
                if not src.is_absolute():
                    src = md_dir / src
                if src.exists():
                    tgt = _ensure_images_dir() / src.name
                    if not tgt.exists():  # 避免覆盖同名
                        tgt = _ensure_images_dir() / f"{len(blocks)}_{src.name}"
                    try:
                        import shutil
                        shutil.copy2(src, tgt)
                    except OSError:
                        tgt = src  # 复制失败回退原路径
                    uri = str(tgt)
                else:
                    uri = raw_uri
                blocks.append({
                    "type": "image",
                    "content": f"![{alt}]({uri})",
                    "caption": alt,
                    "source_uri": uri,
                    "local": src.exists(),
                })

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # 代码围栏
            fence_match = re.match(r"^(`{3,}|~{3,})\s*([\w+-]*)", stripped)
            if fence_match:
                fence = fence_match.group(1)[0]
                lang = fence_match.group(2) or ""
                code_lines: List[str] = []
                i += 1
                closing = ("```" if fence == "`" else "~~~")
                while i < n:
                    if lines[i].strip().startswith(closing):
                        i += 1
                        break
                    code_lines.append(lines[i])
                    i += 1
                blocks.append({
                    "type": "code",
                    "content": "\n".join(code_lines),
                    "language": lang,
                })
                continue

            # GFM 表格：遇到分隔行且前一行是表头时，收集整表为一个 TABLE 块
            if self._is_table_separator(stripped) and i > 0 and lines[i - 1].strip().startswith("|"):
                header = lines[i - 1]
                table_rows = [header, line]
                j = i + 1
                while j < n and lines[j].strip().startswith("|"):
                    table_rows.append(lines[j])
                    j += 1
                # 从已有文本块移除表头行
                blocks[:] = [
                    b for b in blocks
                    if not (b.get("type") == "text" and header.strip() in b["content"])
                ]
                blocks.append({
                    "type": "table",
                    "content": "\n".join(table_rows),
                })
                i = j
                continue

            # 图片：纯图片行产出 image 块
            img_matches = list(re.finditer(r"!\[([^\]]*)\]\(([^)\s]+)\)", line))
            if img_matches and line.strip() == img_matches[0].group(0):
                _handle_image(img_matches[0])
                i += 1
                continue

            # 标题与普通文本
            _emit_text(line)
            i += 1

        return blocks

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        """判断是否为 GFM 表格分隔行，如 | -- | -- | 或 --- | ---。"""
        stripped = line.strip()
        if not stripped:
            return False
        stripped = stripped.strip("|")
        cells = [c.strip() for c in stripped.split("|")] if "|" in stripped else [stripped]
        if not cells:
            return False
        return all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)

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
