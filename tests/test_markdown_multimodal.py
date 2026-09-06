# -*- coding: utf-8 -*-
"""Markdown 多模态解析与分块测试。"""

import os
import tempfile
import unittest
from pathlib import Path

from backend.models.schemas.knowledge import ModalityType
from backend.services.ingestion.parser import DocumentParser
from backend.services.ingestion.chunker import DocumentChunker


SAMPLE_MD = """# 标题章节

这是一段普通文本内容。

| 季度 | 收入 | 利润 |
|---|---|---|
| Q1 | 100 | 20 |
| Q2 | 150 | 30 |

```python
def process_order(order_id):
    return order_id * 2
```

图片说明：

![架构图](../../assets/arch.png)

![外链图](https://example.com/chart.png)
"""


def _make_md_with_image(tmp_root: Path) -> Path:
    """创建含真实本地图片的 MD，返回 md 路径。"""
    assets = tmp_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    img = assets / "arch.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fakepngdata")
    md = tmp_root / "sample.md"
    md.write_text(
        "# 标题\n\n文本\n\n![架构图](assets/arch.png)\n\n![外链图](https://example.com/chart.png)\n",
        encoding="utf-8",
    )
    return md


class MarkdownMultimodalParserTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.md_path = self.root / "sample.md"
        self.md_path.write_text(SAMPLE_MD, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_parser_produces_blocks(self):
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        self.assertEqual(parsed["type"], "markdown")
        blocks = parsed["blocks"]
        types = [b["type"] for b in blocks]
        self.assertIn("table", types)
        self.assertIn("code", types)
        self.assertIn("image", types)

    def test_table_block_content(self):
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        table = next(b for b in parsed["blocks"] if b["type"] == "table")
        self.assertIn("季度", table["content"])
        self.assertIn("Q2", table["content"])

    def test_code_block_keeps_language(self):
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        code = next(b for b in parsed["blocks"] if b["type"] == "code")
        self.assertEqual(code["language"], "python")
        self.assertIn("process_order", code["content"])
        # 代码原样保留，不按句切分
        self.assertIn("return order_id * 2", code["content"])

    def test_image_local_and_remote(self):
        md = _make_md_with_image(self.root)
        parsed = DocumentParser().parse_markdown(str(md))
        images = [b for b in parsed["blocks"] if b["type"] == "image"]
        self.assertEqual(len(images), 2)
        local = next(b for b in images if b["local"])
        remote = next(b for b in images if not b["local"])
        self.assertEqual(remote["source_uri"], "https://example.com/chart.png")
        self.assertEqual(remote["caption"], "外链图")
        # 本地真实图片被复制到 images/ 子目录
        self.assertTrue(Path(local["source_uri"]).exists())
        self.assertEqual(local["caption"], "架构图")


class MarkdownMultimodalChunkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.md_path = self.root / "sample.md"
        self.md_path.write_text(SAMPLE_MD, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_chunker_dispatch_modalities(self):
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        chunker = DocumentChunker()
        parents, children = chunker.chunk_document(parsed, "md_doc")

        parent_modalities = {p.modality for p in parents}
        self.assertIn(ModalityType.TABLE, parent_modalities)
        self.assertIn(ModalityType.CODE, parent_modalities)
        self.assertIn(ModalityType.IMAGE, parent_modalities)

        # 代码/图片块不按句切碎，父块与子块一一对应。
        # （表格块只对「小表格」保持 1:1——大表格拆分由 test_large_table_split_by_rows 覆盖）
        for modality in (ModalityType.CODE, ModalityType.IMAGE):
            non_text_parents = [p for p in parents if p.modality == modality]
            for p in non_text_parents:
                children_of_p = [c for c in children if c.parent_id == p.chunk_id]
                self.assertEqual(len(children_of_p), 1, f"{modality} 块被切碎")
                self.assertEqual(children_of_p[0].modality, modality)

    def test_image_chunk_metadata(self):
        md = _make_md_with_image(self.root)
        parsed = DocumentParser().parse_markdown(str(md))
        chunker = DocumentChunker()
        parents, children = chunker.chunk_document(parsed, "md_doc")
        image_parents = [p for p in parents if p.modality == ModalityType.IMAGE]
        self.assertEqual(len(image_parents), 2)
        # 外链图片 source_uri 指向 url
        remote = next(
            p for p in image_parents
            if "http" in p.metadata.get("source_uri", "")
        )
        self.assertEqual(remote.metadata["source_uri"], "https://example.com/chart.png")
        # 本地图片 source_uri 指向已复制文件
        local = next(p for p in image_parents if not p.metadata.get("source_uri", "").startswith("http"))
        self.assertTrue(Path(local.metadata["source_uri"]).exists())

    def test_code_language_passthrough_small_block(self):
        """小代码块（低于 code_split_threshold）语言透传，且不切碎。"""
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        chunker = DocumentChunker()
        parents, children = chunker.chunk_document(parsed, "md_doc")
        code_parents = [p for p in parents if p.modality == ModalityType.CODE]
        self.assertEqual(len(code_parents), 1)
        self.assertEqual(code_parents[0].code_language, "python")
        self.assertEqual(code_parents[0].metadata["code_language"], "python")
        # 子块同样透传语言
        code_child = next(c for c in children if c.parent_id == code_parents[0].chunk_id)
        self.assertEqual(code_child.code_language, "python")

    def test_large_code_block_split_by_functions(self):
        """超阈值代码块按函数边界拆成多个 Parent，child 指向对应 part parent。"""
        funcs = ["import os", "import json", "from typing import List, Dict, Any, Optional"]
        for i in range(10):
            funcs.append(
                f"def handler_{i}(payload: dict):\n"
                f"    item = payload.get(\"item_{i}\", {{}})\n"
                f"    if not item:\n"
                f"        return None\n"
                f"    result = item.get(\"value\") * 2\n"
                f"    return result\n"
            )
        big_code = "# service.py\n\n" + "\n\n".join(funcs)
        md_path = self.root / "big_code.md"
        md_path.write_text(f"# 大代码块\n\n```python\n{big_code}```\n", encoding="utf-8")
        parsed = DocumentParser().parse_markdown(str(md_path))
        block = next(b for b in parsed["blocks"] if b["type"] == "code")
        self.assertEqual(block["language"], "python")

        # 缩小 parent_chunk_size/code_split_threshold 使代码块按函数边界拆成多个 Parent
        chunker = DocumentChunker(parent_chunk_size=120, code_split_threshold=50)
        parents, children = chunker.chunk_document(parsed, "big_doc")
        code_parents = [p for p in parents if p.modality == ModalityType.CODE]
        # 超阈值 → 拆成多个 parent
        self.assertGreater(len(code_parents), 1)
        self.assertTrue(all(p.code_language == "python" for p in code_parents))
        # child 的 parent_id 指向 code part parent（而非原代码块根）
        code_children = [c for c in children if c.modality == ModalityType.CODE]
        self.assertGreater(len(code_children), 1)
        child_parent_ids = {c.parent_id for c in code_children}
        part_ids = {p.chunk_id for p in code_parents}
        self.assertTrue(child_parent_ids.issubset(part_ids), "child 未指向 code part parent")
        for c in code_children:
            self.assertEqual(c.code_language, "python")

    def test_small_code_block_not_split(self):
        """小代码块在 create_parent_chunks 中不被 _split_large_chunk 拆分。"""
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        chunker = DocumentChunker()
        parents = chunker.create_parent_chunks(parsed, "md_doc")
        code_parents = [p for p in parents if p.modality == ModalityType.CODE]
        self.assertEqual(len(code_parents), 1)
        self.assertEqual(code_parents[0].code_language, "python")
        # 直接调用 _split_large_chunk 也应保持块级完整（低于阈值不拆）
        resplit = chunker._split_large_chunk(code_parents[0])
        self.assertEqual(len(resplit), 1)
        self.assertEqual(resplit[0].chunk_id, code_parents[0].chunk_id)

    # ---- 表格拆分 ----

    def _table_md(self, rows: int) -> Path:
        """生成含 rows 行数据的大表格 MD。"""
        lines = ["| 季度 | 收入 | 利润 |", "|---|---|---|"]
        for i in range(rows):
            lines.append(f"| Q{i} | 收入 {i * 100} 万 | 利润 {i * 30} 万 |")
        md_path = self.root / "big_table.md"
        md_path.write_text("# 大表格\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return md_path

    def test_large_table_split_by_rows(self):
        """超阈值大表按行拆成多个 part parent，每个都内嵌表头，child 指向 part parent。"""
        md = self._table_md(50)
        parsed = DocumentParser().parse_markdown(str(md))
        block = next(b for b in parsed["blocks"] if b["type"] == "table")
        self.assertIn("季度", block["content"])

        # 缩小 parent_chunk_size/table_split_threshold 使表格按行分组拆成多个 Parent
        chunker = DocumentChunker(parent_chunk_size=40, table_split_threshold=30)
        parents, children = chunker.chunk_document(parsed, "big_doc")
        table_parents = [p for p in parents if p.modality == ModalityType.TABLE]
        # 超阈值 → 拆成多个 parent
        self.assertGreater(len(table_parents), 1)
        # 每个 part parent 都内嵌表头行（自洽）
        for p in table_parents:
            self.assertTrue(p.content.startswith("| 季度 | 收入 | 利润 |"), f"表头未保留: {p.chunk_id}")
            self.assertIn("|---|---|---|", p.content.split("\n")[1])
        # part parent 的 parent_id 指向原表格块根
        roots = {p.parent_id for p in table_parents}
        self.assertEqual(len(roots), 1)
        # child 的 parent_id ⊆ part parent id 集，且每个 child 也内嵌表头
        code_children = [c for c in children if c.modality == ModalityType.TABLE]
        self.assertGreater(len(code_children), 1)
        child_parent_ids = {c.parent_id for c in code_children}
        part_ids = {p.chunk_id for p in table_parents}
        self.assertTrue(child_parent_ids.issubset(part_ids), "child 未指向 table part parent")
        for c in code_children:
            self.assertTrue(c.content.startswith("| 季度 | 收入 | 利润 |"), f"child 表头未保留: {c.chunk_id}")

    def test_table_single_group_not_wrapped(self):
        """表格略超阈值但数据行只够一组 → _split_large_chunk 返回原块（无 _part_0 包装）。"""
        md = self._table_md(3)
        parsed = DocumentParser().parse_markdown(str(md))
        chunker = DocumentChunker(parent_chunk_size=120, table_split_threshold=10)
        parents = chunker.create_parent_chunks(parsed, "doc")
        table_parents = [p for p in parents if p.modality == ModalityType.TABLE]
        self.assertEqual(len(table_parents), 1)
        # 直接调用也应保持原块完整（一组时不拆）
        resplit = chunker._split_large_chunk(table_parents[0])
        self.assertEqual(len(resplit), 1)
        self.assertEqual(resplit[0].chunk_id, table_parents[0].chunk_id)

    def test_small_table_unchanged(self):
        """小表格仍 1 parent 1 child，不被 _split_large_chunk 拆分。"""
        parsed = DocumentParser().parse_markdown(str(self.md_path))
        chunker = DocumentChunker()
        parents, children = chunker.chunk_document(parsed, "doc")
        table_parents = [p for p in parents if p.modality == ModalityType.TABLE]
        self.assertEqual(len(table_parents), 1)
        children_of_table = [c for c in children if c.parent_id == table_parents[0].chunk_id]
        self.assertEqual(len(children_of_table), 1)
        self.assertEqual(children_of_table[0].modality, ModalityType.TABLE)
        # 直接调用 _split_large_chunk 也应保持块级完整（低于阈值不拆）
        resplit = chunker._split_large_chunk(table_parents[0])
        self.assertEqual(len(resplit), 1)
        self.assertEqual(resplit[0].chunk_id, table_parents[0].chunk_id)


if __name__ == "__main__":
    unittest.main()