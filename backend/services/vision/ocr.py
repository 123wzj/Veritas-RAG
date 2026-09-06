# -*- coding: utf-8 -*-
"""
OCR and multimodal helpers.

This module now initializes OCR lazily so the text-first RAG pipeline can
start even when optional OCR dependencies are not installed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings


class OCRService:
    """Base OCR service contract."""

    async def extract_text_async(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def extract_text(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def describe_image(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        raise NotImplementedError


class NoOpOCRService(OCRService):
    """Graceful fallback when no OCR backend is available."""

    def __init__(self, reason: str = "OCR service unavailable"):
        self.reason = reason

    async def extract_text_async(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return self.extract_text(image_path, language)

    def extract_text(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return {
            "text": "",
            "language": language,
            "model": "noop-ocr",
            "confidence": 0.0,
            "warning": self.reason,
        }

    def describe_image(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return self.extract_text(image_path, language)


class QwenVisionService(OCRService):
    """OCR backed by the Qwen vision API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-max",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.VISION_API_KEY
        self.model = model
        self.base_url = base_url or settings.VISION_BASE_URL

        if not self.api_key:
            raise ValueError("VISION_API_KEY is required for QwenVisionService")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_for_path(image_path: str) -> str:
        """根据图片文件扩展名返回 MIME type，供 data URI 使用。"""
        ext = Path(image_path).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")

    def _build_payload(self, image_path: str, language: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        image_base64 = self._encode_image(image_path)
        lang_instruction = "中文" if language == "zh" else "英文"
        text = prompt or (
            f"请识别这张图片中的所有文字内容，以{lang_instruction}输出。"
            "如果是表格，请转换为 Markdown 表格格式。"
            "如果是文档，请尽量保持原有结构。"
        )
        return {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:{self._mime_for_path(image_path)};base64,{image_base64}",
                            },
                            {
                                "text": text,
                            },
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }

    def _resolve_url(self) -> str:
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/services/aigc/multimodal-generation/generation"
        return "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    @staticmethod
    def _parse_response(result: Dict[str, Any], language: str, model: str) -> Dict[str, Any]:
        if "output" not in result or "choices" not in result["output"]:
            raise ValueError(f"Invalid vision response: {result}")

        text = result["output"]["choices"][0]["message"]["content"][0]["text"]
        return {
            "text": text,
            "language": language,
            "model": model,
            "confidence": 0.95,
        }

    async def extract_text_async(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._resolve_url(),
                headers=headers,
                json=self._build_payload(image_path, language),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Vision API error: {response.status} - {error_text}")
                result = await response.json()

        return self._parse_response(result, language, self.model)

    def extract_text(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self._resolve_url(),
            headers=headers,
            json=self._build_payload(image_path, language),
            timeout=60,
        )
        response.raise_for_status()
        return self._parse_response(response.json(), language, self.model)

    def describe_image(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        """Generate a semantic description of an image (VLM), distinct from OCR.

        The description captures the meaning/subject of the image (charts,
        diagrams, photos) so it can be embedded and retrieved like text.
        """
        import requests

        lang_instruction = "中文" if language == "zh" else "英文"
        prompt = (
            f"请用{lang_instruction}详细描述这张图片的内容：它是什么、包含哪些关键信息、"
            "数据、图表含义、场景或结构。不要只罗列图中文字，要描述图片的整体语义，"
            "便于后续根据文字检索到这张图片。"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self._resolve_url(),
            headers=headers,
            json=self._build_payload(image_path, language, prompt=prompt),
            timeout=60,
        )
        response.raise_for_status()
        return self._parse_response(response.json(), language, self.model)


class PaddleOCRService(OCRService):
    """Local OCR backend powered by PaddleOCR."""

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr"
            ) from exc

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
        )

    async def extract_text_async(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_text, image_path, language)

    def extract_text(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        result = self.ocr.ocr(image_path, cls=True)

        texts: List[str] = []
        total_confidence = 0.0
        count = 0

        if result and result[0]:
            for line in result[0]:
                if line:
                    _, (text, confidence) = line
                    texts.append(text)
                    total_confidence += confidence
                    count += 1

        avg_confidence = total_confidence / count if count else 0.0
        return {
            "text": "\n".join(texts),
            "language": language,
            "model": "paddleocr",
            "confidence": avg_confidence,
            "details": result,
        }

    def describe_image(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        # PaddleOCR 无语义描述能力，回退为 OCR 文字
        return self.extract_text(image_path, language)


class MultimodalEmbeddingService:
    """Image embedding helper.

    使用 DashScope Multimodal-Embedding API（将图片与文本映射到同一语义空间）。
    model 默认取 settings.EMBEDDING_MODEL（BGE-M3 文本模型），但图片向量化需要
    多模态模型（如 qwen3-vl-embedding / qwen2.5-vl-embedding / tongyi-embedding-vision-plus）。
    若传入的 model 不含多模态能力，调用会失败并回退为全 0 向量。
    """

    def __init__(
        self,
        provider: str = "qwen",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.model = model or "qwen3-vl-embedding"
        self.base_url = base_url or getattr(settings, "EMBEDDING_BASE_URL", None)
        self.dimension = getattr(settings, "EMBEDDING_DIMENSION", 1024) or 1024

    def embed_image(self, image_path: str) -> List[float]:
        if self.provider == "qwen":
            return self._embed_image_qwen(image_path)
        raise NotImplementedError(f"Image embedding not implemented for {self.provider}")

    def _embed_image_qwen(self, image_path: str) -> List[float]:
        import requests

        with open(image_path, "rb") as file:
            image_base64 = base64.b64encode(file.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "contents": [
                    {"text": "描述这张图片的内容"},
                    {"image": f"data:{self._mime_for_path(image_path)};base64,{image_base64}"},
                ]
            },
            "parameters": {"dimension": self.dimension},
        }

        url = (
            f"{self.base_url.rstrip('/')}/services/embeddings/multimodal-embedding/multimodal-embedding"
            if self.base_url
            else "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            if "output" not in result or "embeddings" not in result["output"]:
                raise ValueError(f"Invalid embedding response: {result}")
            return result["output"]["embeddings"][0]["embedding"]
        except Exception as exc:
            print(f"Image embedding error: {exc}")
            return [0.0] * self.dimension


def _build_ocr_service() -> OCRService:
    provider = settings.VISION_PROVIDER.lower()

    if provider == "qwen":
        try:
            return QwenVisionService()
        except Exception as exc:
            print(f"Failed to initialize QwenVisionService: {exc}")
            print("Falling back to PaddleOCR...")
            try:
                return PaddleOCRService()
            except Exception as fallback_exc:
                print(f"Failed to initialize PaddleOCRService: {fallback_exc}")
                return NoOpOCRService(str(fallback_exc))

    if provider == "paddleocr":
        try:
            return PaddleOCRService()
        except Exception as exc:
            print(f"Failed to initialize PaddleOCRService: {exc}")
            return NoOpOCRService(str(exc))

    return NoOpOCRService(f"Unknown vision provider: {provider}")


class LazyOCRService(OCRService):
    """Proxy that initializes the real OCR backend on first use."""

    def __init__(self):
        self._service: Optional[OCRService] = None

    def _get_service(self) -> OCRService:
        if self._service is None:
            self._service = _build_ocr_service()
        return self._service

    async def extract_text_async(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return await self._get_service().extract_text_async(image_path, language)

    def extract_text(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return self._get_service().extract_text(image_path, language)

    def describe_image(
        self,
        image_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        return self._get_service().describe_image(image_path, language)


def get_ocr_service() -> OCRService:
    """Return a lazy OCR proxy so imports stay lightweight."""
    return LazyOCRService()


def get_multimodal_embedding_service() -> MultimodalEmbeddingService:
    return MultimodalEmbeddingService()


ocr_service = get_ocr_service()
multimodal_embedding_service = get_multimodal_embedding_service()
