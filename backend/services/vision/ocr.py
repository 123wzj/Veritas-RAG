# -*- coding: utf-8 -*-
"""
OCR and multimodal helpers.

This module now initializes OCR lazily so the text-first RAG pipeline can
start even when optional OCR dependencies are not installed.
"""

from __future__ import annotations

import base64
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

    def _build_payload(self, image_path: str, language: str) -> Dict[str, Any]:
        image_base64 = self._encode_image(image_path)
        lang_instruction = "中文" if language == "zh" else "英文"
        return {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:image/jpeg;base64,{image_base64}",
                            },
                            {
                                "text": (
                                    f"请识别这张图片中的所有文字内容，以{lang_instruction}输出。"
                                    "如果是表格，请转换为 Markdown 表格格式。"
                                    "如果是文档，请尽量保持原有结构。"
                                ),
                            },
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }

    def _resolve_url(self) -> str:
        if self.base_url:
            return f"{self.base_url}/services/vision/multimodal-generation"
        return "https://dashscope.aliyuncs.com/services/vision/multimodal-generation"

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


class MultimodalEmbeddingService:
    """Image embedding helper."""

    def __init__(
        self,
        provider: str = "qwen",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.model = model or settings.EMBEDDING_MODEL

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
            "model": "multimodal-embedding",
            "input": {
                "texts": ["描述这张图片的内容"],
                "images": [f"data:image/jpeg;base64,{image_base64}"],
            },
            "parameters": {"text_type": "document"},
        }

        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/embedding/embedding",
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
            return [0.0] * 1024


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


def get_ocr_service() -> OCRService:
    """Return a lazy OCR proxy so imports stay lightweight."""
    return LazyOCRService()


def get_multimodal_embedding_service() -> MultimodalEmbeddingService:
    return MultimodalEmbeddingService()


ocr_service = get_ocr_service()
multimodal_embedding_service = get_multimodal_embedding_service()
