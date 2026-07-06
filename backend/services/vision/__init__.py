# -*- coding: utf-8 -*-
"""
视觉和 OCR 模块
"""

from backend.services.vision.ocr import (
    ocr_service,
    multimodal_embedding_service,
    get_ocr_service,
    get_multimodal_embedding_service,
    OCRService,
    QwenVisionService,
    PaddleOCRService,
    MultimodalEmbeddingService,
)

__all__ = [
    "ocr_service",
    "multimodal_embedding_service",
    "get_ocr_service",
    "get_multimodal_embedding_service",
    "OCRService",
    "QwenVisionService",
    "PaddleOCRService",
    "MultimodalEmbeddingService",
]
