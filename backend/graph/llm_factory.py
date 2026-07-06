# -*- coding: utf-8 -*-
"""Shared DeepSeek model initialization."""

from functools import lru_cache
from typing import Literal

from langchain_deepseek import ChatDeepSeek

from backend.core.config import settings


ModelTier = Literal["flash", "pro"]


@lru_cache(maxsize=2)
def get_llm(tier: ModelTier = "flash") -> ChatDeepSeek:
    """Create a shared DeepSeek V4 client for the requested workload tier."""
    model = (
        settings.DEEPSEEK_FLASH_MODEL
        if tier == "flash"
        else settings.DEEPSEEK_PRO_MODEL
    )
    return ChatDeepSeek(
        model=model,
        api_key=settings.DEEPSEEK_API_KEY or settings.LLM_API_KEY,
        api_base=(
            settings.DEEPSEEK_BASE_URL
            or settings.LLM_BASE_URL
            or "https://api.deepseek.com"
        ),
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
