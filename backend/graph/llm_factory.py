# -*- coding: utf-8 -*-
"""Shared DeepSeek model initialization."""

from functools import lru_cache
from typing import Literal

from langchain_deepseek import ChatDeepSeek

from backend.core.config import settings


ModelTier = Literal["flash", "pro"]


@lru_cache(maxsize=2)
def get_llm(tier: ModelTier = "flash") -> ChatDeepSeek:
    """Create the shared cost-controlled DeepSeek Flash client.

    ``tier`` remains as a compatibility argument for existing callers, but V1
    deliberately maps every workload to Flash so a single request cannot
    silently incur Pro-tier cost.
    """
    model = settings.DEEPSEEK_FLASH_MODEL or "deepseek-v4-flash"
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
