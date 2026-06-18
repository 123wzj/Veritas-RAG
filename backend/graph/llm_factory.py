# -*- coding: utf-8 -*-
"""OpenAI-compatible chat model initialization.

Prefer LLM_* settings; OPENAI_* remains as a compatibility alias.
"""

from langchain_openai import ChatOpenAI

try:
    from core.config import settings
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.core.config import settings


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance from the project config."""
    return ChatOpenAI(
        model=settings.LLM_MODEL or settings.OPENAI_MODEL,
        api_key=settings.LLM_API_KEY or settings.OPENAI_API_KEY,
        base_url=settings.LLM_BASE_URL or settings.OPENAI_BASE_URL or None,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
