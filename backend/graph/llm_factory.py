# -*- coding: utf-8 -*-
"""OpenAI chat model initialization.

To switch models, edit OPENAI_MODEL in backend/core/config.py or backend/.env.
"""

from langchain_openai import ChatOpenAI

try:
    from core.config import settings
except ImportError:  # pragma: no cover - fallback for package-style imports
    from backend.core.config import settings


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance from the project config."""
    return ChatOpenAI(
        model=settings.OPENAI_MODEL or settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
