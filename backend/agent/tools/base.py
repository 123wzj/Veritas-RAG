"""Tool specifications and adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.agent.schemas import ToolExecutionContext, ToolObservation


class ToolSpec(BaseModel):
    name: str
    version: str = "1.0"
    description: str
    input_schema: Dict[str, Any]
    required_permissions: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    timeout_ms: int = 30000
    max_retries: int = 1
    idempotent: bool = True
    parallel_safe: bool = True
    max_output_tokens: int = 6000


class AgentTool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def execute(
        self,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolObservation:
        raise NotImplementedError
