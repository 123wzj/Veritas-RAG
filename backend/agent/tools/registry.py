"""Tool registry; only registered adapters may be called by the agent."""

from __future__ import annotations

from typing import Dict, Iterable, List

from backend.agent.tools.base import AgentTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool, *, replace: bool = False) -> None:
        name = tool.spec.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def require(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unregistered tool: {name}") from exc

    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self, names: Iterable[str] | None = None) -> List[dict]:
        selected = self.names() if names is None else list(names)
        return [self.require(name).spec.model_dump() for name in selected]


tool_registry = ToolRegistry()
