"""Register built-in tools exactly once."""

from backend.agent.tools.knowledge_search import KnowledgeSearchTool
from backend.agent.tools.registry import tool_registry
from backend.agent.tools.web_search import WebSearchTool


def register_builtin_tools() -> None:
    for tool in (KnowledgeSearchTool(), WebSearchTool()):
        if tool.spec.name not in tool_registry.names():
            tool_registry.register(tool)
