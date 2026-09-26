"""Primary controlled ReAct runtime for all online RAG requests."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

from backend.agent.graph import react_graph
from backend.agent.schemas import RuntimeBudget
from backend.agent.state import create_agent_state
from backend.core.config import settings


REACT_RUNTIME_MODE = "react"
GRAPH_RUN_CONFIG = {
    "recursion_limit": max(25, settings.GRAPH_RECURSION_LIMIT),
}


def get_runtime_mode(_override: Optional[str] = None) -> str:
    """Return the only supported online runtime.

    Keeping this helper preserves existing API and trace integration while
    ensuring stale environment values cannot route traffic to the old graph.
    """

    return REACT_RUNTIME_MODE


def build_runtime_budget() -> RuntimeBudget:
    profile = str(settings.AGENT_CONTEXT_PROFILE or "rag_standard")
    if profile not in {"chat", "rag_standard", "rag_deep", "long_context", "max_context"}:
        profile = "rag_standard"
    return RuntimeBudget(
        profile=profile,
        input_token_limit=max(2048, int(settings.AGENT_CONTEXT_INPUT_TOKEN_LIMIT)),
        output_token_reserve=max(512, int(settings.AGENT_OUTPUT_TOKEN_RESERVE)),
        max_iterations=max(1, int(settings.AGENT_REACT_MAX_ITERATIONS)),
        max_tool_calls=max(0, int(settings.AGENT_MAX_TOOL_CALLS)),
        max_kb_calls=max(0, int(settings.AGENT_MAX_KB_CALLS)),
        max_web_calls=max(0, int(settings.AGENT_MAX_WEB_CALLS)),
        deadline_ms=max(1000, int(settings.AGENT_RUN_DEADLINE_MS)),
    )


def _initial_react_state(
    *,
    query: str,
    user_id: int,
    request_id: str,
    session_id: str,
    kb_id: Optional[int],
    web_enabled: bool,
    top_k: Optional[int],
    branch_id: Optional[int],
) -> Dict[str, Any]:
    budget = build_runtime_budget()
    return create_agent_state(
        query=query,
        user_id=user_id,
        request_id=request_id,
        run_id=request_id,
        session_id=session_id,
        kb_id=kb_id,
        web_enabled=web_enabled,
        top_k=top_k,
        runtime_mode=REACT_RUNTIME_MODE,
        branch_id=branch_id,
        budgets=budget.model_dump(mode="json"),
    )


async def run_rag_runtime(
    query: str,
    user_id: int,
    request_id: str,
    kb_id: Optional[int] = None,
    session_id: Optional[str] = None,
    web_enabled: bool = False,
    stream_events: bool = True,
    top_k: Optional[int] = None,
    branch_id: Optional[int] = None,
    runtime_mode: Optional[str] = None,
    **_unused_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run every online RAG request through the controlled ReAct graph."""

    if not session_id:
        raise ValueError("session_id is required by the ReAct runtime")

    # Normalize old callers and stale environment values to the ReAct path.
    _ = get_runtime_mode(runtime_mode)
    react_state = _initial_react_state(
        query=query,
        user_id=user_id,
        request_id=request_id,
        session_id=session_id,
        kb_id=kb_id,
        web_enabled=web_enabled,
        top_k=top_k,
        branch_id=branch_id,
    )
    timeout = max(1.0, build_runtime_budget().deadline_ms / 1000.0)

    if stream_events:
        iterator = react_graph.astream(
            react_state,
            config=GRAPH_RUN_CONFIG,
        ).__aiter__()
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                yield {
                    "runtime": {
                        "error": "react runtime deadline exceeded",
                        "stop_reason": "deadline_exceeded",
                    }
                }
                return
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                yield {
                    "runtime": {
                        "error": "react runtime deadline exceeded",
                        "stop_reason": "deadline_exceeded",
                    }
                }
                return
            yield event
        return

    try:
        yield await asyncio.wait_for(
            react_graph.ainvoke(react_state, config=GRAPH_RUN_CONFIG),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        yield {
            "error": "react runtime deadline exceeded",
            "stop_reason": "deadline_exceeded",
        }


__all__ = ["build_runtime_budget", "get_runtime_mode", "run_rag_runtime"]
