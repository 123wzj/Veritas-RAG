"""Runtime dispatcher for legacy, read-only shadow, and controlled ReAct modes."""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any, AsyncGenerator, Dict, Optional

from backend.agent.graph import react_graph
from backend.agent.schemas import RuntimeBudget
from backend.agent.state import create_agent_state
from backend.core.config import settings
from backend.graph.graph import GRAPH_RUN_CONFIG, run_agentic_rag


VALID_RUNTIME_MODES = {"legacy", "react_shadow", "react"}


def get_runtime_mode(override: Optional[str] = None) -> str:
    mode = str(override or settings.AGENT_RUNTIME_MODE or "legacy").strip().lower()
    return mode if mode in VALID_RUNTIME_MODES else "legacy"


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
    mode: str,
    branch_id: Optional[int],
    shadow: bool = False,
) -> Dict[str, Any]:
    budget = build_runtime_budget()
    return create_agent_state(
        query=query,
        user_id=user_id,
        request_id=request_id,
        run_id=f"{request_id}:shadow" if shadow else request_id,
        session_id=session_id,
        kb_id=kb_id,
        web_enabled=web_enabled,
        top_k=top_k,
        runtime_mode="react_shadow" if shadow else mode,
        branch_id=branch_id,
        budgets=budget.model_dump(mode="json"),
    )


async def _invoke_shadow(state: Dict[str, Any]) -> Dict[str, Any]:
    timeout = max(1.0, float(settings.AGENT_SHADOW_TIMEOUT_MS) / 1000.0)
    try:
        final = await asyncio.wait_for(
            react_graph.ainvoke(state, config=GRAPH_RUN_CONFIG),
            timeout=timeout,
        )
        return {
            "status": "completed" if final.get("final_answer") else "failed",
            "stop_reason": final.get("stop_reason"),
            "iteration_count": int(final.get("iteration") or 0),
            "tool_call_count": len(final.get("tool_calls") or []),
            "evidence_count": len((final.get("evidence_ledger") or {}).get("entries") or {}),
            "citation_count": len(final.get("citations") or []),
            "confidence": float(final.get("confidence") or 0.0),
            "verification": final.get("verification") or {},
            "answer_present": bool(final.get("final_answer")),
            "answer_hash": hashlib.sha256(
                str(final.get("final_answer") or "").encode("utf-8")
            ).hexdigest(),
            "error": final.get("error"),
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "react shadow timeout"}
    except Exception as exc:  # pragma: no cover - provider/infrastructure failures
        return {"status": "failed", "error": str(exc)}


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
    **legacy_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    if not session_id:
        raise ValueError("session_id is required by the runtime dispatcher")
    mode = get_runtime_mode(runtime_mode)

    if mode == "legacy":
        async for event in run_agentic_rag(
            query=query,
            user_id=user_id,
            request_id=request_id,
            kb_id=kb_id,
            session_id=session_id,
            web_enabled=web_enabled,
            stream_events=stream_events,
            top_k=top_k,
            branch_id=branch_id,
            **legacy_kwargs,
        ):
            yield event
        return

    react_state = _initial_react_state(
        query=query,
        user_id=user_id,
        request_id=request_id,
        session_id=session_id,
        kb_id=kb_id,
        web_enabled=web_enabled,
        top_k=top_k,
        mode=mode,
        branch_id=branch_id,
        shadow=mode == "react_shadow",
    )

    if mode == "react":
        timeout = max(1.0, build_runtime_budget().deadline_ms / 1000.0)
        if stream_events:
            iterator = react_graph.astream(react_state, config=GRAPH_RUN_CONFIG).__aiter__()
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    yield {"runtime": {"error": "react runtime deadline exceeded", "stop_reason": "deadline_exceeded"}}
                    return
                try:
                    event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield {"runtime": {"error": "react runtime deadline exceeded", "stop_reason": "deadline_exceeded"}}
                    return
                yield event
        else:
            try:
                yield await asyncio.wait_for(
                    react_graph.ainvoke(react_state, config=GRAPH_RUN_CONFIG),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                yield {"error": "react runtime deadline exceeded", "stop_reason": "deadline_exceeded"}
        return

    sample_rate = max(0.0, min(float(settings.AGENT_SHADOW_SAMPLE_RATE), 1.0))
    shadow_task = None
    if sample_rate > 0 and random.random() <= sample_rate:
        shadow_task = asyncio.create_task(_invoke_shadow(react_state))

    legacy_answer = ""
    legacy_citation_count = 0
    async for event in run_agentic_rag(
        query=query,
        user_id=user_id,
        request_id=request_id,
        kb_id=kb_id,
        session_id=session_id,
        web_enabled=web_enabled,
        stream_events=stream_events,
        top_k=top_k,
        branch_id=branch_id,
        **legacy_kwargs,
    ):
        if isinstance(event, dict):
            for value in event.values():
                if isinstance(value, dict) and value.get("final_answer"):
                    legacy_answer = str(value.get("final_answer") or "")
                    legacy_citation_count = len(value.get("citations") or [])
        yield event

    metrics = {"status": "not_sampled"}
    if shadow_task is not None:
        metrics = await shadow_task
        if metrics.get("answer_present"):
            legacy_hash = hashlib.sha256(legacy_answer.encode("utf-8")).hexdigest()
            metrics["exact_answer_match"] = bool(legacy_answer) and metrics.get("answer_hash") == legacy_hash
            metrics["legacy_answer_present"] = bool(legacy_answer)
            metrics["legacy_citation_count"] = legacy_citation_count
    yield {"shadow_evaluation": {"shadow_metrics": metrics, "runtime_mode": mode}}


__all__ = ["build_runtime_budget", "get_runtime_mode", "run_rag_runtime"]
