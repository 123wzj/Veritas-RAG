"""Validated, authorized and idempotent tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.agent.schemas import ToolCallRequest, ToolExecutionContext, ToolObservation
from backend.agent.tools.policy import ToolPolicyError, tool_policy
from backend.agent.tools.registry import ToolRegistry, tool_registry
from backend.models.database.user import AgentObservationTable, AgentToolCallTable


class ToolGateway:
    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or tool_registry

    @staticmethod
    def fingerprint(call: ToolCallRequest) -> str:
        canonical = json.dumps(
            call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(f"{call.tool_name}:{canonical}".encode("utf-8")).hexdigest()

    async def execute(
        self,
        call: ToolCallRequest,
        *,
        context: ToolExecutionContext,
        db: Session,
        prior_calls: Iterable[Dict[str, Any]],
        idempotency_cache: Dict[str, Dict[str, Any]],
    ) -> Tuple[ToolObservation, str]:
        cache_key = (
            f"{context.user_id}:{context.session_id}:"
            f"{context.run_id}:{call.tool_call_id}"
        )
        fingerprint = self.fingerprint(call)
        if cache_key in idempotency_cache:
            cached = ToolObservation.model_validate(idempotency_cache[cache_key])
            cached.reused = True
            return cached, fingerprint

        persisted = (
            db.query(AgentToolCallTable)
            .filter(
                AgentToolCallTable.run_id == context.run_id,
                AgentToolCallTable.tool_call_id == call.tool_call_id,
                AgentToolCallTable.user_id == context.user_id,
                AgentToolCallTable.session_id == context.session_id,
            )
            .first()
        )
        if persisted:
            if persisted.args_hash != fingerprint or persisted.tool_name != call.tool_name:
                observation = ToolObservation(
                    observation_id=str(uuid.uuid4()),
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="denied",
                    summary="Tool call id was already used with different arguments",
                    missing_slots=call.target_slots,
                    next_hint="stop",
                    error_code="idempotency_conflict",
                )
                return observation, fingerprint
            if persisted.observation_json:
                cached = ToolObservation.model_validate(persisted.observation_json)
                cached.reused = True
                idempotency_cache[cache_key] = cached.model_dump(mode="json")
                return cached, fingerprint
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="failed",
                summary="An identical tool call is already in progress",
                missing_slots=call.target_slots,
                next_hint="stop",
                error_code="tool_call_in_progress",
                retryable=True,
            )
            return observation, fingerprint

        started = time.perf_counter()
        persisted = AgentToolCallTable(
            run_id=context.run_id,
            request_id=context.request_id,
            tool_call_id=call.tool_call_id,
            user_id=context.user_id,
            session_id=context.session_id,
            branch_id=context.branch_id,
            kb_id=context.kb_id,
            tool_name=call.tool_name,
            args_hash=fingerprint,
            arguments_json=call.arguments,
            status="started",
        )
        db.add(persisted)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            concurrent = (
                db.query(AgentToolCallTable)
                .filter(
                    AgentToolCallTable.run_id == context.run_id,
                    AgentToolCallTable.tool_call_id == call.tool_call_id,
                    AgentToolCallTable.user_id == context.user_id,
                    AgentToolCallTable.session_id == context.session_id,
                )
                .first()
            )
            if concurrent and concurrent.args_hash == fingerprint and concurrent.observation_json:
                cached = ToolObservation.model_validate(concurrent.observation_json)
                cached.reused = True
                idempotency_cache[cache_key] = cached.model_dump(mode="json")
                return cached, fingerprint
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="failed",
                summary="A concurrent identical tool call is still running",
                missing_slots=call.target_slots,
                next_hint="stop",
                error_code="tool_call_in_progress",
                retryable=True,
            )
            return observation, fingerprint
        retry_count = 0
        try:
            tool = self.registry.require(call.tool_name)
            safe_args = tool_policy.authorize(
                tool=tool,
                arguments=call.arguments,
                context=context,
                db=db,
                prior_calls=prior_calls,
            )
            observation = None
            for attempt in range(max(0, tool.spec.max_retries) + 1):
                try:
                    observation = await asyncio.wait_for(
                        tool.execute(safe_args, context),
                        timeout=max(0.1, tool.spec.timeout_ms / 1000),
                    )
                    observation.tool_call_id = call.tool_call_id
                    retry_count = attempt
                    break
                except (ValueError, ValidationError):
                    raise
                except asyncio.TimeoutError:
                    retry_count = attempt
                    if attempt < max(0, tool.spec.max_retries):
                        continue
                    raise
                except Exception:
                    retry_count = attempt
                    if attempt < max(0, tool.spec.max_retries):
                        continue
                    raise
            if observation is None:  # pragma: no cover - loop always assigns or raises
                raise RuntimeError("Tool did not return an observation")
        except ToolPolicyError as exc:
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="denied",
                summary=str(exc),
                missing_slots=call.target_slots,
                next_hint="clarify" if exc.code == "web_not_enabled" else "stop",
                error_code=exc.code,
            )
        except (ValueError, ValidationError) as exc:
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="failed",
                summary=str(exc),
                missing_slots=call.target_slots,
                next_hint="stop",
                error_code="invalid_tool_call",
            )
        except asyncio.TimeoutError:
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="failed",
                summary="Tool execution timed out",
                missing_slots=call.target_slots,
                next_hint="refine_query",
                error_code="tool_timeout",
                retryable=True,
            )
        except Exception as exc:  # pragma: no cover - adapter failures vary by provider
            observation = ToolObservation(
                observation_id=str(uuid.uuid4()),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="failed",
                summary=str(exc),
                missing_slots=call.target_slots,
                next_hint="try_other_tool",
                error_code="tool_execution_failed",
                retryable=True,
            )
        observation.metrics["latency_ms"] = int((time.perf_counter() - started) * 1000)
        idempotency_cache[cache_key] = observation.model_dump(mode="json")
        persisted.status = observation.status
        persisted.retry_count = retry_count
        persisted.latency_ms = int(observation.metrics.get("latency_ms") or 0)
        persisted.error_code = observation.error_code
        persisted.observation_json = observation.model_dump(mode="json")
        persisted.completed_at = datetime.now(timezone.utc)
        existing_observation = (
            db.query(AgentObservationTable)
            .filter(AgentObservationTable.observation_id == observation.observation_id)
            .first()
        )
        if not existing_observation:
            db.add(AgentObservationTable(
                observation_id=observation.observation_id,
                run_id=context.run_id,
                tool_call_id=call.tool_call_id,
                user_id=context.user_id,
                session_id=context.session_id,
                tool_name=call.tool_name,
                status=observation.status,
                summary=observation.summary,
                evidence_ids=observation.evidence_ids,
                supported_slots=observation.supported_slots,
                missing_slots=observation.missing_slots,
                conflicts=observation.conflicts,
                observation_json=observation.model_dump(mode="json"),
            ))
        db.commit()
        return observation, fingerprint
