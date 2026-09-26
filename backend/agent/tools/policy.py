"""Server-side authorization and run budget checks for tool calls."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from sqlalchemy.orm import Session

from backend.agent.schemas import ToolExecutionContext
from backend.agent.tools.base import AgentTool
from backend.models.database.user import SessionTable
from backend.services.acl.permission import permission_service


class ToolPolicyError(PermissionError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ToolPolicy:
    def authorize(
        self,
        *,
        tool: AgentTool,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
        db: Session,
        prior_calls: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session = (
            db.query(SessionTable)
            .filter(
                SessionTable.session_id == context.session_id,
                SessionTable.user_id == context.user_id,
            )
            .first()
        )
        if not session:
            raise ToolPolicyError("session_not_owned", "Session is not owned by current user")

        call_rows = list(prior_calls)
        if len(call_rows) >= context.budget.max_tool_calls:
            raise ToolPolicyError("tool_budget_exhausted", "Tool call budget exhausted")

        same_tool_count = sum(
            1 for call in call_rows if call.get("tool_name") == tool.spec.name
        )
        safe_args = dict(arguments)
        for protected in {"user_id", "session_id", "request_id", "run_id", "kb_id", "branch_id"}:
            safe_args.pop(protected, None)

        if tool.spec.name == "knowledge_search":
            if context.kb_id is None:
                raise ToolPolicyError("kb_not_selected", "Knowledge base is not selected")
            if same_tool_count >= context.budget.max_kb_calls:
                raise ToolPolicyError("kb_call_budget_exhausted", "Knowledge search budget exhausted")
            if not permission_service.check_permission(
                context.kb_id,
                context.user_id,
                "read",
                db,
            ):
                raise ToolPolicyError("kb_access_denied", "Knowledge base access denied")
        elif tool.spec.name == "web_search":
            if not context.web_enabled:
                raise ToolPolicyError("web_not_enabled", "Web search is not enabled")
            if same_tool_count >= context.budget.max_web_calls:
                raise ToolPolicyError("web_call_budget_exhausted", "Web search budget exhausted")
        return safe_args


tool_policy = ToolPolicy()
