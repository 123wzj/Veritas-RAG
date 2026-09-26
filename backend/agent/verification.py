"""Deterministic publication gate for a proposed ReAct answer."""

from __future__ import annotations

from typing import Any, Dict

from backend.agent.schemas import AgentDecision, EvidenceLedger, VerificationResult


def verify_decision(
    decision_value: Dict[str, Any],
    ledger_value: Dict[str, Any],
    *,
    had_tool_calls: bool,
) -> VerificationResult:
    decision = AgentDecision.model_validate(decision_value)
    ledger = EvidenceLedger.model_validate(ledger_value or {})
    if decision.type in {"refusal", "clarification"}:
        return VerificationResult(
            publishable=True,
            grounded=True,
            useful=bool((decision.answer or "").strip()),
            citation_valid=True,
            recommended_action="publish",
            reason=f"controlled_{decision.type}",
        )

    cited = list(dict.fromkeys(decision.cited_evidence_ids))
    unknown = [item for item in cited if item not in ledger.entries]
    missing_slots = [
        slot_id
        for slot_id, coverage in ledger.slot_coverage.items()
        if coverage.status not in {"supported"}
    ]
    citation_valid = not unknown
    grounded = citation_valid and (bool(cited) if had_tool_calls else True)
    useful = bool((decision.answer or "").strip())
    publishable = grounded and useful
    return VerificationResult(
        publishable=publishable,
        grounded=grounded,
        useful=useful,
        citation_valid=citation_valid,
        missing_slots=missing_slots,
        unsupported_claims=(
            [f"unknown_evidence_id:{item}" for item in unknown]
            + (["tool-assisted answer contains no evidence citation"] if had_tool_calls and not cited else [])
        ),
        conflicts_not_disclosed=[],
        recommended_action="publish" if publishable else "retry",
        reason="verified" if publishable else "answer failed deterministic grounding checks",
    )
