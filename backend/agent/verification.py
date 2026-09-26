"""Deterministic publication gate for a proposed ReAct answer."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Set

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.schemas import AgentDecision, EvidenceLedger, VerificationResult
from backend.core.config import settings
from backend.services.llm_factory import get_llm


def verify_decision(
    decision_value: Dict[str, Any],
    ledger_value: Dict[str, Any],
    *,
    had_tool_calls: bool,
    working_memory: Dict[str, Any] | None = None,
) -> VerificationResult:
    decision = AgentDecision.model_validate(decision_value)
    ledger = EvidenceLedger.model_validate(ledger_value or {})
    if decision.type in {"refusal", "clarification"}:
        useful = bool((decision.answer or "").strip())
        return VerificationResult(
            publishable=useful,
            grounded=True,
            useful=useful,
            citation_valid=True,
            recommended_action="publish" if useful else "retry",
            reason=f"controlled_{decision.type}",
        )

    cited = list(dict.fromkeys(decision.cited_evidence_ids))
    unknown = [item for item in cited if item not in ledger.entries]
    claims = list(decision.claims)
    claim_ids = [claim.claim_id for claim in claims]
    duplicate_claim_ids = {
        claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1
    }
    claim_support: list[Dict[str, Any]] = []
    claim_evidence: Set[str] = set()
    supported_slots: Set[str] = set()
    unsupported_claims: list[str] = []
    unsupported_claims.extend(
        f"duplicate_claim_id:{claim_id}" for claim_id in sorted(duplicate_claim_ids)
    )

    for claim in claims:
        evidence_ids = list(dict.fromkeys(claim.evidence_ids))
        unknown_claim_evidence = [item for item in evidence_ids if item not in ledger.entries]
        valid_evidence = [item for item in evidence_ids if item in ledger.entries]
        claim_evidence.update(valid_evidence)
        slot_supported = False
        if claim.slot_id and valid_evidence:
            slot_evidence = set(
                (ledger.slot_coverage.get(claim.slot_id).evidence_ids
                 if ledger.slot_coverage.get(claim.slot_id) else [])
            )
            slot_supported = bool(slot_evidence.intersection(valid_evidence))
            # Evidence items carry the retrieval target slots. This is a
            # second deterministic guard against arbitrary slot assignment.
            if not slot_supported:
                slot_supported = any(
                    claim.slot_id in (ledger.entries[evidence_id].target_slots or [])
                    for evidence_id in valid_evidence
                )
        # Factual claims can never opt out of evidence by setting a model-
        # controlled boolean to false.
        requires_evidence = bool(
            claim.claim_type == "factual"
            or (claim.requires_evidence and claim.claim_type != "non_factual")
        )
        if claim.slot_id and not requires_evidence:
            slot_supported = True
        claim_ok = not unknown_claim_evidence and (bool(valid_evidence) if requires_evidence else True)
        if requires_evidence and not slot_supported and claim.slot_id:
            claim_ok = False
        if not claim_ok:
            unsupported_claims.append(
                f"{claim.claim_id}:" + (
                    "unknown_evidence" if unknown_claim_evidence else "missing_support"
                )
            )
        elif claim.slot_id and slot_supported:
            supported_slots.add(claim.slot_id)
        claim_support.append({
            "claim_id": claim.claim_id,
            "slot_id": claim.slot_id,
            "evidence_ids": valid_evidence,
            "supported": claim_ok,
        })

    required_slots = {
        str(slot.get("id"))
        for slot in (working_memory or {}).get("answer_slots") or []
        if slot.get("required", True)
    }
    if not required_slots:
        required_slots = set(ledger.slot_coverage.keys())
    # Keep the old low-level contract usable for direct callers that do not
    # provide task slots. Normal graph runs always have WorkingMemory slots.
    legacy_unscoped_answer = not required_slots and not claims and not ledger.slot_coverage
    missing_slots = sorted(required_slots - supported_slots)
    citation_valid = not unknown and not (claim_evidence - set(cited))
    relevant_conflicts = set()
    for conflict in ledger.conflicts:
        conflict_evidence = set(str(item) for item in (conflict.get("evidence_ids") or []))
        if not conflict_evidence or conflict_evidence.intersection(set(cited) | claim_evidence):
            if conflict.get("conflict_id"):
                relevant_conflicts.add(str(conflict["conflict_id"]))
    conflicts_not_disclosed = sorted(
        relevant_conflicts - set(decision.disclosed_conflict_ids)
    )
    grounded = citation_valid and not unsupported_claims and not missing_slots and not conflicts_not_disclosed
    if had_tool_calls and not cited and not legacy_unscoped_answer:
        grounded = False
        unsupported_claims.append("tool-assisted answer contains no evidence citation")
    if not had_tool_calls and any(
        claim.claim_type == "factual" or claim.requires_evidence
        for claim in claims
    ):
        grounded = grounded and bool(cited)
    useful = bool((decision.answer or "").strip())
    publishable = grounded and useful
    return VerificationResult(
        publishable=publishable,
        grounded=grounded,
        useful=useful,
        citation_valid=citation_valid,
        missing_slots=missing_slots,
        unsupported_claims=[f"unknown_evidence_id:{item}" for item in unknown] + unsupported_claims,
        conflicts_not_disclosed=conflicts_not_disclosed,
        supported_slot_ids=sorted(supported_slots),
        claim_support=claim_support,
        recommended_action="publish" if publishable else "retry",
        reason="verified" if publishable else "answer failed deterministic grounding checks",
    )


def _parse_json_response(response: Any) -> Dict[str, Any]:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


async def verify_decision_with_semantics(
    decision_value: Dict[str, Any],
    ledger_value: Dict[str, Any],
    *,
    had_tool_calls: bool,
    working_memory: Dict[str, Any] | None = None,
) -> tuple[VerificationResult, Dict[str, int]]:
    """Run structural checks, then an evidence-entailment judge when configured."""

    structural = verify_decision(
        decision_value,
        ledger_value,
        had_tool_calls=had_tool_calls,
        working_memory=working_memory,
    )
    decision = AgentDecision.model_validate(decision_value)
    ledger = EvidenceLedger.model_validate(ledger_value or {})
    evidence_claims = [
        claim
        for claim in decision.claims
        if claim.claim_type == "factual"
        or (claim.requires_evidence and claim.claim_type != "non_factual")
    ][: max(1, int(settings.AGENT_SEMANTIC_VERIFICATION_MAX_CLAIMS))]
    if (
        decision.type != "final_answer"
        or not structural.publishable
        or not evidence_claims
        or not settings.AGENT_SEMANTIC_VERIFICATION_ENABLED
        or not (settings.DEEPSEEK_API_KEY or settings.LLM_API_KEY)
    ):
        return structural, {}

    evidence_ids = list(dict.fromkeys(
        evidence_id
        for claim in evidence_claims
        for evidence_id in claim.evidence_ids
        if evidence_id in ledger.entries
    ))
    payload = {
        "user_goal": (working_memory or {}).get("user_goal") or "",
        "required_sources": (working_memory or {}).get("required_sources") or [],
        "answer": decision.answer,
        "claims": [claim.model_dump(mode="json") for claim in evidence_claims],
        "evidence": [
            {
                "evidence_id": evidence_id,
                "source_type": ledger.entries[evidence_id].source_type,
                "title": ledger.entries[evidence_id].title,
                "published_at": ledger.entries[evidence_id].published_at,
                "fetched_at": ledger.entries[evidence_id].fetched_at,
                "support_snippet": (
                    ledger.entries[evidence_id].support_snippet
                    or ledger.entries[evidence_id].snippet
                )[:1600],
            }
            for evidence_id in evidence_ids
        ],
    }
    system_prompt = """你是独立答案证据验证器。只判断给定 Evidence 是否直接支持 Claim，不补充外部知识。
输出严格 JSON：
{"claims":[{"claim_id":"...","status":"supported|partial|unsupported|conflict","reason":"简短原因"}]}
标准：supported=证据直接蕴含该结论；partial=只支持一部分；unsupported=没有直接支持；conflict=证据相互冲突。
若用户要求“最新、当前、截至某日”等时效信息，但网页证据缺少可判断的时间或明显不满足时效要求，判为 partial 或 unsupported。若任务要求指定来源类型而证据不满足，也不能判 supported。
只有 supported 才允许最终发布。不要输出思维链。"""
    try:
        response = await get_llm("flash").ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        parsed = _parse_json_response(response)
        judged = {
            str(item.get("claim_id")): {
                "status": str(item.get("status") or "unsupported"),
                "reason": str(item.get("reason") or "")[:500],
            }
            for item in parsed.get("claims") or []
            if isinstance(item, dict) and item.get("claim_id")
        }
        support_rows = []
        semantic_failures = []
        for row in structural.claim_support:
            claim_id = str(row.get("claim_id") or "")
            judgment = judged.get(claim_id, {"status": "unsupported", "reason": "missing_judgment"})
            supported = bool(row.get("supported")) and judgment["status"] == "supported"
            support_rows.append({**row, "supported": supported, "semantic": judgment})
            if not supported:
                semantic_failures.append(f"{claim_id}:semantic_{judgment['status']}")
        structural.claim_support = support_rows
        structural.unsupported_claims = list(dict.fromkeys([
            *structural.unsupported_claims,
            *semantic_failures,
        ]))
        structural.grounded = not structural.unsupported_claims and not structural.missing_slots and not structural.conflicts_not_disclosed
        structural.publishable = structural.grounded and structural.useful and structural.citation_valid
        structural.recommended_action = "publish" if structural.publishable else "retry"
        structural.reason = "verified" if structural.publishable else "semantic evidence verification failed"
        usage = getattr(response, "usage_metadata", None) or {}
        return structural, {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        }
    except Exception as exc:
        structural.publishable = False
        structural.grounded = False
        structural.recommended_action = "retry"
        structural.reason = "semantic verifier unavailable"
        structural.unsupported_claims = list(dict.fromkeys([
            *structural.unsupported_claims,
            f"semantic_verifier_unavailable:{type(exc).__name__}",
        ]))
        return structural, {}
