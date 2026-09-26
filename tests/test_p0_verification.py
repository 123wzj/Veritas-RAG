import asyncio

import backend.agent.verification as verification_module
from backend.agent.schemas import EvidenceItem
from backend.agent.verification import verify_decision, verify_decision_with_semantics


def _ledger(*, conflicts=None):
    return {
        "entries": {
            "E1": EvidenceItem(
                evidence_id="E1",
                source_type="knowledge_base",
                doc_id="doc-1",
                chunk_id="chunk-1",
                support_snippet="Veritas RAG uses a controlled ReAct graph.",
                target_slots=["slot-1"],
            ).model_dump(mode="json"),
            "E2": EvidenceItem(
                evidence_id="E2",
                source_type="web",
                url="https://example.com",
                support_snippet="Trace attempts retain failure history.",
                target_slots=["slot-2"],
            ).model_dump(mode="json"),
        },
        "slot_coverage": {
            "slot-1": {"status": "partial", "evidence_ids": ["E1"]},
            "slot-2": {"status": "partial", "evidence_ids": ["E2"]},
        },
        "conflicts": conflicts or [],
        "next_index": 3,
        "next_conflict_index": 1,
    }


def _working_memory():
    return {
        "answer_slots": [
            {"id": "slot-1", "question": "framework", "required": True},
            {"id": "slot-2", "question": "trace", "required": True},
        ]
    }


def test_verification_blocks_missing_required_slot():
    result = verify_decision(
        {
            "type": "final_answer",
            "answer": "It uses ReAct [E1].",
            "cited_evidence_ids": ["E1"],
            "claims": [{
                "claim_id": "C1",
                "text": "It uses ReAct.",
                "slot_id": "slot-1",
                "evidence_ids": ["E1"],
            }],
        },
        _ledger(),
        had_tool_calls=True,
        working_memory=_working_memory(),
    )
    assert result.publishable is False
    assert result.missing_slots == ["slot-2"]


def test_verification_requires_conflict_disclosure():
    decision = {
        "type": "final_answer",
        "answer": "It uses ReAct and keeps attempts [E1][E2].",
        "cited_evidence_ids": ["E1", "E2"],
        "claims": [
            {"claim_id": "C1", "text": "It uses ReAct.", "slot_id": "slot-1", "evidence_ids": ["E1"]},
            {"claim_id": "C2", "text": "Attempts are retained.", "slot_id": "slot-2", "evidence_ids": ["E2"]},
        ],
    }
    ledger = _ledger(conflicts=[{"conflict_id": "X1", "evidence_ids": ["E2"]}])
    blocked = verify_decision(
        decision,
        ledger,
        had_tool_calls=True,
        working_memory=_working_memory(),
    )
    assert blocked.publishable is False
    assert blocked.conflicts_not_disclosed == ["X1"]

    decision["disclosed_conflict_ids"] = ["X1"]
    allowed = verify_decision(
        decision,
        ledger,
        had_tool_calls=True,
        working_memory=_working_memory(),
    )
    assert allowed.publishable is True


def test_semantic_verifier_blocks_unsupported_claim(monkeypatch):
    class Response:
        content = '{"claims":[{"claim_id":"C1","status":"unsupported","reason":"not entailed"},{"claim_id":"C2","status":"supported","reason":"direct"}]}'
        usage_metadata = {"input_tokens": 30, "output_tokens": 10}

    class FakeLlm:
        async def ainvoke(self, _messages):
            return Response()

    monkeypatch.setattr(verification_module.settings, "AGENT_SEMANTIC_VERIFICATION_ENABLED", True)
    monkeypatch.setattr(verification_module.settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(verification_module, "get_llm", lambda _tier: FakeLlm())
    result, usage = asyncio.run(verify_decision_with_semantics(
        {
            "type": "final_answer",
            "answer": "It uses ReAct and keeps attempts [E1][E2].",
            "cited_evidence_ids": ["E1", "E2"],
            "claims": [
                {"claim_id": "C1", "text": "It is a pipeline.", "slot_id": "slot-1", "evidence_ids": ["E1"]},
                {"claim_id": "C2", "text": "Attempts are retained.", "slot_id": "slot-2", "evidence_ids": ["E2"]},
            ],
        },
        _ledger(),
        had_tool_calls=True,
        working_memory=_working_memory(),
    ))
    assert result.publishable is False
    assert "C1:semantic_unsupported" in result.unsupported_claims
    assert usage == {"input_tokens": 30, "output_tokens": 10}


def test_non_factual_direct_answer_can_cover_slot_without_evidence():
    result = verify_decision(
        {
            "type": "final_answer",
            "answer": "你好！有什么我可以帮你的？",
            "cited_evidence_ids": [],
            "claims": [{
                "claim_id": "C1",
                "text": "礼貌回应用户问候",
                "slot_id": "slot-1",
                "evidence_ids": [],
                "claim_type": "non_factual",
                "requires_evidence": False,
            }],
        },
        {"entries": {}, "slot_coverage": {}, "conflicts": [], "next_index": 1},
        had_tool_calls=False,
        working_memory={"answer_slots": [{"id": "slot-1", "required": True}]},
    )
    assert result.publishable is True
    assert result.supported_slot_ids == ["slot-1"]
