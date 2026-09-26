from backend.agent.evidence.ledger import EvidenceLedgerService
from backend.agent.schemas import EvidenceItem, ToolObservation


def _observation(observation_id: str):
    return ToolObservation(
        observation_id=observation_id,
        tool_call_id="call-1",
        tool_name="knowledge_search",
        status="success",
        summary="found",
        evidence=[EvidenceItem(
            source_type="knowledge_base",
            doc_id="doc-1",
            chunk_id="chunk-1",
            support_snippet="same evidence",
            target_slots=["slot-1"],
        )],
        supported_slots=["slot-1"],
        next_hint="answer",
    )


def test_evidence_ledger_deduplicates_and_assigns_stable_ids():
    service = EvidenceLedgerService()
    ledger, first, added = service.add_observation({}, _observation("o1"))
    ledger, second, added_again = service.add_observation(
        ledger.model_dump(mode="json"), _observation("o2")
    )
    assert added == 1
    assert added_again == 0
    assert first.evidence_ids == ["E1"]
    assert second.evidence_ids == ["E1"]
    assert ledger.slot_coverage["slot-1"].status == "supported"
