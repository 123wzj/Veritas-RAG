"""Run-scoped evidence ledger with stable E# identifiers."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Tuple

from backend.agent.schemas import EvidenceItem, EvidenceLedger, SlotCoverage, ToolObservation


class EvidenceLedgerService:
    @staticmethod
    def _identity(item: EvidenceItem) -> str:
        source = {
            "source_type": item.source_type,
            "url": item.url,
            "doc_id": item.doc_id,
            "chunk_id": item.chunk_id,
            "parent_id": item.parent_id,
            "snippet": (item.support_snippet or item.snippet)[:500],
        }
        return hashlib.sha256(
            json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def add_observation(
        self,
        ledger_value: Dict,
        observation: ToolObservation,
    ) -> Tuple[EvidenceLedger, ToolObservation, int]:
        ledger = EvidenceLedger.model_validate(ledger_value or {})
        identities = {
            self._identity(item): evidence_id
            for evidence_id, item in ledger.entries.items()
        }
        added = 0
        observation_ids: List[str] = []

        for raw in observation.evidence:
            item = EvidenceItem.model_validate(raw)
            identity = self._identity(item)
            evidence_id = identities.get(identity)
            if not evidence_id:
                evidence_id = f"E{ledger.next_index}"
                ledger.next_index += 1
                item.evidence_id = evidence_id
                ledger.entries[evidence_id] = item
                identities[identity] = evidence_id
                added += 1
            else:
                current = ledger.entries[evidence_id]
                if item.score > current.score:
                    item.evidence_id = evidence_id
                    ledger.entries[evidence_id] = item
            observation_ids.append(evidence_id)

        observation.evidence_ids = list(dict.fromkeys(observation_ids))
        observation.evidence = [ledger.entries[item_id] for item_id in observation.evidence_ids]
        candidate_slots = {
            slot_id
            for item in observation.evidence
            for slot_id in item.target_slots
        }
        for slot_id in candidate_slots:
            coverage = ledger.slot_coverage.get(slot_id, SlotCoverage())
            if coverage.status != "supported":
                coverage.status = "partial"
            coverage.evidence_ids = list(dict.fromkeys([
                *coverage.evidence_ids,
                *observation.evidence_ids,
            ]))
            ledger.slot_coverage[slot_id] = coverage
        for slot_id in observation.supported_slots:
            coverage = ledger.slot_coverage.get(slot_id, SlotCoverage())
            coverage.status = "supported"
            coverage.evidence_ids = list(dict.fromkeys([
                *coverage.evidence_ids,
                *observation.evidence_ids,
            ]))
            ledger.slot_coverage[slot_id] = coverage
        for slot_id in observation.missing_slots:
            ledger.slot_coverage.setdefault(slot_id, SlotCoverage(status="missing"))
        for raw_conflict in observation.conflicts:
            conflict = dict(raw_conflict)
            if not conflict.get("conflict_id"):
                conflict["conflict_id"] = f"X{ledger.next_conflict_index}"
                ledger.next_conflict_index += 1
            ledger.conflicts.append(conflict)
        return ledger, observation, added

    @staticmethod
    def citations(ledger_value: Dict, evidence_ids: Iterable[str]) -> List[Dict]:
        ledger = EvidenceLedger.model_validate(ledger_value or {})
        result: List[Dict] = []
        for evidence_id in evidence_ids:
            item = ledger.entries.get(str(evidence_id))
            if not item:
                continue
            result.append({
                "evidence_id": evidence_id,
                "source_type": item.source_type,
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "parent_id": item.parent_id,
                "title": item.title,
                "page_no": item.page_no,
                "section_path": item.section_path,
                "url": item.url,
                "snippet": item.support_snippet or item.snippet,
                "score": item.score,
            })
        return result


evidence_ledger_service = EvidenceLedgerService()
