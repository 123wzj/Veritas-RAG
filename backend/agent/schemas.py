"""Typed contracts shared by the ReAct runtime and tool gateway."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


DecisionType = Literal["tool_calls", "final_answer", "refusal", "clarification"]
ToolStatus = Literal["success", "partial", "empty", "failed", "denied"]


class AnswerSlot(BaseModel):
    id: str
    question: str
    required: bool = True
    status: Literal["missing", "partial", "supported", "conflict"] = "missing"
    evidence_ids: List[str] = Field(default_factory=list)


class WorkingMemoryV2(BaseModel):
    schema_version: Literal["wm.v2"] = "wm.v2"
    run_id: str
    user_goal: str
    answer_slots: List[AnswerSlot] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    active_entities: List[str] = Field(default_factory=list)
    known_facts: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    required_sources: List[str] = Field(default_factory=list)
    user_expectations: List[str] = Field(default_factory=list)
    attempted_actions: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_slots: List[str] = Field(default_factory=list)
    current_focus: str = ""
    iteration: int = 0


class ToolCallRequest(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""
    target_slots: List[str] = Field(default_factory=list)


class AgentDecision(BaseModel):
    type: DecisionType
    tool_calls: List[ToolCallRequest] = Field(default_factory=list)
    answer: Optional[str] = None
    cited_evidence_ids: List[str] = Field(default_factory=list)
    reason_summary: str = Field(default="", max_length=500)
    unresolved_slots: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "AgentDecision":
        if self.type == "tool_calls" and not self.tool_calls:
            raise ValueError("tool_calls decision requires at least one tool call")
        if self.type != "tool_calls" and self.tool_calls:
            raise ValueError("only tool_calls decisions may contain tool calls")
        if self.type == "final_answer" and not (self.answer or "").strip():
            raise ValueError("final_answer decision requires answer")
        return self


class EvidenceItem(BaseModel):
    evidence_id: str = ""
    source_type: Literal["knowledge_base", "web", "official_web"]
    title: str = ""
    support_snippet: str = ""
    snippet: str = ""
    url: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    section_path: Optional[str] = None
    page_no: Optional[int] = None
    score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    query: str = ""
    tool_call_id: str = ""
    target_slots: List[str] = Field(default_factory=list)
    fetched_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    observation_id: str
    tool_call_id: str
    tool_name: str
    status: ToolStatus
    summary: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    supported_slots: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    next_hint: Literal[
        "answer", "refine_query", "try_other_tool", "clarify", "stop"
    ] = "stop"
    error_code: Optional[str] = None
    retryable: bool = False
    output_ref: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reused: bool = False


class SlotCoverage(BaseModel):
    status: Literal["missing", "partial", "supported", "conflict"] = "missing"
    evidence_ids: List[str] = Field(default_factory=list)


class EvidenceLedger(BaseModel):
    entries: Dict[str, EvidenceItem] = Field(default_factory=dict)
    slot_coverage: Dict[str, SlotCoverage] = Field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    next_index: int = 1


class RuntimeBudget(BaseModel):
    profile: Literal[
        "chat", "rag_standard", "rag_deep", "long_context", "max_context"
    ] = "rag_standard"
    input_token_limit: int = 32000
    output_token_reserve: int = 8192
    max_iterations: int = 6
    max_tool_calls: int = 6
    max_kb_calls: int = 3
    max_web_calls: int = 2
    deadline_ms: int = 120000


class ToolExecutionContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    request_id: str
    user_id: int
    session_id: str
    branch_id: Optional[int] = None
    kb_id: Optional[int] = None
    web_enabled: bool = False
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)


class VerificationResult(BaseModel):
    publishable: bool
    grounded: bool
    useful: bool
    citation_valid: bool
    missing_slots: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    conflicts_not_disclosed: List[str] = Field(default_factory=list)
    recommended_action: Literal["publish", "retry", "refuse"]
    reason: str = ""
