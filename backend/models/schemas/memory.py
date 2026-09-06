from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


MemoryStatus = Literal["active", "pending_confirmation", "rejected", "inactive", "deleted", "superseded"]
MemoryType = Literal["profile", "preference", "constraint", "project_state", "user_profile", "project_context", "project_decision"]
ScopeType = Literal["user", "project"]
MemorySource = Literal["explicit_user", "user_confirmed", "inferred", "imported", "system"]


class LongTermMemoryPatch(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    status: Optional[MemoryStatus] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    expires_at: Optional[datetime] = None
    operation: Optional[Literal["confirm", "reject", "invalidate", "delete", "edit"]] = None
    request_id: Optional[str] = Field(None, min_length=1, max_length=64)


class LongTermMemoryResponse(BaseModel):
    memory_id: str
    user_id: int
    kb_id: Optional[int] = None
    memory_type: str
    content: str
    scope_type: str
    source: str
    confidence: float
    status: str
    last_confirmed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = {"from_attributes": True}


class MemoryListResponse(BaseModel):
    items: list[LongTermMemoryResponse]
    page: int
    page_size: int
    total: int


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=64)
    rating: Literal["positive", "negative"]
    comment: Optional[str] = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: int
    request_id: str
    rating: Literal["positive", "negative"]
    comment: Optional[str] = None


class TraceSpanResponse(BaseModel):
    id: int
    request_id: str
    span_name: str
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    latency_ms: Optional[int] = None
    model_name: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict = {}
    error: Optional[str] = None


class TraceRunResponse(BaseModel):
    id: int
    request_id: str
    user_id: int
    session_id: str
    kb_id: Optional[int] = None
    route_type: Optional[str] = None
    final_status: str
    answer_mode: Optional[str] = None
    reflection_count: int
    total_latency_ms: Optional[int] = None
    input_tokens: int
    output_tokens: int
    selected_evidence_ids: list[str] = []
    selected_memory_ids: list[str] = []
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    spans: list[TraceSpanResponse] = []


class TraceResponse(BaseModel):
    session_id: str
    runs: list[TraceRunResponse]
