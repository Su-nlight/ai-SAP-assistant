from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    message_id: uuid.UUID


class EvaluationResponse(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID | None
    context_precision: float | None
    context_recall: float | None
    answer_relevancy: float | None
    faithfulness: float | None
    groundedness: float | None
    latency_ms: float | None
    total_tokens: int | None
    evaluator: str

    model_config = {"from_attributes": True}


class MetricsResponse(BaseModel):
    avg_answer_relevancy: float | None
    avg_groundedness: float | None
    avg_faithfulness: float | None
    avg_latency_ms: float | None
    avg_total_tokens: float | None
    evaluation_count: int


class DashboardResponse(BaseModel):
    total_users: int
    total_documents: int
    total_chat_sessions: int
    total_messages: int
    total_feedback_positive: int
    total_feedback_negative: int
    active_llm_provider: str
    active_vector_store: str


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    resource: str | None
    ip_address: str | None
    details_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}
