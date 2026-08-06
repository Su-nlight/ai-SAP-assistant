from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: uuid.UUID | None = None


class CitationResponse(BaseModel):
    marker: str
    source_name: str
    source_type: str
    excerpt: str


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[CitationResponse]
    provider: str
    model: str
    latency_ms: float
    total_tokens: int | None = None


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    question: str | None
    answer: str | None
    citations: list[CitationResponse]
    timestamp: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    sessions: list[ChatSessionResponse]


class FeedbackRequest(BaseModel):
    message_id: uuid.UUID
    is_positive: bool
    comment: str | None = Field(default=None, max_length=1000)
