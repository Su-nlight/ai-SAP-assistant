"""Feedback, Evaluation, AuditLog, SAPCache tables."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.database.types import PortableJSON, PortableUUID


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_positive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Low-rated responses get flagged so the evaluation pipeline can pull
    # them into a regression/eval dataset (feedback loop from the spec).
    flagged_for_eval: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationResult(Base):
    __tablename__ = "evaluation"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluator: Mapped[str] = mapped_column(String(50), nullable=False, default="custom")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details_json: Mapped[dict] = mapped_column(PortableJSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SAPCache(Base):
    __tablename__ = "sap_cache"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_type: Mapped[str] = mapped_column(String(50), nullable=False)  # material | vendor | po | invoice
    object_id: Mapped[str] = mapped_column(String(100), nullable=False)
    json_data: Mapped[dict] = mapped_column(PortableJSON, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = ()
