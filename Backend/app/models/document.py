"""Document and DocumentChunk tables. Embedding vectors themselves live in the vector store,
not PostgreSQL, in the default (FAISS) configuration — the `embedding_id` column here is the
join key back to the vector store record. A pgvector deployment would instead store the vector
directly in `document_chunks.embedding` (see docs/ARCHITECTURE.md)."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base
from app.database.types import PortableJSON, PortableUUID


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, default="upload")
    uploaded_by: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="processing")
    metadata_json: Mapped[dict] = mapped_column(PortableJSON, nullable=False, default=dict)

    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(PortableUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
