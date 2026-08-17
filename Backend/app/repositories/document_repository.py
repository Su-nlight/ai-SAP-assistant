from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.document import Document, DocumentChunk
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def list_with_chunk_counts(self, limit: int = 100, offset: int = 0) -> list[Document]:
        stmt = select(Document).options(selectinload(Document.chunks)).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_chunks_for_document(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_chunks(self) -> list[DocumentChunk]:
        """Used to (re)build the BM25 sparse index from the current corpus."""
        result = await self._session.execute(select(DocumentChunk))
        return list(result.scalars().all())


def get_document_repository(session: AsyncSession) -> DocumentRepository:
    return DocumentRepository(session)
