"""
Background document ingestion task.

Mirrors `DocumentService.ingest_upload`'s pipeline (extract -> chunk ->
embed -> index -> persist) but runs it out-of-process via Celery, using the
synchronous DB session since Celery workers aren't async. The HTTP-facing
`DocumentService` stays the source of truth for *synchronous* ingestion
(and for tests); this task is what `DOCUMENT_PROCESSING_MODE=async` enqueues
so a large upload doesn't block the request thread.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.database.sync_session import get_sync_db_session
from app.embeddings.factory import get_embedding_provider
from app.models.document import Document, DocumentChunk
from app.rag.chunker import chunk_document
from app.services.file_parsers import extract_text
from app.vectorstore.base import VectorRecord
from app.vectorstore.factory import get_vector_store
from app.workers.celery_app import celery_app

logger = logging.getLogger("ai_sap_assistant.workers")


def _mark_document_failed(doc_uuid: uuid.UUID) -> None:
    try:
        with get_sync_db_session() as session:
            document = session.get(Document, doc_uuid)
            if document is not None:
                document.status = "failed"
                session.commit()
    except Exception:  # noqa: BLE001 — best-effort status update only
        logger.exception("Failed to mark document %s as failed", doc_uuid)


@celery_app.task(name="process_document_upload", bind=True, max_retries=2, default_retry_delay=10)
def process_document_upload(self, document_id: str, storage_path: str, file_extension: str, filename: str) -> dict:
    settings = get_settings()
    doc_uuid = uuid.UUID(document_id)

    try:
        raw_bytes = Path(storage_path).read_bytes()
        text = extract_text(file_extension, raw_bytes)
        chunks = chunk_document(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        embedder = get_embedding_provider()
        vector_store = get_vector_store()

        with get_sync_db_session() as session:
            document = session.get(Document, doc_uuid)
            if document is None:
                logger.error("process_document_upload: document %s no longer exists — aborting.", document_id)
                return {"status": "aborted", "reason": "document not found"}

            if chunks:
                texts = [c.text for c in chunks]
                vectors = embedder.embed_documents(texts)

                records: list[VectorRecord] = []
                for chunk, vector in zip(chunks, vectors):
                    embedding_id = str(uuid.uuid4())
                    records.append(
                        VectorRecord(
                            id=embedding_id,
                            vector=vector,
                            document=chunk.text,
                            metadata={
                                "document_id": document_id,
                                "source_name": filename,
                                "source_type": "document",
                                "chunk_index": chunk.chunk_index,
                            },
                        )
                    )
                    session.add(
                        DocumentChunk(
                            document_id=doc_uuid,
                            chunk_index=chunk.chunk_index,
                            chunk_text=chunk.text,
                            embedding_id=embedding_id,
                        )
                    )

                vector_store.upsert(records)
                vector_store.persist()

            document.status = "ready"
            session.commit()

        logger.info("Document %s processed asynchronously: %d chunks indexed.", document_id, len(chunks))
        return {"status": "ready", "chunk_count": len(chunks)}

    except Exception as exc:  # noqa: BLE001 — Celery needs the exception to trigger retry/failure state
        logger.exception("process_document_upload failed for document %s", document_id)
        _mark_document_failed(doc_uuid)
        raise self.retry(exc=exc) from exc
