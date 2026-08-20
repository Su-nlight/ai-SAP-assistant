"""Application-layer document ingestion: upload -> extract -> chunk -> embed -> index -> persist."""
from __future__ import annotations

import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import FileTooLargeError, NotFoundError, UnsupportedFileTypeError
from app.embeddings.base import EmbeddingProvider
from app.models.document import Document, DocumentChunk
from app.rag.chunker import chunk_document
from app.repositories.document_repository import DocumentRepository
from app.services.file_parsers import extract_text
from app.vectorstore.base import VectorRecord, VectorStore


class DocumentService:
    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._repo = document_repo
        self._vector_store = vector_store
        self._embedder = embedder
        self._settings = settings

    async def ingest_upload(
        self,
        filename: str,
        raw_bytes: bytes,
        uploaded_by: uuid.UUID,
    ) -> Document:
        document, _extension = await self._stage_document_row(filename, raw_bytes, uploaded_by)
        await self._process_synchronously(document, filename)
        return document

    async def enqueue_async_ingestion(
        self,
        filename: str,
        raw_bytes: bytes,
        uploaded_by: uuid.UUID,
    ) -> Document:
        """Stages the upload and hands off chunk/embed/index work to a Celery
        worker (`app.workers.tasks.process_document_upload`), returning
        immediately with `status="processing"`. Used when
        `DOCUMENT_PROCESSING_MODE=async`."""
        document, extension = await self._stage_document_row(filename, raw_bytes, uploaded_by)
        await self._repo.commit()

        from app.workers.tasks import process_document_upload

        process_document_upload.delay(
            document_id=str(document.id),
            storage_path=document.storage_path,
            file_extension=extension,
            filename=filename,
        )
        return document

    async def _stage_document_row(
        self, filename: str, raw_bytes: bytes, uploaded_by: uuid.UUID
    ) -> tuple[Document, str]:
        extension = Path(filename).suffix.lower()
        if extension not in self._settings.ALLOWED_UPLOAD_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"'{extension}' is not supported. Allowed: {self._settings.ALLOWED_UPLOAD_EXTENSIONS}"
            )

        max_bytes = self._settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(raw_bytes) > max_bytes:
            raise FileTooLargeError(f"File exceeds the {self._settings.MAX_UPLOAD_SIZE_MB}MB limit.")

        storage_dir = Path("./data/uploads")
        storage_dir.mkdir(parents=True, exist_ok=True)
        document_id = uuid.uuid4()
        storage_path = storage_dir / f"{document_id}{extension}"
        storage_path.write_bytes(raw_bytes)

        document = Document(
            id=document_id,
            title=filename,
            file_type=extension.lstrip("."),
            source_type="upload",
            uploaded_by=uploaded_by,
            storage_path=str(storage_path),
            status="processing",
            metadata_json={},
        )
        document = await self._repo.add(document)
        return document, extension

    async def _process_synchronously(self, document: Document, filename: str) -> None:
        text = extract_text(Path(document.storage_path).suffix.lower(), Path(document.storage_path).read_bytes())

        chunks = chunk_document(text, self._settings.CHUNK_SIZE, self._settings.CHUNK_OVERLAP)
        if chunks:
            texts = [c.text for c in chunks]
            vectors = self._embedder.embed_documents(texts)

            records: list[VectorRecord] = []
            chunk_rows: list[DocumentChunk] = []
            for chunk, vector in zip(chunks, vectors):
                embedding_id = str(uuid.uuid4())
                records.append(
                    VectorRecord(
                        id=embedding_id,
                        vector=vector,
                        document=chunk.text,
                        metadata={
                            "document_id": str(document.id),
                            "source_name": filename,
                            "source_type": "document",
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                )
                chunk_rows.append(
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk.chunk_index,
                        chunk_text=chunk.text,
                        embedding_id=embedding_id,
                    )
                )

            self._vector_store.upsert(records)
            self._vector_store.persist()
            for row in chunk_rows:
                self._repo._session.add(row)  # noqa: SLF001 (repository doesn't expose bulk-add; acceptable within same layer)

        document.status = "ready"
        await self._repo.commit()

    async def list_documents(self, limit: int = 100, offset: int = 0) -> list[Document]:
        return await self._repo.list_with_chunk_counts(limit=limit, offset=offset)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        document = await self._repo.get_by_id(document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found.")

        chunks = await self._repo.get_chunks_for_document(document_id)
        embedding_ids = [c.embedding_id for c in chunks]
        if embedding_ids:
            self._vector_store.delete(embedding_ids)
            self._vector_store.persist()

        await self._repo.delete(document)
        await self._repo.commit()
