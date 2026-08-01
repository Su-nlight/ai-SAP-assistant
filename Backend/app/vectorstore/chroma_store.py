"""
Chroma vector store adapter.

Chroma is a free, local, persistent vector database with a much simpler
operational story than FAISS for delete/update (native `upsert`/`delete`
by id — no rebuild-the-index dance). We always pass precomputed embeddings
(from our own `EmbeddingProvider`) rather than letting Chroma compute its
own, so swapping FAISS <-> Chroma never changes embedding behavior.

`chromadb` is a heavier dependency than FAISS (pulls in its own small
server stack even in local/persistent mode), so it's lazy-imported here —
importing `app.vectorstore.chroma_store` costs nothing unless this adapter
is actually selected via `VECTOR_STORE_PROVIDER=chroma`.
"""
from __future__ import annotations

from typing import Any

from app.vectorstore.base import VectorRecord, VectorSearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    provider_name = "chroma"

    def __init__(self, storage_dir: str, collection_name: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=storage_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            documents=[r.document for r in records],
            metadatas=[r.metadata for r in records],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if self._collection.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
            where=metadata_filter or None,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results: list[VectorSearchResult] = []
        for rid, doc, meta, distance in zip(ids, documents, metadatas, distances):
            # Chroma's default space is squared-L2 distance on raw vectors;
            # since our embeddings are unit-normalized, a monotonic
            # `1 - distance/2` maps distance back into the same
            # "higher is more similar" cosine-similarity scale FAISS
            # returns, so callers never need to know which backend answered.
            similarity = 1.0 - (distance / 2.0)
            results.append(VectorSearchResult(id=rid, document=doc, metadata=meta or {}, score=similarity))
        return results

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def persist(self) -> None:
        # Chroma's PersistentClient writes through on every mutation —
        # there is no separate flush/save step, unlike FAISS.
        pass
