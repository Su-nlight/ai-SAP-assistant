"""
Hybrid retrieval: dense (embeddings via the vector store) + sparse (BM25),
merged by weighted min-max-normalized score fusion.

Dense retrieval generalizes across paraphrases ("leave policy" ~ "vacation
policy"); BM25 nails exact tokens (`ME21N`, `MAT-100`, `INV-51003`) that
embeddings sometimes conflate. Fusing both consistently outperforms either
alone on this kind of mixed natural-language + identifier-heavy corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.embeddings.base import EmbeddingProvider
from app.rag.bm25_retriever import BM25Document, BM25Retriever
from app.vectorstore.base import VectorSearchResult, VectorStore


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    document: str
    metadata: dict
    dense_score: float
    sparse_score: float
    fused_score: float


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        dense_weight: float,
        sparse_weight: float,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        top_k: int,
        bm25_corpus: list[BM25Document],
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed_query(query)
        dense_results: list[VectorSearchResult] = self._vector_store.search(
            query_vector, top_k=top_k * 2, metadata_filter=metadata_filter
        )

        bm25 = BM25Retriever(bm25_corpus)
        sparse_results = bm25.search(query, top_k=top_k * 2)

        dense_scores = {r.id: r.score for r in dense_results}
        sparse_scores = {r.id: r.score for r in sparse_results}
        dense_norm = _min_max_normalize(dense_scores)
        sparse_norm = _min_max_normalize(sparse_scores)

        by_id: dict[str, RetrievedChunk] = {}
        for result in dense_results:
            by_id[result.id] = RetrievedChunk(
                id=result.id,
                document=result.document,
                metadata=result.metadata,
                dense_score=result.score,
                sparse_score=sparse_scores.get(result.id, 0.0),
                fused_score=0.0,
            )
        for doc in bm25_corpus:
            if doc.id in sparse_scores and doc.id not in by_id:
                by_id[doc.id] = RetrievedChunk(
                    id=doc.id,
                    document=doc.text,
                    metadata=doc.metadata,
                    dense_score=dense_scores.get(doc.id, 0.0),
                    sparse_score=sparse_scores[doc.id],
                    fused_score=0.0,
                )

        for chunk_id, chunk in by_id.items():
            chunk.fused_score = (
                self._dense_weight * dense_norm.get(chunk_id, 0.0)
                + self._sparse_weight * sparse_norm.get(chunk_id, 0.0)
            )

        ranked = sorted(by_id.values(), key=lambda c: c.fused_score, reverse=True)
        return ranked[:top_k]
