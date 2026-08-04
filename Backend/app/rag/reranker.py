"""
Cross-encoder reranker.

Hybrid retrieval is fast but imprecise (it scores query and chunk
independently). A cross-encoder reads the (query, chunk) pair *jointly*
and produces a much more accurate relevance score — at the cost of being
too slow to run over the whole corpus. Standard pattern: retrieve a wide
candidate set cheaply, rerank a small top-N precisely.

Model is a small (~66MB), free, local MS MARCO cross-encoder — no API key.
"""
from __future__ import annotations

from threading import Lock

from app.rag.hybrid_retriever import RetrievedChunk

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    _model_lock = Lock()
    _model = None  # loaded lazily and shared across instances (process-wide singleton)

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        if cls._model is not None:
            return
        with cls._model_lock:
            if cls._model is None:
                from sentence_transformers import CrossEncoder

                cls._model = CrossEncoder(_DEFAULT_MODEL)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.document) for c in candidates]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]
        for candidate, score in zip(candidates, scores):
            candidate.fused_score = float(score)  # cross-encoder score supersedes fused hybrid score
        return sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:top_k]
