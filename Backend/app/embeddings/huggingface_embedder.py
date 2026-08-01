"""
Local HuggingFace embedding provider.

Runs `sentence-transformers` entirely on-device (CPU by default) — no API
key, no network dependency, no per-call cost. This is deliberate: embeddings
are computed far more often than LLM calls (every chunk, every query), so
keeping them free and local is the highest-leverage cost decision in the
whole system.

BGE models expect a retrieval instruction prefix on the *query* side only
(not on documents) for best accuracy — this class handles that automatically
so callers never need to know the model-specific quirk.
"""
from __future__ import annotations

from threading import Lock

from app.embeddings.base import EmbeddingProvider

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class HuggingFaceEmbedder(EmbeddingProvider):
    provider_name = "huggingface-local"

    _model_lock = Lock()

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        # Imported lazily so importing this module doesn't force torch to
        # load until an embedder is actually constructed (keeps tests and
        # unrelated code paths fast).
        from sentence_transformers import SentenceTransformer

        with self._model_lock:
            self._model = SentenceTransformer(model_name, device=device)

        self._model_name = model_name
        self._batch_size = batch_size
        self.dimension = self._model.get_sentence_embedding_dimension()
        self._uses_bge_instruction = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        query_text = f"{_BGE_QUERY_INSTRUCTION}{text}" if self._uses_bge_instruction else text
        vector = self._model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector[0].tolist()
