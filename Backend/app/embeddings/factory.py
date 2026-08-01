"""Embedding provider factory — mirrors llm/factory.py's shape for consistency."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.huggingface_embedder import HuggingFaceEmbedder


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    # Only one local strategy today; the seam is here for a future
    # OpenAI/Cohere embedding provider without touching callers.
    return HuggingFaceEmbedder(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Cached singleton — loading the transformer model is expensive; do it once per process."""
    return build_embedding_provider(get_settings())
