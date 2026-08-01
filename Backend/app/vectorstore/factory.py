"""Vector store factory. Chroma/pgvector adapters plug in here — same interface, one new branch."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.vectorstore.base import VectorStore
from app.vectorstore.faiss_store import FaissVectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    match settings.VECTOR_STORE_PROVIDER:
        case "faiss":
            return FaissVectorStore(dimension=settings.VECTOR_DIMENSION, storage_dir=settings.VECTOR_STORE_DIR)
        case "chroma":
            from app.vectorstore.chroma_store import ChromaVectorStore

            return ChromaVectorStore(
                storage_dir=settings.VECTOR_STORE_DIR, collection_name=settings.CHROMA_COLLECTION_NAME
            )
        case "pgvector":
            from app.vectorstore.pgvector_store import PgVectorStore

            return PgVectorStore(
                dsn=settings.DATABASE_URL, table_name=settings.PGVECTOR_TABLE_NAME, dimension=settings.VECTOR_DIMENSION
            )
        case other:
            raise NotImplementedError(f"Unknown VECTOR_STORE_PROVIDER '{other}'.")


@lru_cache
def get_vector_store() -> VectorStore:
    return build_vector_store(get_settings())
