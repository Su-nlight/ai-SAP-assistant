"""Vector store abstraction (Adapter pattern) — FAISS today, Chroma/pgvector as drop-in adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    document: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class VectorSearchResult:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float  # higher is more similar (cosine similarity, normalized)


class VectorStore(ABC):
    provider_name: str

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        raise NotImplementedError
