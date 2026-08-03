"""
BM25 sparse retriever.

Complements dense (embedding) retrieval: BM25 is excellent at exact
keyword/identifier matches (transaction codes like `ME21N`, material
numbers like `MAT-100`, invoice numbers) that dense embeddings sometimes
blur together. The hybrid retriever combines both signals.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass(slots=True)
class BM25Document:
    id: str
    text: str
    metadata: dict


@dataclass(slots=True)
class BM25Result:
    id: str
    score: float


class BM25Retriever:
    """Stateless-ish wrapper: rebuilt from the current corpus on each query.

    For this project's document volumes, rebuilding per query is fast
    enough (milliseconds) and avoids the complexity of incremental BM25
    index maintenance. If corpus size grows past what's comfortable here,
    swap in an Elasticsearch/OpenSearch BM25 backend behind this same
    class signature.
    """

    def __init__(self, documents: list[BM25Document]) -> None:
        self._documents = documents
        self._corpus_tokens = [self._tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int) -> list[BM25Result]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self._documents, scores), key=lambda pair: pair[1], reverse=True)
        return [BM25Result(id=doc.id, score=float(score)) for doc, score in ranked[:top_k] if score > 0]
