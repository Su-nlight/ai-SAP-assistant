"""
FAISS vector store adapter.

FAISS itself only stores vectors + integer positions — no metadata, no
string ids, no delete-by-id. This adapter fills that gap with a small
sidecar records list so the rest of the app can treat it like any other
vector database. At this project's scale (thousands–low millions of
chunks) a flat inner-product index with normalized vectors (== cosine
similarity) is simpler and faster to reason about than an approximate
index, and delete-by-rebuild is cheap enough not to matter.

Swap in `ChromaVectorStore` or `PgVectorStore` (same `VectorStore`
interface) for a real deployment without touching any calling code —
that is the entire point of the Adapter pattern here.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.vectorstore.base import VectorRecord, VectorSearchResult, VectorStore


class FaissVectorStore(VectorStore):
    provider_name = "faiss"

    def __init__(self, dimension: int, storage_dir: str) -> None:
        import faiss  # lazy import: keeps faiss off the import path for code that doesn't need it

        self._faiss = faiss
        self._dimension = dimension
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._storage_dir / "index.faiss"
        self._records_path = self._storage_dir / "records.pkl"

        self._records: dict[str, VectorRecord] = {}
        self._position_to_id: list[str] = []

        if self._index_path.exists() and self._records_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with self._records_path.open("rb") as fh:
                state = pickle.load(fh)
                self._records = state["records"]
                self._position_to_id = state["position_to_id"]
        else:
            self._index = faiss.IndexFlatIP(dimension)

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        # Any record whose id already exists must be dropped from the FAISS
        # index (no native update) and re-added at the end.
        existing_ids = [r.id for r in records if r.id in self._records]
        if existing_ids:
            self.delete(existing_ids)

        vectors = np.array([r.vector for r in records], dtype="float32")
        self._index.add(vectors)
        for record in records:
            self._records[record.id] = record
            self._position_to_id.append(record.id)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype="float32")
        # Over-fetch when a metadata filter is present, since FAISS can't
        # filter natively — we filter the candidate set in Python instead.
        fetch_k = top_k * 5 if metadata_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, positions = self._index.search(query, fetch_k)

        results: list[VectorSearchResult] = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos >= len(self._position_to_id):
                continue
            record_id = self._position_to_id[pos]
            record = self._records.get(record_id)
            if record is None:
                continue  # deleted record whose position was never compacted
            if metadata_filter and not self._matches_filter(record.metadata, metadata_filter):
                continue
            results.append(
                VectorSearchResult(id=record.id, document=record.document, metadata=record.metadata, score=float(score))
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    def delete(self, ids: list[str]) -> None:
        """FAISS has no native delete, so we rebuild the index without the
        deleted ids. Acceptable at this project's scale; a production
        pgvector/Chroma adapter would use native `DELETE`/`delete()` instead."""
        ids_to_remove = set(ids)
        if not ids_to_remove:
            return

        surviving_ids = [rid for rid in self._position_to_id if rid not in ids_to_remove]
        for rid in ids_to_remove:
            self._records.pop(rid, None)

        if surviving_ids:
            vectors = np.array([self._records[rid].vector for rid in surviving_ids], dtype="float32")
            new_index = self._faiss.IndexFlatIP(self._dimension)
            new_index.add(vectors)
        else:
            new_index = self._faiss.IndexFlatIP(self._dimension)

        self._index = new_index
        self._position_to_id = surviving_ids

    def persist(self) -> None:
        self._faiss.write_index(self._index, str(self._index_path))
        with self._records_path.open("wb") as fh:
            pickle.dump({"records": self._records, "position_to_id": self._position_to_id}, fh)
        # A human-readable manifest for quick debugging/ops visibility.
        manifest = {"provider": self.provider_name, "dimension": self._dimension, "count": len(self._records)}
        (self._storage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
