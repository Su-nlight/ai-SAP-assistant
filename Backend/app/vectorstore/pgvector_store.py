"""
pgvector adapter.

Stores embeddings directly inside PostgreSQL using the `vector` extension
— no separate vector database process to run/back up, which is attractive
once a team already operates Postgres for the relational schema. Uses a
dedicated *synchronous* psycopg2 connection rather than the app's async
SQLAlchemy engine: `VectorStore.search`/`upsert` are synchronous methods
(matching FAISS/Chroma), and mixing a sync psycopg2 connection with the
async engine's connection pool would risk pool exhaustion under load —
better to keep them fully separate.

Requires `CREATE EXTENSION vector;` to have been run once on the target
database (a one-time DBA/migration step, not done implicitly here since
it requires elevated privileges the app's runtime role may not have).
"""
from __future__ import annotations

import json
from typing import Any

from app.vectorstore.base import VectorRecord, VectorSearchResult, VectorStore


class PgVectorStore(VectorStore):
    provider_name = "pgvector"

    def __init__(self, dsn: str, table_name: str, dimension: int) -> None:
        import psycopg2
        import psycopg2.extras

        self._psycopg2 = psycopg2
        self._dimension = dimension
        self._table_name = table_name
        # A plain psycopg2 DSN — strip the SQLAlchemy `+driver` component if present.
        self._dsn = dsn.replace("postgresql+psycopg2://", "postgresql://").replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self._dimension}) NOT NULL,
                    document TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
                ON {self._table_name} USING hnsw (embedding vector_cosine_ops);
                """
            )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        with self._conn.cursor() as cur:
            for record in records:
                cur.execute(
                    f"""
                    INSERT INTO {self._table_name} (id, embedding, document, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata;
                    """,
                    (record.id, record.vector, record.document, json.dumps(record.metadata)),
                )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        where_clause = ""
        params: list[Any] = [query_vector]
        if metadata_filter:
            # Simple equality filtering via JSONB containment — sufficient
            # for the {"document_id": "..."} style filters this app uses.
            where_clause = "WHERE metadata @> %s::jsonb"
            params.append(json.dumps(metadata_filter))
        params.extend([query_vector, top_k])

        query = f"""
            SELECT id, document, metadata, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self._table_name}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            VectorSearchResult(id=row[0], document=row[1], metadata=row[2] or {}, score=float(row[3]))
            for row in rows
        ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table_name} WHERE id = ANY(%s);", (ids,))

    def persist(self) -> None:
        # Every write above is already committed (autocommit=True) —
        # nothing to flush, unlike FAISS's in-memory index.
        pass

    def close(self) -> None:
        self._conn.close()
