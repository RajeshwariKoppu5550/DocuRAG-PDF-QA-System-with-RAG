from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .schemas import Chunk, DocumentPage, SearchResult, SourceSummary


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


class SQLiteVectorStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_files (
                    source TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
                CREATE INDEX IF NOT EXISTS idx_embeddings_dim ON embeddings(vector_dim);
                """
            )

    def get_source_fingerprint(self, source: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT file_hash FROM source_files WHERE source = ?",
                (source,),
            ).fetchone()
        return None if row is None else str(row["file_hash"])

    def get_source_summary(self, source: str) -> SourceSummary | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    sf.source,
                    sf.file_hash,
                    sf.ingested_at,
                    (SELECT COUNT(*) FROM documents d WHERE d.source = sf.source) AS page_count,
                    (SELECT COUNT(*) FROM chunks c WHERE c.source = sf.source) AS chunk_count
                FROM source_files sf
                WHERE sf.source = ?
                """,
                (source,),
            ).fetchone()

        if row is None:
            return None

        return SourceSummary(
            source=str(row["source"]),
            file_hash=str(row["file_hash"]),
            page_count=int(row["page_count"]),
            chunk_count=int(row["chunk_count"]),
            ingested_at=str(row["ingested_at"]),
        )

    def list_sources(self) -> list[SourceSummary]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sf.source,
                    sf.file_hash,
                    sf.ingested_at,
                    (SELECT COUNT(*) FROM documents d WHERE d.source = sf.source) AS page_count,
                    (SELECT COUNT(*) FROM chunks c WHERE c.source = sf.source) AS chunk_count
                FROM source_files sf
                ORDER BY sf.source ASC
                """
            ).fetchall()

        return [
            SourceSummary(
                source=str(row["source"]),
                file_hash=str(row["file_hash"]),
                page_count=int(row["page_count"]),
                chunk_count=int(row["chunk_count"]),
                ingested_at=str(row["ingested_at"]),
            )
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            source_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM source_files").fetchone()["count"]
            )
            document_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
            )
            chunk_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            )

        return {
            "sources": source_count,
            "documents": document_count,
            "chunks": chunk_count,
        }

    def upsert_source(
        self,
        source: str,
        file_hash: str,
        pages: Sequence[DocumentPage],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        ingested_at = datetime.now(timezone.utc).isoformat()

        with self._connection() as connection:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM documents WHERE source = ?", (source,))
            connection.execute("DELETE FROM source_files WHERE source = ?", (source,))
            connection.execute(
                """
                INSERT INTO source_files (source, file_hash, ingested_at)
                VALUES (?, ?, ?)
                """,
                (source, file_hash, ingested_at),
            )

            connection.executemany(
                """
                INSERT INTO documents (id, source, page_number, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        page.id,
                        page.source,
                        page.page_number,
                        page.text,
                        json.dumps(page.metadata, sort_keys=True),
                    )
                    for page in pages
                ],
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    id,
                    document_id,
                    source,
                    page_number,
                    text,
                    start_char,
                    end_char,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page_number,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        json.dumps(chunk.metadata, sort_keys=True),
                    )
                    for chunk in chunks
                ],
            )

            connection.executemany(
                """
                INSERT INTO embeddings (chunk_id, vector_json, vector_dim)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        json.dumps(list(vector)),
                        len(vector),
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
            connection.commit()

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 4,
        source: str | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []

        with self._connection() as connection:
            if source is None:
                rows = connection.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.source,
                        c.page_number,
                        c.text,
                        c.start_char,
                        c.end_char,
                        c.metadata_json,
                        e.vector_json
                    FROM chunks c
                    INNER JOIN embeddings e ON c.id = e.chunk_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.source,
                        c.page_number,
                        c.text,
                        c.start_char,
                        c.end_char,
                        c.metadata_json,
                        e.vector_json
                    FROM chunks c
                    INNER JOIN embeddings e ON c.id = e.chunk_id
                    WHERE c.source = ?
                    """,
                    (source,),
                ).fetchall()

        scored_results: list[SearchResult] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = _cosine_similarity(query_vector, vector)
            chunk = Chunk(
                id=str(row["id"]),
                document_id=str(row["document_id"]),
                source=str(row["source"]),
                page_number=int(row["page_number"]),
                text=str(row["text"]),
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                metadata=json.loads(row["metadata_json"]),
            )
            scored_results.append(SearchResult(chunk=chunk, score=score))

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:limit]
