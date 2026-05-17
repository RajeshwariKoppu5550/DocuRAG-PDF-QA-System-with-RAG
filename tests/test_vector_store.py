from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docurag.embeddings import SimpleEmbeddingProvider
from docurag.schemas import Chunk, DocumentPage
from docurag.vector_store import SQLiteVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_most_relevant_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "docurag.db"
            vector_store = SQLiteVectorStore(db_path)
            embedding_provider = SimpleEmbeddingProvider()

            source = str((Path(temp_dir) / "sample.pdf").resolve())
            page = DocumentPage(
                id="page-1",
                source=source,
                page_number=1,
                text="A placeholder page body.",
                metadata={"filename": "sample.pdf"},
            )
            chunks = [
                Chunk(
                    id="chunk-1",
                    document_id="page-1",
                    source=source,
                    page_number=1,
                    text="The vector database stores embeddings for semantic search.",
                    start_char=0,
                    end_char=60,
                    metadata={},
                ),
                Chunk(
                    id="chunk-2",
                    document_id="page-1",
                    source=source,
                    page_number=1,
                    text="Authentication middleware validates user sessions.",
                    start_char=61,
                    end_char=115,
                    metadata={},
                ),
            ]

            vectors = embedding_provider.embed_texts([chunk.text for chunk in chunks])
            vector_store.upsert_source(
                source=source,
                file_hash="hash-1",
                pages=[page],
                chunks=chunks,
                vectors=vectors,
            )

            results = vector_store.search(
                query_vector=embedding_provider.embed_query("How are embeddings searched?"),
                limit=1,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.id, "chunk-1")
            self.assertEqual(vector_store.counts()["chunks"], 2)
            self.assertEqual(vector_store.get_source_fingerprint(source), "hash-1")


if __name__ == "__main__":
    unittest.main()
