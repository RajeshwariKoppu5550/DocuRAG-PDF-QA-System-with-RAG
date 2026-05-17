from __future__ import annotations

from .embeddings import EmbeddingProvider
from .schemas import SearchResult
from .vector_store import SQLiteVectorStore


class Retriever:
    def __init__(
        self,
        vector_store: SQLiteVectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        source: str | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedding_provider.embed_query(question)
        return self.vector_store.search(
            query_vector=query_vector,
            limit=top_k or self.top_k,
            source=source,
        )

