from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _batched(items: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


class SimpleEmbeddingProvider:
    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            primary_bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            secondary_bucket = int.from_bytes(digest[5:9], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (min(len(token), 12) / 12.0)

            vector[primary_bucket] += sign * weight
            vector[secondary_bucket] += sign * 0.35

        return _normalize_vector(vector)


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str, batch_size: int = 96) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI embeddings require project dependencies. Run `pip install -e .` first."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in _batched(list(texts), self.batch_size):
            response = self._client.embeddings.create(model=self.model, input=list(batch))
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


def create_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_backend == "simple":
        return SimpleEmbeddingProvider()

    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when DOCURAG_EMBEDDING_BACKEND=openai.")
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")

