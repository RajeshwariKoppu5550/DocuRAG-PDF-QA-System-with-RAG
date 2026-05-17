from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentPage:
    id: str
    source: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    source: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class SourceSummary:
    source: str
    file_hash: str
    page_count: int
    chunk_count: int
    skipped: bool = False
    ingested_at: str | None = None


@dataclass(slots=True)
class SourceCitation:
    source: str
    page_number: int
    score: float
    snippet: str


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[SourceCitation]
    matches: list[SearchResult] = field(default_factory=list)

