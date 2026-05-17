from __future__ import annotations

from pathlib import Path

from .chunking import TextChunker
from .config import Settings
from .embeddings import create_embedding_provider
from .llm import create_answerer
from .loaders import discover_pdf_paths, fingerprint_file, load_pdf_pages
from .retriever import Retriever
from .schemas import AnswerResult, SourceCitation, SourceSummary
from .vector_store import SQLiteVectorStore


class DocuRAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_storage()

        self.vector_store = SQLiteVectorStore(self.settings.db_path)
        self.chunker = TextChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_provider = create_embedding_provider(self.settings)
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            top_k=self.settings.top_k,
        )
        self.answerer = create_answerer(self.settings)

    def ingest_path(self, path: str | Path, force: bool = False) -> list[SourceSummary]:
        pdf_paths = discover_pdf_paths(Path(path))
        summaries: list[SourceSummary] = []

        for pdf_path in pdf_paths:
            source = str(pdf_path.expanduser().resolve())
            file_hash = fingerprint_file(pdf_path)

            existing_hash = self.vector_store.get_source_fingerprint(source)
            if existing_hash == file_hash and not force:
                existing_summary = self.vector_store.get_source_summary(source)
                if existing_summary is not None:
                    existing_summary.skipped = True
                    summaries.append(existing_summary)
                    continue

            pages = load_pdf_pages(pdf_path)
            chunks = self.chunker.split_pages(pages)
            vectors = (
                self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
                if chunks
                else []
            )
            self.vector_store.upsert_source(
                source=source,
                file_hash=file_hash,
                pages=pages,
                chunks=chunks,
                vectors=vectors,
            )
            summaries.append(
                SourceSummary(
                    source=source,
                    file_hash=file_hash,
                    page_count=len(pages),
                    chunk_count=len(chunks),
                    skipped=False,
                )
            )

        return summaries

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        source: str | None = None,
    ) -> AnswerResult:
        normalized_source = None
        if source:
            normalized_source = str(Path(source).expanduser().resolve())

        matches = self.retriever.retrieve(
            question=question,
            top_k=top_k,
            source=normalized_source,
        )
        answer = self.answerer.generate(question, matches)
        sources = [
            SourceCitation(
                source=match.chunk.source,
                page_number=match.chunk.page_number,
                score=round(match.score, 4),
                snippet=self._truncate(match.chunk.text, 220),
            )
            for match in matches
        ]
        return AnswerResult(answer=answer, sources=sources, matches=matches)

    def list_sources(self) -> list[SourceSummary]:
        return self.vector_store.list_sources()

    def status(self) -> dict[str, object]:
        return {
            "counts": self.vector_store.counts(),
            "sources": self.list_sources(),
            "backends": {
                "embedding": self.settings.embedding_backend,
                "llm": self.settings.llm_backend,
            },
        }

    @staticmethod
    def _truncate(text: str, length: int) -> str:
        if len(text) <= length:
            return text
        return text[: max(0, length - 3)].rstrip() + "..."

