from __future__ import annotations

import hashlib

from .schemas import Chunk, DocumentPage


class TextChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be zero or greater and smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: list[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self.split_page(page))
        return chunks

    def split_page(self, page: DocumentPage) -> list[Chunk]:
        text = page.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            proposed_end = min(start + self.chunk_size, text_length)
            end = self._find_split_boundary(text, start, proposed_end)
            if end <= start:
                end = proposed_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha1(
                    f"{page.id}:{start}:{end}:{chunk_text}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=page.id,
                        source=page.source,
                        page_number=page.page_number,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        metadata=dict(page.metadata),
                    )
                )

            if end >= text_length:
                break

            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _find_split_boundary(self, text: str, start: int, proposed_end: int) -> int:
        if proposed_end >= len(text):
            return len(text)

        minimum_chunk_end = min(proposed_end, start + max(1, self.chunk_size // 2))
        separators = ("\n\n", ". ", "? ", "! ", "; ", "\n", ", ", " ")

        for separator in separators:
            position = text.rfind(separator, minimum_chunk_end, proposed_end)
            if position != -1:
                return position + len(separator)

        return proposed_end

