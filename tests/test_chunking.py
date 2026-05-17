from __future__ import annotations

import unittest

from docurag.chunking import TextChunker
from docurag.schemas import DocumentPage


class TextChunkerTests(unittest.TestCase):
    def test_split_page_creates_multiple_overlapping_chunks(self) -> None:
        page = DocumentPage(
            id="page-1",
            source="sample.pdf",
            page_number=1,
            text=(
                "DocuRAG loads PDFs. It turns them into chunks with overlap so retrieval "
                "has enough context to answer questions. This sentence makes the example "
                "long enough to force multiple chunks in the test."
            ),
        )

        chunker = TextChunker(chunk_size=75, chunk_overlap=15)
        chunks = chunker.split_page(page)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source, "sample.pdf")
        self.assertLess(chunks[1].start_char, chunks[0].end_char)
        self.assertTrue(all(chunk.text for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

