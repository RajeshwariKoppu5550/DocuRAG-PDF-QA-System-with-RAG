from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schemas import DocumentPage


_WHITESPACE_RE = re.compile(r"[ \t]+")
_PARAGRAPH_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _PARAGRAPH_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def discover_pdf_paths(path: Path) -> list[Path]:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    if resolved_path.is_file():
        if resolved_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {resolved_path.name}")
        return [resolved_path]

    pdf_paths = sorted(candidate for candidate in resolved_path.rglob("*.pdf") if candidate.is_file())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under: {resolved_path}")

    return pdf_paths


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_pdf_pages(path: Path) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF ingestion requires project dependencies. Run `pip install -e .` first."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    source = str(path.expanduser().resolve())

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = normalize_text(page.extract_text() or "")
        page_id = hashlib.sha1(
            f"{source}:{page_number}:{extracted_text}".encode("utf-8")
        ).hexdigest()
        pages.append(
            DocumentPage(
                id=page_id,
                source=source,
                page_number=page_number,
                text=extracted_text,
                metadata={"filename": path.name},
            )
        )

    return pages

