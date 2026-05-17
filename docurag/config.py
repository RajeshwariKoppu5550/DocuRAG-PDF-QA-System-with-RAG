from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc


@dataclass(slots=True)
class Settings:
    db_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_backend: str
    llm_backend: str
    openai_api_key: str | None
    openai_embedding_model: str
    openai_chat_model: str
    max_context_chars: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")

    @classmethod
    def from_env(cls) -> "Settings":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_backend = (
            os.getenv("DOCURAG_EMBEDDING_BACKEND")
            or ("openai" if openai_api_key else "simple")
        )
        llm_backend = (
            os.getenv("DOCURAG_LLM_BACKEND")
            or ("openai" if openai_api_key else "heuristic")
        )

        return cls(
            db_path=Path(os.getenv("DOCURAG_DB_PATH", "storage/docurag.db")),
            chunk_size=_env_int("DOCURAG_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("DOCURAG_CHUNK_OVERLAP", 150),
            top_k=_env_int("DOCURAG_TOP_K", 4),
            embedding_backend=embedding_backend.lower(),
            llm_backend=llm_backend.lower(),
            openai_api_key=openai_api_key,
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
            max_context_chars=_env_int("DOCURAG_MAX_CONTEXT_CHARS", 6000),
        )

    def ensure_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

