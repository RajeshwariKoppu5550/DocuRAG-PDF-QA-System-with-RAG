from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

from .schemas import SearchResult


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class Answerer(Protocol):
    def generate(self, question: str, matches: Sequence[SearchResult]) -> str:
        ...


def _important_terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _build_context(question: str, matches: Sequence[SearchResult], max_chars: int) -> str:
    context_blocks: list[str] = []
    running_length = 0

    for index, match in enumerate(matches, start=1):
        source_name = Path(match.chunk.source).name
        block = f"[{index}] {source_name} p.{match.chunk.page_number}\n{match.chunk.text.strip()}"
        projected_length = running_length + len(block) + 2
        if projected_length > max_chars and context_blocks:
            break
        context_blocks.append(block)
        running_length = projected_length

    joined_context = "\n\n".join(context_blocks)
    return f"Question:\n{question}\n\nContext:\n{joined_context}"


class HeuristicAnswerer:
    def generate(self, question: str, matches: Sequence[SearchResult]) -> str:
        if not matches:
            return "I could not find relevant document context for that question yet."

        question_terms = _important_terms(question)
        sentence_scores: list[tuple[float, str]] = []
        seen_sentences: set[str] = set()

        for match in matches:
            sentences = _SENTENCE_RE.split(match.chunk.text.strip())
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                if len(cleaned_sentence) < 20:
                    continue

                sentence_key = cleaned_sentence.lower()
                if sentence_key in seen_sentences:
                    continue
                seen_sentences.add(sentence_key)

                overlap = len(question_terms.intersection(_important_terms(cleaned_sentence)))
                score = match.score + overlap
                sentence_scores.append((score, cleaned_sentence))

        if not sentence_scores:
            return matches[0].chunk.text.strip()

        sentence_scores.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        total_length = 0
        for _, sentence in sentence_scores:
            projected_length = total_length + len(sentence) + 1
            if projected_length > 700 and selected:
                break
            selected.append(sentence)
            total_length = projected_length
            if len(selected) == 3:
                break

        return " ".join(selected)


class OpenAIAnswerer:
    def __init__(self, api_key: str, model: str, max_context_chars: int = 6000) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI answer generation requires project dependencies. Run `pip install -e .` first."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_context_chars = max_context_chars

    def generate(self, question: str, matches: Sequence[SearchResult]) -> str:
        if not matches:
            return "I could not find relevant document context for that question yet."

        prompt = _build_context(question, matches, self.max_context_chars)
        response = self._client.responses.create(
            model=self.model,
            instructions=(
                "Answer the user's question only from the provided context. "
                "If the context is insufficient, say so clearly. "
                "Cite supporting snippets with square brackets like [1] or [2]."
            ),
            input=prompt,
        )
        return response.output_text.strip()


def create_answerer(settings) -> Answerer:
    if settings.llm_backend == "heuristic":
        return HeuristicAnswerer()

    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when DOCURAG_LLM_BACKEND=openai.")
        return OpenAIAnswerer(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            max_context_chars=settings.max_context_chars,
        )

    raise ValueError(f"Unsupported LLM backend: {settings.llm_backend}")

