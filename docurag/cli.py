from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from .pipeline import DocuRAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocuRAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file or folder of PDFs.")
    ingest_parser.add_argument("path", help="A PDF file path or a directory containing PDFs.")
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if their fingerprint has not changed.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested documents.")
    ask_parser.add_argument("question", help="The question to answer.")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Override the retrieval depth.")
    ask_parser.add_argument(
        "--source",
        default=None,
        help="Limit retrieval to one ingested source path.",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    subparsers.add_parser("status", help="Show ingestion and store status.")
    return parser


def _print_status(status: dict[str, object]) -> None:
    counts = status["counts"]
    backends = status["backends"]
    sources = status["sources"]

    print("Store")
    print(f"  Sources: {counts['sources']}")
    print(f"  Documents: {counts['documents']}")
    print(f"  Chunks: {counts['chunks']}")
    print(f"  Embeddings: {backends['embedding']}")
    print(f"  Answering: {backends['llm']}")

    if sources:
        print("\nSources")
        for source in sources:
            print(
                f"  - {source.source} | pages={source.page_count} | "
                f"chunks={source.chunk_count} | ingested_at={source.ingested_at}"
            )


def _print_ingest_summaries(summaries) -> None:
    for summary in summaries:
        state = "skipped" if summary.skipped else "ingested"
        print(
            f"{state}: {summary.source} | pages={summary.page_count} | "
            f"chunks={summary.chunk_count}"
        )


def _print_answer(answer) -> None:
    print(answer.answer)
    if answer.sources:
        print("\nSources")
        for source in answer.sources:
            print(
                f"  - {source.source} (page {source.page_number}, score={source.score})"
            )
            print(f"    {source.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit(
                "The API server requires project dependencies. Run `pip install -e .` first."
            ) from exc

        uvicorn.run("docurag.api:create_app", host=args.host, port=args.port, factory=True)
        return 0

    pipeline = DocuRAGPipeline()

    if args.command == "ingest":
        summaries = pipeline.ingest_path(args.path, force=args.force)
        _print_ingest_summaries(summaries)
        return 0

    if args.command == "ask":
        answer = pipeline.answer_question(
            question=args.question,
            top_k=args.top_k,
            source=args.source,
        )
        _print_answer(answer)
        return 0

    if args.command == "status":
        _print_status(pipeline.status())
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

