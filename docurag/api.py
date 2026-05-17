from __future__ import annotations

from dataclasses import asdict

from .pipeline import DocuRAGPipeline


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "The API server requires project dependencies. Run `pip install -e .` first."
        ) from exc

    pipeline = DocuRAGPipeline()
    app = FastAPI(title="DocuRAG", version="0.1.0")

    class IngestRequest(BaseModel):
        path: str = Field(..., description="A PDF file path or a directory of PDFs.")
        force: bool = False

    class AskRequest(BaseModel):
        question: str = Field(..., min_length=1)
        top_k: int | None = Field(default=None, gt=0)
        source: str | None = None

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "backends": pipeline.status()["backends"],
        }

    @app.get("/sources")
    def sources() -> dict[str, object]:
        status = pipeline.status()
        return {
            "counts": status["counts"],
            "sources": [asdict(source) for source in status["sources"]],
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, object]:
        try:
            summaries = pipeline.ingest_path(request.path, force=request.force)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"sources": [asdict(summary) for summary in summaries]}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        try:
            answer = pipeline.answer_question(
                question=request.question,
                top_k=request.top_k,
                source=request.source,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "answer": answer.answer,
            "sources": [asdict(source) for source in answer.sources],
        }

    return app

