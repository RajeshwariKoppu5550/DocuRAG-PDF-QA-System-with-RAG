# DocuRAG

DocuRAG is a small, production-shaped RAG project that turns PDFs into searchable context and uses that context to answer questions.

## Pipeline

```text
PDF -> Documents -> Chunks -> Embeddings -> SQLite Vector Store
                                              |
                                              v
                                          Retriever
                                              |
                                              v
                                       LLM + Context
                                              |
                                              v
                                            Answer
```

## What This Project Includes

- PDF ingestion with `pypdf`
- Page-to-chunk splitting with overlap
- A local deterministic embedding fallback with no API dependency
- Optional OpenAI embeddings and answer generation
- A SQLite-backed vector store
- A retriever that ranks chunks with cosine similarity
- A CLI for ingesting and querying
- A FastAPI app for serving the pipeline over HTTP

## Quick Start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

2. Copy `.env.example` into `.env` if you want to customize settings.

3. Ingest a PDF file or a folder of PDFs:

```powershell
docurag ingest .\data\pdfs
```

4. Ask a question:

```powershell
docurag ask "What does the document say about deployment?"
```

5. Start the API:

```powershell
docurag serve --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health`
- `GET /sources`
- `POST /ingest`
- `POST /ask`

Example ingest request:

```json
{
  "path": "./data/pdfs",
  "force": false
}
```

Example ask request:

```json
{
  "question": "What is the refund policy?",
  "top_k": 4
}
```

## Local Mode vs OpenAI Mode

The default experience is zero-setup:

- `DOCURAG_EMBEDDING_BACKEND=simple`
- `DOCURAG_LLM_BACKEND=heuristic`

If you set `OPENAI_API_KEY`, the app automatically switches to:

- `DOCURAG_EMBEDDING_BACKEND=openai`
- `DOCURAG_LLM_BACKEND=openai`

You can still override those backends explicitly with environment variables.

## Project Structure

```text
docurag/
  api.py
  chunking.py
  cli.py
  config.py
  embeddings.py
  llm.py
  loaders.py
  pipeline.py
  retriever.py
  schemas.py
  vector_store.py
tests/
```

## Notes

- The current vector store uses SQLite plus in-process similarity scoring, which keeps setup simple and portable.
- For larger workloads, you can swap the vector store layer for pgvector, Qdrant, Chroma, or another dedicated engine without changing the rest of the pipeline shape.
- The heuristic answerer is meant as a practical offline fallback; for higher-quality synthesis, enable an LLM backend.

