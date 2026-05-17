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



## Notes

- The current vector store uses SQLite plus in-process similarity scoring, which keeps setup simple and portable.
- For larger workloads, you can swap the vector store layer for pgvector, Qdrant, Chroma, or another dedicated engine without changing the rest of the pipeline shape.
- The heuristic answerer is meant as a practical offline fallback; for higher-quality synthesis, enable an LLM backend.

