"""
FastAPI application exposing the RAG documentation assistant.

Endpoints:
    POST /query       - submit a question, returns answer with sources
    POST /ingest       - ingest new documents (file uploads or URLs)
    GET  /documents    - list indexed documents and chunk counts
    POST /feedback     - submit thumbs up/down + optional comment
    GET  /health       - basic health check

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import (
    DocumentInfo,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HallucinationCheck,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.config import CORPUS_DIR, VECTOR_STORE_DIR
from app.graph.graph import get_compiled_graph
from app.ingest import (
    add_documents_to_existing_store,
    fetch_documents_from_urls,
    load_documents_from_dir,
    split_documents,
)

app = FastAPI(
    title="Technical Documentation RAG Assistant",
    description="A self-corrective RAG system over FastAPI documentation, built with LangGraph.",
    version="1.0.0",
)

FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent / "feedback_log.jsonl"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the LangGraph workflow for a user question."""
    if not VECTOR_STORE_DIR.exists() or not (VECTOR_STORE_DIR / "index.faiss").exists():
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Run `python -m app.ingest` or POST /ingest first.",
        )

    graph = get_compiled_graph()

    try:
        result = graph.invoke(
            {
                "original_question": request.question,
                "question": request.question,
                "retry_count": 0,
            }
        )
    except RuntimeError as exc:
        # Surfaces config errors like a missing GROQ_API_KEY as a clean 500.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [SourceItem(**s) for s in result.get("sources", [])]
    hallucination = result.get("hallucination_check", {"grounded": True, "explanation": ""})

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        query_type=result.get("query_type", "conceptual"),
        retries_used=result.get("retry_count", 0),
        hallucination_check=HallucinationCheck(**hallucination),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] | None = File(default=None),
    urls: str | None = None,
):
    """
    Ingest new documents into the existing vector store.

    Accepts:
    - One or more file uploads (multipart/form-data, field name "files"); markdown,
      text, or HTML.
    - An optional "urls" form field containing a JSON array of URL strings to fetch,
      e.g. '["https://example.com/docs"]'.

    New chunks are added to the existing FAISS index (not a full rebuild). For a full
    rebuild from the corpus directory, run `python -m app.ingest` directly.
    """
    documents = []

    if files:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for upload in files:
                if not upload.filename:
                    continue
                suffix = Path(upload.filename).suffix.lower()
                if suffix not in {".md", ".markdown", ".txt", ".html", ".htm"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {upload.filename}. "
                        "Allowed: .md, .markdown, .txt, .html, .htm",
                    )
                dest = tmp_path / upload.filename
                content = await upload.read()
                dest.write_bytes(content)

                # Also persist a copy into the corpus dir so a future full rebuild includes it.
                CORPUS_DIR.mkdir(parents=True, exist_ok=True)
                (CORPUS_DIR / upload.filename).write_bytes(content)

            documents.extend(load_documents_from_dir(tmp_path))

    if urls:
        try:
            url_list = json.loads(urls)
            if not isinstance(url_list, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="`urls` must be a JSON array of strings")
        documents.extend(fetch_documents_from_urls(url_list))

    if not documents:
        raise HTTPException(
            status_code=400, detail="No valid files or URLs provided for ingestion."
        )

    chunks = split_documents(documents)
    n = add_documents_to_existing_store(chunks)

    return IngestResponse(chunks_indexed=n, message=f"Indexed {n} new chunk(s).")


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """List the documents currently in the corpus directory and their chunk counts."""
    documents = load_documents_from_dir(CORPUS_DIR)
    if not documents:
        return DocumentsResponse(documents=[], total_chunks=0)

    chunks = split_documents(documents)

    counts: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    doc_infos = [DocumentInfo(source=src, chunk_count=cnt) for src, cnt in counts.items()]
    return DocumentsResponse(documents=doc_infos, total_chunks=len(chunks))


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    """Log thumbs up/down feedback (with optional comment) to a local JSONL file."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Simple incrementing ID based on existing line count.
    feedback_id = 1
    if FEEDBACK_LOG_PATH.exists():
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
            feedback_id = sum(1 for _ in f) + 1

    entry = {
        "id": feedback_id,
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }

    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
