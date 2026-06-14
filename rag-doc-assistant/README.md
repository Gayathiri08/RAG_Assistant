# RAG-Based Technical Documentation Assistant

A self-corrective Retrieval-Augmented Generation system over the FastAPI documentation,
built with **LangGraph**, **FAISS**, **HuggingFace embeddings**, **Groq (Llama 3.1)**, and
served via **FastAPI**.

---

## 1. Project Overview

The system answers natural language questions about FastAPI by:

1. Analyzing and rewriting the user's query for better retrieval.
2. Retrieving the top-k most relevant chunks from a FAISS vector store.
3. Grading each retrieved chunk for relevance with an LLM (the self-corrective step).
4. If nothing relevant is found, rewriting the query and re-retrieving (up to a retry limit).
5. Generating a grounded answer with inline source citations.
6. Checking the generated answer for hallucinations against the retrieved context (bonus).

---

## 2. Architecture

### LangGraph Workflow

```
                ┌──────────────────┐
        ┌──────▶│  query_analysis  │
        │       └────────┬─────────┘
        │                │
        │                ▼
        │       ┌──────────────────┐
        │       │    retrieval      │
        │       └────────┬─────────┘
        │                │
        │                ▼
        │       ┌──────────────────┐
        │       │ document_grading  │
        │       └────────┬─────────┘
        │                │
        │     ┌──────────┴───────────┐
        │     │  conditional edge      │
        │     │ (route_after_grading)  │
        │     └──────────┬───────────┘
        │   relevant docs │  no relevant docs
        │   found         │
        │                 ▼
        │       ┌──────────────────┐     no relevant docs
        │       │   generation      │◀── & retries exhausted
        │       └────────┬─────────┘
        │                │
        │                ▼
        │       ┌────────────────────┐
        │       │ hallucination_check │
        │       └────────┬───────────┘
        │                │
        │                ▼
        │              END
        │
        │   no relevant docs & retries remain
        │                ▲
        │       ┌──────────────────┐
        └───────│  increment_retry  │
                └──────────────────┘
```

### Nodes

| Node | Responsibility |
|---|---|
| `query_analysis` | Classifies the question (`conceptual` / `how_to` / `troubleshooting` / `api_reference`) and rewrites it for better retrieval. On retries, rewrites more aggressively. |
| `retrieval` | FAISS similarity search, returns top-k chunks with source metadata. |
| `document_grading` | LLM grades each chunk as relevant/irrelevant. **Self-corrective core.** |
| `increment_retry` | Bumps `retry_count`. Separated out so the conditional edge has one decision point. |
| `generation` | Generates a cited answer from relevant chunks, or returns a graceful "I don't know" if none exist. |
| `hallucination_check` (bonus) | Self-RAG-style check: does the generated answer's claims actually appear in the retrieved context? |

### State Schema (`app/graph/state.py`)

Key fields and why they exist:

- **`original_question`** vs **`question`**: `question` is mutated by query rewriting (and re-rewritten on retries), but the final answer is always generated and graded against `original_question` so the user's actual intent is never lost across rewrites.
- **`retrieved_docs`** vs **`graded_docs`**: kept separate so it's possible to inspect/debug what was retrieved vs. what survived grading and was actually used for generation.
- **`retry_count`**: integer counter, incremented by `increment_retry`. The conditional edge compares it to `MAX_RETRIES` (default 2, configurable via `.env`) to decide retry vs. give up.
- **`query_type`**: passed through to generation to lightly adjust prompt style (e.g. "prioritize code examples" for `how_to`).
- **`hallucination_check`**: `{"grounded": bool, "explanation": str}`. Always populated, even for the "I don't know" path (trivially `grounded=True` since no context was used).

### Conditional Edge (`route_after_grading`)

```python
def route_after_grading(state):
    if state["graded_docs"]:
        return "generation"
    if state["retry_count"] < MAX_RETRIES:
        return "increment_retry"
    return "generation"  # generation handles empty graded_docs as "I don't know"
```

This is the single decision point for the self-corrective loop: relevant docs → answer;
no relevant docs but retries remain → rewrite & retry; no relevant docs and retries
exhausted → answer gracefully degrades to "I don't know" rather than looping forever.

---

## 3. Document Ingestion & Chunking Strategy

The corpus (`corpus/`) contains 5 markdown files covering core FastAPI topics: first
steps, request bodies/Pydantic models, dependency injection, error handling, and file
uploads/forms — chosen to cover `conceptual`, `how_to`, `api_reference`, and
`troubleshooting`-style questions.

- **Splitter**: `MarkdownTextSplitter` for `.md` files (header- and code-block aware, so
  a chunk doesn't get cut mid-code-block or separate a header from its body),
  `RecursiveCharacterTextSplitter` for everything else.
- **Chunk size**: 800 characters (~150–200 tokens). Large enough to hold a full code
  snippet plus surrounding explanation, small enough to stay topically focused for
  precise retrieval.
- **Overlap**: 120 characters (~15%) to preserve context across chunk boundaries.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — runs locally, no API key,
  fast, good enough quality for a small technical corpus.
- **Vector store**: FAISS, persisted to disk under `vector_store/`.

Run ingestion standalone:

```bash
python -m app.ingest
```

This is idempotent — re-running rebuilds the index from whatever is currently in `corpus/`.

---

## 4. Setup & Running

### Prerequisites

- Python 3.10+
- A free Groq API key: https://console.groq.com/keys

### Install

```bash
git clone <your-repo-url>
cd rag-doc-assistant
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

### Build the index

```bash
python -m app.ingest
```

### Run the API

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## 5. API Reference & Example Requests

### `POST /query`

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I upload a file in FastAPI?"}'
```

Response:

```json
{
  "answer": "FastAPI supports file uploads using `UploadFile`... [Source: 05_file_uploads.md]",
  "sources": [
    {"source": "05_file_uploads.md", "chunk_id": "05_file_uploads.md::chunk_1"}
  ],
  "query_type": "how_to",
  "retries_used": 0,
  "hallucination_check": {"grounded": true, "explanation": "Answer matches context."}
}
```

### `POST /ingest`

Add new documents (files and/or URLs) to the existing index:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@new_doc.md" \
  -F 'urls=["https://fastapi.tiangolo.com/tutorial/security/"]'
```

Response:

```json
{"chunks_indexed": 12, "message": "Indexed 12 new chunk(s)."}
```

### `GET /documents`

```bash
curl http://127.0.0.1:8000/documents
```

```json
{
  "documents": [
    {"source": "01_first_steps.md", "chunk_count": 9},
    {"source": "02_request_body.md", "chunk_count": 7}
  ],
  "total_chunks": 76
}
```

### `POST /feedback`

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "answer": "...", "rating": "up", "comment": "Helpful!"}'
```

```json
{"status": "recorded", "feedback_id": 1}
```

Feedback is appended to `feedback_log.jsonl`.

### `GET /health`

Basic health check, returns `{"status": "ok"}`.

---

## 6. Design Decisions & Tradeoffs

- **Groq + Llama 3.1 8B**: chosen for the generous free tier and very fast inference,
  important for a graph with multiple LLM calls per query (analysis, grading per chunk,
  generation, hallucination check). Tradeoff: smaller model is less reliable at strict
  JSON output, so all JSON-parsing nodes (`query_analysis`, `hallucination_check`) have
  fallback defaults if parsing fails.
- **Per-chunk grading**: grading each of the top-k chunks individually (rather than
  grading the whole batch in one call) is more LLM calls but gives cleaner, more
  reliable relevance signal and avoids the model conflating one relevant chunk with
  several irrelevant ones in a single judgment.
- **Retry by query rewriting from `original_question`**: rewrites always start from
  the original question (not the previous rewrite) to avoid compounding semantic drift
  across retries.
- **`/ingest` does incremental additions, not full rebuild**: keeps the endpoint fast
  and avoids re-embedding the entire corpus on every upload. A full rebuild remains
  available via `python -m app.ingest`.
- **Hallucination check is a single-pass groundedness check**, not a full Self-RAG
  regenerate-on-failure loop, given the 2-day scope. If not grounded, the result is
  surfaced to the caller rather than blocking the response (retries are already
  exhausted by this point).

---

## 7. Assumptions

- The corpus is small enough (5 docs, ~76 chunks) that a full FAISS rebuild on
  `python -m app.ingest` is cheap; this would need to change (e.g. incremental
  embedding caching) for a much larger corpus.
- `MAX_RETRIES=2` and `TOP_K=4` are reasonable defaults for a small corpus; both are
  configurable via `.env`.
- Feedback storage is a local JSONL file — sufficient for a take-home, would be a
  proper database in production.

---

## 8. What I'd Improve With More Time

- Full Self-RAG loop: if the hallucination check fails, route back to generation with
  an instruction to be more conservative, or back to retrieval with a stricter query.
- Web search fallback (Tavily/Serper) when `graded_docs` is empty after all retries.
- Conversation memory / multi-turn follow-up questions via LangGraph's checkpointing.
- A minimal Streamlit UI for interactive Q&A.
- Batch grading (one LLM call grading all k chunks at once with structured output) to
  cut latency, with per-chunk grading as a fallback if the batch call's JSON parsing fails.
- More extensive unit tests for each node in isolation.

---

## 9. Project Structure

```
rag-doc-assistant/
├── app/
│   ├── main.py                  # FastAPI app & endpoints
│   ├── config.py                # Environment-driven configuration
│   ├── ingest.py                # Document loading, chunking, FAISS indexing
│   ├── api/
│   │   └── schemas.py           # Pydantic request/response models
│   └── graph/
│       ├── state.py             # GraphState TypedDict
│       ├── llm.py                # LLM / embeddings / vector store singletons
│       ├── nodes_query_analysis.py
│       ├── nodes_retrieval.py
│       ├── nodes_grading.py
│       ├── nodes_retry.py
│       ├── nodes_generation.py
│       ├── nodes_hallucination.py
│       └── graph.py             # StateGraph wiring + conditional routing
├── corpus/                       # FastAPI doc corpus (markdown)
├── tests/                         # Integration tests with mocked LLM/embeddings
├── requirements.txt
├── .env.example
└── README.md
```
