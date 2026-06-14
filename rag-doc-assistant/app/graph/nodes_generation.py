"""
Node 4: Generation.

Generates the final answer using `graded_docs` as context. If `graded_docs` is empty
(meaning all retries were exhausted with no relevant chunks found), this node returns a
graceful "I don't know" response instead of calling the LLM with no context, since an
LLM given an empty context block tends to either hallucinate or produce a verbose
non-answer anyway.

The prompt lightly adapts its instructions based on `query_type` (set in query analysis):
- how_to / api_reference -> prioritize code examples and concrete steps
- troubleshooting -> prioritize likely causes and fixes
- conceptual -> prioritize a clear explanation, examples optional

Citations: each source chunk's `source` filename is included in the context block as
"[Source: filename]" and the prompt instructs the model to cite sources inline using
that same bracket format, e.g. "[Source: 02_request_body.md]".
"""

from __future__ import annotations

from app.graph.llm import get_llm
from app.graph.state import GraphState

STYLE_HINTS = {
    "how_to": "Prioritize concrete steps and a working code example.",
    "api_reference": "Prioritize precise parameter/behavior details and a short code example.",
    "troubleshooting": "Prioritize likely causes and how to fix them.",
    "conceptual": "Prioritize a clear, well-organized explanation.",
}

GENERATION_PROMPT = """You are a helpful technical documentation assistant for FastAPI.

Answer the user's question using ONLY the information in the context chunks below. {style_hint}

Cite the source of each piece of information inline using the format [Source: filename], where \
filename matches the "Source:" label of the chunk you used. If the context does not contain \
enough information to fully answer the question, say so explicitly rather than guessing.

Context:
{context}

User question: {question}

Answer:
"""

NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the indexed documentation to answer this question "
    "confidently. The retrieved content didn't appear relevant even after rewriting the query. "
    "You may want to rephrase the question, or this topic may not be covered by the current "
    "document corpus."
)


def _format_context(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generation_node(state: GraphState) -> dict:
    graded_docs = state.get("graded_docs", [])
    question = state.get("original_question") or state["question"]

    if not graded_docs:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "sources": [],
        }

    query_type = state.get("query_type", "conceptual")
    style_hint = STYLE_HINTS.get(query_type, STYLE_HINTS["conceptual"])

    llm = get_llm()
    prompt = GENERATION_PROMPT.format(
        style_hint=style_hint,
        context=_format_context(graded_docs),
        question=question,
    )
    response = llm.invoke(prompt)

    sources = [
        {"source": doc.metadata.get("source", "unknown"), "chunk_id": doc.metadata.get("chunk_id", "")}
        for doc in graded_docs
    ]
    # De-duplicate sources while preserving order.
    seen = set()
    unique_sources = []
    for s in sources:
        key = s["source"]
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {"answer": response.content.strip(), "sources": unique_sources}
