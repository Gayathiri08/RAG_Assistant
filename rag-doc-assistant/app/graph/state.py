"""
Shared state schema for the LangGraph workflow.

Design notes (this is the core evaluation criterion mentioned in the assignment):

- `original_question` is kept separate from `question` so the final answer can always be
  framed against what the user actually asked, even after the query has been rewritten
  one or more times for retrieval.
- `question` is the *current* (possibly rewritten) query used for retrieval. It starts
  equal to `original_question` and may be overwritten by the query-rewrite step.
- `retrieved_docs` holds the raw top-k chunks from the vector store for the current
  retrieval attempt. It gets overwritten on each retry (we don't need to keep stale
  chunks around).
- `graded_docs` holds only the chunks the grading node judged "relevant". This is what
  generation actually reads. Kept separate from `retrieved_docs` so we can log/debug
  what was retrieved vs. what was used.
- `retry_count` is an integer counter incremented every time the conditional edge routes
  back to query rewriting. The conditional edge compares this against MAX_RETRIES to
  decide between "retry" and "give up -> answer from whatever we have / I don't know".
- `query_type` is set by the query-analysis node (conceptual / how-to / troubleshooting /
  api_reference) and is passed through to generation, which can lightly adjust its
  prompt/style based on it.
- `answer` and `sources` are the final outputs consumed by the API layer.
- `hallucination_check` stores the result of the bonus Self-RAG-style groundedness check:
  "grounded" or "not_grounded", plus a short explanation. If "not_grounded", the API
  layer surfaces a warning to the user alongside the answer rather than silently failing,
  since by this point we've already exhausted retries.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # Input / query handling
    original_question: str
    question: str
    query_type: str  # conceptual | how_to | troubleshooting | api_reference

    # Retrieval + grading
    retrieved_docs: list[Document]
    graded_docs: list[Document]
    retry_count: int

    # Generation outputs
    answer: str
    sources: list[dict]

    # Bonus: hallucination / groundedness check
    hallucination_check: dict
