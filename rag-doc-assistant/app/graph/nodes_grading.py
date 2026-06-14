"""
Node 3: Document Grading.

The self-corrective component of the pipeline. For each retrieved chunk, an LLM grades
it as "relevant" or "irrelevant" to the (original) user question. Irrelevant chunks are
filtered out before generation.

This node does NOT decide the routing itself -- it just populates `graded_docs` and
increments nothing. The actual retry/proceed decision is a conditional edge
(see app/graph/graph.py: route_after_grading) which reads `graded_docs` and
`retry_count`.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.graph.llm import get_llm
from app.graph.state import GraphState

GRADE_PROMPT = """You are a relevance grader. Determine whether the following document chunk \
contains information that helps answer the user's question.

User question: {question}

Document chunk:
\"\"\"
{chunk}
\"\"\"

Respond with a single word: "relevant" or "irrelevant". No explanation.
"""


def _grade_chunk(question: str, chunk: Document) -> bool:
    llm = get_llm()
    prompt = GRADE_PROMPT.format(question=question, chunk=chunk.page_content)
    response = llm.invoke(prompt)
    verdict = response.content.strip().lower()
    return "relevant" in verdict and "irrelevant" not in verdict


def document_grading_node(state: GraphState) -> dict:
    question = state.get("original_question") or state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    graded_docs = [doc for doc in retrieved_docs if _grade_chunk(question, doc)]

    return {"graded_docs": graded_docs}
