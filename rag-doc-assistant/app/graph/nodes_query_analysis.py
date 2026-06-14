"""
Node 1: Query Analysis.

Takes the user's raw question and:
- Classifies it into one of: conceptual | how_to | troubleshooting | api_reference
- Produces a "retrieval query": the original question optionally expanded with synonyms /
  clarified terms to improve similarity search against the FastAPI docs corpus.

On the *first* pass through this node, `question` is set to `original_question`. On
retries (after a failed grading round), this node is called again but with a hint that
the previous retrieval failed, so it should rewrite more aggressively (e.g. drop overly
specific phrasing, add likely FastAPI-specific terminology).
"""

from __future__ import annotations

import json

from app.graph.llm import get_llm
from app.graph.state import GraphState

CLASSIFY_AND_REWRITE_PROMPT = """You are a query analysis assistant for a technical documentation \
search system about FastAPI (a Python web framework).

Given a user's question, do two things:
1. Classify it into exactly one category: "conceptual", "how_to", "troubleshooting", or \
"api_reference".
2. Rewrite the question into a search-optimized query: expand abbreviations, add likely \
FastAPI-specific terms/synonyms (e.g. "endpoint" -> also consider "path operation"), and remove \
conversational filler. Keep it concise (one sentence).

{retry_hint}

User question: {question}

Respond with ONLY a JSON object, no other text, in this exact format:
{{"query_type": "...", "rewritten_query": "..."}}
"""


def query_analysis_node(state: GraphState) -> dict:
    is_retry = state.get("retry_count", 0) > 0
    retry_hint = (
        "NOTE: A previous search with a similar query returned no relevant results. "
        "Rewrite the query more broadly or with different terminology this time."
        if is_retry
        else ""
    )

    # Always rewrite from the original question, not the previous rewrite, to avoid
    # compounding drift across retries.
    base_question = state.get("original_question") or state["question"]

    llm = get_llm()
    prompt = CLASSIFY_AND_REWRITE_PROMPT.format(retry_hint=retry_hint, question=base_question)
    response = llm.invoke(prompt)

    query_type = "conceptual"
    rewritten_query = base_question

    try:
        content = response.content.strip()
        # Strip markdown code fences if the model adds them despite instructions.
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        query_type = parsed.get("query_type", query_type)
        rewritten_query = parsed.get("rewritten_query", rewritten_query) or rewritten_query
    except (json.JSONDecodeError, AttributeError, KeyError):
        # Fall back gracefully: use the original question verbatim, default category.
        pass

    return {
        "original_question": base_question,
        "question": rewritten_query,
        "query_type": query_type,
    }
