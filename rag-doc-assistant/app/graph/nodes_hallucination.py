"""
Bonus Node: Hallucination Check (Self-RAG style).

After generation, an LLM checks whether the generated answer is actually supported by the
retrieved context (`graded_docs`). This is a lightweight, single-pass groundedness check
-- not a full Self-RAG re-generation loop -- intentionally kept simple given the 2-day
scope.

Result is stored in `hallucination_check` as:
    {"grounded": bool, "explanation": str}

This runs at the end of the graph regardless of outcome. If `graded_docs` is empty (the
"I don't know" path), we skip the LLM call entirely and mark it grounded by definition --
there's nothing to hallucinate from when we already declined to answer.

The API layer surfaces `hallucination_check.grounded == False` as a warning to the user
alongside the answer, rather than blocking the response -- by this point retries are
exhausted, so silently failing would be worse than a flagged answer.
"""

from __future__ import annotations

import json

from app.graph.llm import get_llm
from app.graph.state import GraphState

HALLUCINATION_PROMPT = """You are a fact-checking assistant. Given a set of context documents and \
a generated answer, determine whether every factual claim in the answer is directly supported by \
the context. Minor rephrasing is fine; the concern is claims that are NOT present in or contradicted \
by the context (hallucinations).

Context:
{context}

Generated answer:
{answer}

Respond with ONLY a JSON object, no other text, in this exact format:
{{"grounded": true or false, "explanation": "one sentence explanation"}}
"""


def _format_context(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def hallucination_check_node(state: GraphState) -> dict:
    graded_docs = state.get("graded_docs", [])
    answer = state.get("answer", "")

    if not graded_docs:
        return {"hallucination_check": {"grounded": True, "explanation": "No context was used (I don't know response)."}}

    llm = get_llm()
    prompt = HALLUCINATION_PROMPT.format(context=_format_context(graded_docs), answer=answer)
    response = llm.invoke(prompt)

    result = {"grounded": True, "explanation": "Could not parse groundedness check; defaulting to grounded."}
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        result = {
            "grounded": bool(parsed.get("grounded", True)),
            "explanation": parsed.get("explanation", ""),
        }
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    return {"hallucination_check": result}
