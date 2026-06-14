"""
Node 2: Retrieval.

Performs similarity search against the FAISS vector store using the (possibly rewritten)
query from the query-analysis node, and returns the top-k chunks with their source
metadata attached.
"""

from __future__ import annotations

from app.graph.llm import get_retriever
from app.graph.state import GraphState


def retrieval_node(state: GraphState) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": docs}
