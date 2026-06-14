"""
Graph assembly: wires the nodes into a LangGraph StateGraph with conditional routing.

Flow:

    query_analysis -> retrieval -> document_grading --(relevant docs found)--> generation -> hallucination_check -> END
                                                       |
                                                       (no relevant docs AND retries remain)
                                                       v
                                                  increment_retry -> query_analysis (loop)
                                                       |
                                                       (no relevant docs AND retries exhausted)
                                                       v
                                                  generation ("I don't know") -> hallucination_check -> END

Conditional edge (`route_after_grading`) is the self-corrective heart of the pipeline:
- If `graded_docs` is non-empty -> proceed to generation.
- If `graded_docs` is empty and `retry_count < MAX_RETRIES` -> go back through
  increment_retry -> query_analysis (rewrite the query and re-retrieve).
- If `graded_docs` is empty and retries are exhausted -> proceed to generation anyway,
  which detects the empty list and returns the "I don't know" response.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import MAX_RETRIES
from app.graph.nodes_generation import generation_node
from app.graph.nodes_grading import document_grading_node
from app.graph.nodes_hallucination import hallucination_check_node
from app.graph.nodes_query_analysis import query_analysis_node
from app.graph.nodes_retrieval import retrieval_node
from app.graph.nodes_retry import increment_retry_node
from app.graph.state import GraphState


def route_after_grading(state: GraphState) -> str:
    graded_docs = state.get("graded_docs", [])
    retry_count = state.get("retry_count", 0)

    if graded_docs:
        return "generation"

    if retry_count < MAX_RETRIES:
        return "increment_retry"

    return "generation"  # exhausted retries -> generation handles empty-context "I don't know"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grading", document_grading_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    workflow.set_entry_point("query_analysis")

    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grading")

    workflow.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generation": "generation",
            "increment_retry": "increment_retry",
        },
    )

    workflow.add_edge("increment_retry", "query_analysis")
    workflow.add_edge("generation", "hallucination_check")
    workflow.add_edge("hallucination_check", END)

    return workflow.compile()


# Module-level compiled graph, built lazily on first import use via get_compiled_graph()
# rather than at import time, so importing this module doesn't require GROQ_API_KEY to
# be set (useful for tests that only check graph structure).
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
