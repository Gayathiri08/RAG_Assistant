"""
Integration tests for the LangGraph workflow, using FakeLLM/FakeEmbeddings so no
network access or API keys are required.
"""

from __future__ import annotations

from unittest.mock import patch

from app.graph.graph import build_graph
from tests.conftest import FakeLLM


def _run_graph(question: str, llm: FakeLLM, vector_store, k: int = 4):
    with (
        patch("app.graph.nodes_query_analysis.get_llm", return_value=llm),
        patch("app.graph.nodes_grading.get_llm", return_value=llm),
        patch("app.graph.nodes_generation.get_llm", return_value=llm),
        patch("app.graph.nodes_hallucination.get_llm", return_value=llm),
        patch("app.graph.nodes_retrieval.get_retriever", return_value=vector_store.as_retriever(search_kwargs={"k": k})),
    ):
        graph = build_graph()
        return graph.invoke(
            {"original_question": question, "question": question, "retry_count": 0}
        )


def test_happy_path_returns_grounded_answer_with_sources(fake_vector_store):
    result = _run_graph("How do I upload a file in FastAPI?", FakeLLM(grade_relevant=True), fake_vector_store)

    assert result["answer"]
    assert len(result["sources"]) > 0
    assert result["query_type"] == "how_to"
    assert result["retry_count"] == 0
    assert result["hallucination_check"]["grounded"] is True


def test_all_irrelevant_exhausts_retries_and_returns_idk(fake_vector_store):
    result = _run_graph(
        "What is the capital of France?", FakeLLM(grade_relevant=False), fake_vector_store
    )

    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
    assert result["retry_count"] == 2  # MAX_RETRIES default
    assert result["graded_docs"] == []
    # No context -> trivially grounded
    assert result["hallucination_check"]["grounded"] is True


def test_state_preserves_original_question_through_rewrites(fake_vector_store):
    original = "How do I upload a file in FastAPI?"
    result = _run_graph(original, FakeLLM(grade_relevant=True), fake_vector_store)

    assert result["original_question"] == original
    # question has been rewritten by query_analysis
    assert result["question"] != original or result["question"] == "how to upload files in FastAPI"
