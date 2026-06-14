"""
Shared pytest fixtures.

Provides a FakeEmbeddings (deterministic, no network/model download) and a configurable
FakeLLM so the graph can be tested end-to-end without GROQ_API_KEY or internet access.
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from app.config import CORPUS_DIR
from app.ingest import load_documents_from_dir, split_documents


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based fake embeddings -- no model download required."""

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.rand(384).tolist()


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """
    Minimal fake LLM. Routes based on prompt content so it can serve all four
    LLM-calling nodes (query_analysis, document_grading, generation,
    hallucination_check) with canned responses.
    """

    def __init__(self, grade_relevant: bool = True):
        self.grade_relevant = grade_relevant

    def invoke(self, prompt: str):
        if "rewritten_query" in prompt:
            return FakeResponse(
                '{"query_type": "how_to", "rewritten_query": "how to upload files in FastAPI"}'
            )
        if "grounded" in prompt:
            return FakeResponse('{"grounded": true, "explanation": "Matches context."}')
        if "relevant" in prompt and "irrelevant" in prompt:
            return FakeResponse("relevant" if self.grade_relevant else "irrelevant")
        return FakeResponse("FastAPI supports file uploads via UploadFile. [Source: 05_file_uploads.md]")


@pytest.fixture(scope="session")
def fake_vector_store():
    docs = load_documents_from_dir(CORPUS_DIR)
    chunks = split_documents(docs)
    return FAISS.from_documents(chunks, FakeEmbeddings())
