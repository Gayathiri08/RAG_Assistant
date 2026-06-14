"""
Shared singletons: LLM client and vector store retriever.

Kept in one module so every graph node imports the same instances instead of each
re-initializing models (slow, and would defeat any in-process caching).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL, TOP_K, VECTOR_STORE_DIR


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://console.groq.com/keys)."
        )
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store() -> FAISS:
    """
    Load the FAISS index from disk.

    Not cached at module level (unlike the LLM/embeddings) because the /ingest endpoint
    can rebuild the index at runtime; callers should get a fresh handle each time they
    need to retrieve.
    """
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        raise RuntimeError(
            f"No vector store found at {VECTOR_STORE_DIR}. "
            "Run `python -m app.ingest` first, or call POST /ingest."
        )
    return FAISS.load_local(
        str(VECTOR_STORE_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def get_retriever(k: int = TOP_K):
    return load_vector_store().as_retriever(search_kwargs={"k": k})
