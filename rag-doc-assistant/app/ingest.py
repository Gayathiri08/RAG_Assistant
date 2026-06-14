"""
Document ingestion pipeline.

Responsibilities:
1. Load documents (Markdown / text / HTML) from the corpus directory, or fetch from URLs.
2. Split documents into overlapping chunks sized for technical content.
3. Embed chunks with a local sentence-transformers model (no API key required).
4. Persist embeddings + metadata to a FAISS vector store on disk.

Run as a standalone script:

    python -m app.ingest

It is idempotent: re-running it rebuilds the index from whatever is currently in CORPUS_DIR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR


def load_documents_from_dir(corpus_dir: Path) -> list[Document]:
    """Load every .md / .txt / .html file in corpus_dir into LangChain Documents."""
    documents: list[Document] = []

    if not corpus_dir.exists():
        return documents

    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".markdown", ".txt", ".html", ".htm"}:
            continue

        raw_text = path.read_text(encoding="utf-8", errors="ignore")

        if path.suffix.lower() in {".html", ".htm"}:
            soup = BeautifulSoup(raw_text, "html.parser")
            raw_text = soup.get_text(separator="\n")

        documents.append(
            Document(
                page_content=raw_text,
                metadata={"source": path.name, "path": str(path)},
            )
        )

    return documents


def fetch_documents_from_urls(urls: list[str]) -> list[Document]:
    """Fetch raw text/HTML content from a list of URLs and wrap each as a Document."""
    documents: list[Document] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [skip] Failed to fetch {url}: {exc}")
            continue

        content_type = response.headers.get("content-type", "")
        text = response.text

        if "html" in content_type:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator="\n")

        documents.append(Document(page_content=text, metadata={"source": url, "path": url}))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks.

    Chunking strategy:
    - Markdown files use MarkdownTextSplitter, which is header- and code-block aware, so a
      chunk doesn't get cut in the middle of a fenced code block or split a header from its
      body. This matters for technical docs where a code example loses all meaning if it's
      separated from the explanation around it.
    - Non-markdown files fall back to RecursiveCharacterTextSplitter with the same chunk
      size, which tries paragraph -> line -> word boundaries in order.
    - CHUNK_SIZE=800 characters (~150-200 tokens) keeps chunks small enough to be specific
      (good for "what does parameter X do" style questions) while staying large enough to
      contain a full code snippet plus a sentence or two of surrounding explanation.
    - CHUNK_OVERLAP=120 characters (~15%) preserves context across chunk boundaries so that
      a concept introduced just before a split isn't lost from the following chunk.
    """
    md_splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        splitter = md_splitter if source.lower().endswith((".md", ".markdown")) else text_splitter
        for i, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata["chunk_id"] = f"{source}::chunk_{i}"
            chunks.append(chunk)

    return chunks


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed chunks and build (or rebuild) the FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def ingest(corpus_dir: Path = CORPUS_DIR, extra_urls: list[str] | None = None) -> int:
    """Run the full ingestion pipeline and persist the vector store. Returns chunk count."""
    documents = load_documents_from_dir(corpus_dir)

    if extra_urls:
        documents.extend(fetch_documents_from_urls(extra_urls))

    if not documents:
        print(f"No documents found in {corpus_dir} (and no URLs provided).")
        return 0

    print(f"Loaded {len(documents)} document(s) from {corpus_dir}")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    vector_store = build_vector_store(chunks)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Saved FAISS index to {VECTOR_STORE_DIR}")

    return len(chunks)


def add_documents_to_existing_store(new_chunks: list[Document]) -> int:
    """Load the existing FAISS store, add new chunks, and persist. Returns new chunk count."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if (VECTOR_STORE_DIR / "index.faiss").exists():
        vector_store = FAISS.load_local(
            str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True
        )
        vector_store.add_documents(new_chunks)
    else:
        vector_store = FAISS.from_documents(new_chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    return len(new_chunks)


if __name__ == "__main__":
    n = ingest()
    if n == 0:
        sys.exit(1)
    print(f"Ingestion complete: {n} chunks indexed.")
