"""
Central configuration for the RAG Technical Documentation Assistant.

All tunables (model names, paths, retry limits) live here so they can be changed
in one place and are loaded from environment variables / .env at import time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "./vector_store").lstrip("./")
CORPUS_DIR = BASE_DIR / os.getenv("CORPUS_DIR", "./corpus").lstrip("./")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Chunking configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
