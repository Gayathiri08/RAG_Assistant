"""Pydantic models for request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural language question")


class SourceItem(BaseModel):
    source: str
    chunk_id: str = ""


class HallucinationCheck(BaseModel):
    grounded: bool
    explanation: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    query_type: str
    retries_used: int
    hallucination_check: HallucinationCheck


class IngestUrlsRequest(BaseModel):
    urls: list[str] = Field(default_factory=list, description="URLs to fetch and add to the index")


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str = Field(..., pattern="^(up|down)$", description="'up' or 'down'")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int
