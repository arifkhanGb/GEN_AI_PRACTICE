"""
app/models/retrieval.py  —  STEP 7 data contracts

Retrieval output must include:
  - score (how similar)
  - text (what the LLM will read in Step 8)
  - citations (page / chunk / document)
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    text: str
    score: float = Field(..., description="Cosine similarity in [-1, 1], higher is better")
    page_numbers: list[int] = Field(default_factory=list)
    chunk_type: str = "text"
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    embedding_model: str
    hit_count: int
    hits: list[RetrievalHit] = Field(default_factory=list)
    # Context string ready to stuff into an LLM prompt (Step 8)
    context_for_llm: str = ""
    searched_document_ids: list[str] = Field(default_factory=list)
