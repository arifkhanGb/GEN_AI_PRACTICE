"""
app/models/embedding.py  —  STEP 6 data contracts

WHY store vectors WITH chunk metadata?
--------------------------------------
A naked float array is useless for RAG.
At retrieval time you need:
  - the original text (to send to the LLM)
  - page numbers (citations)
  - document_id / chunk_id (debugging, updates, deletes)

Production vector DBs (Chroma, Qdrant, Pinecone, pgvector) all store
"vector + payload". We mimic that contract locally so Step 7 is easy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    """One chunk + its embedding vector."""

    chunk_id: str
    document_id: str
    source_filename: str
    text: str
    chunk_index: int
    page_numbers: list[int] = Field(default_factory=list)
    chunk_type: str = "text"
    embedding: list[float] = Field(default_factory=list)
    embedding_model: str = ""
    embedding_dim: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingIndex(BaseModel):
    """
    Local index for one document (or a batch).

    Step 7 will load this and run cosine similarity search.
    Later you can swap the backend for Chroma/Qdrant without changing
    the EmbeddedChunk contract.
    """

    document_id: str
    source_filename: str
    embedding_model: str
    embedding_dim: int
    chunk_count: int
    items: list[EmbeddedChunk] = Field(default_factory=list)
