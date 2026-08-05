"""
app/models/chunk.py  —  STEP 5 data contract

WHY a Chunk model (separate from ContentBlock)?
----------------------------------------------
ContentBlock = what the PDF contained (layout truth)
Chunk        = what we store/search in the vector DB (retrieval unit)

They are related but NOT the same:
  - One table block  → usually one chunk (keep structure intact)
  - One long section → many overlapping chunks
  - Tiny paragraphs → may be MERGED into one chunk

Production RAG fails when people embed raw pages or random 500-char slices
with no metadata. Metadata is how you cite sources later ("page 3, table").
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    One retrieval unit that will be embedded in Step 6.

    text          — the content the embedding model sees
    chunk_id      — stable id for updates/deletes in a vector DB
    document_id   — which PDF this came from
    page_numbers  — citation support (can span pages if merged)
    chunk_type    — text | table | image_ocr (different retrieval behavior later)
    metadata      — anything else useful for filters / debugging
    """

    chunk_id: str
    document_id: str
    source_filename: str
    text: str
    chunk_index: int = Field(..., ge=0)

    # Provenance / citations
    page_numbers: list[int] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)
    chunk_type: str = "text"  # text | table | image_ocr

    # Useful for debugging chunk quality
    char_count: int = 0
    token_estimate: int = 0  # rough ~4 chars/token heuristic for learning

    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_estimate:
            # WHY estimate? Real tokenizers differ by model.
            # For learning/config, chars/4 is a decent rule of thumb.
            self.token_estimate = max(1, len(self.text) // 4) if self.text else 0


class ChunkBundle(BaseModel):
    """All chunks produced for one document (Step 5 output)."""

    document_id: str
    source_filename: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[Chunk] = Field(default_factory=list)
    strategy: str = "structure_aware_overlap"
