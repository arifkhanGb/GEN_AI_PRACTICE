"""
app/models/answer.py  —  STEP 8 data contracts
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalHit


class RagAnswer(BaseModel):
    question: str
    answer: str
    # Where the answer came from (for trust / debugging)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_hits: list[RetrievalHit] = Field(default_factory=list)
    context_used: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    grounded: bool = True  # True = instructed to use ONLY retrieved context
    refusal: bool = False  # True if we said "not enough info in context"
