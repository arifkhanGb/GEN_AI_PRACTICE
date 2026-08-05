"""
app/api/retrieve.py — Step 7 endpoints (Qdrant ANN search)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.embeddings.qdrant_store import collection_info
from app.embeddings.vector_store import list_indexes
from app.retrieval.retriever import retrieve

router = APIRouter(prefix="/retrieve", tags=["Step 7 — Retrieval (Qdrant)"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What pressure is required for the wellhead test?"])
    top_k: int = Field(3, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional Qdrant payload filter on document_id",
    )
    min_score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Drop hits below this cosine similarity",
    )


@router.post("/query")
async def retrieve_query(body: RetrieveRequest):
    """Semantic retrieval via Qdrant cosine ANN search."""
    try:
        result = retrieve(
            body.query,
            top_k=body.top_k,
            document_ids=body.document_ids,
            min_score=body.min_score,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    return {
        "message": "Top chunks retrieved from Qdrant.",
        "vector_db": "qdrant",
        "qdrant": collection_info(),
        "result": result.model_dump(),
        "why_this_matters": (
            "The LLM can only answer from retrieved chunks. "
            "Qdrant finds the nearest vectors fast, even as data grows."
        ),
    }


@router.get("/indexes")
async def retrieve_indexes():
    return {"indexes": list_indexes(), "qdrant": collection_info()}


@router.get("/health")
async def retrieve_health():
    return {
        "status": "ok",
        "step": 7,
        "feature": "Qdrant cosine ANN retrieval",
        "indexed_documents": len(list_indexes()),
        "qdrant": collection_info(),
    }
