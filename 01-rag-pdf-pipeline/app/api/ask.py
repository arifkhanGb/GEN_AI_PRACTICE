"""
app/api/ask.py — Step 8: full RAG question answering
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.generation.llm_client import LlmConfigError
from app.generation.rag_answer import answer_question

router = APIRouter(prefix="/ask", tags=["Step 8 — RAG Answer"])


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["What pressure must the wellhead hold during testing?"],
    )
    top_k: int = Field(3, ge=1, le=10)
    document_ids: list[str] | None = None
    min_score: float = Field(0.15, ge=-1.0, le=1.0)


@router.post("")
async def ask(body: AskRequest):
    """
    Full RAG loop with a REAL OpenAI Chat Completions call:
      retrieve top chunks → grounded LLM answer → citations
    """
    try:
        result = answer_question(
            body.question,
            top_k=body.top_k,
            document_ids=body.document_ids,
            min_score=body.min_score,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LlmConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc

    return {
        "message": "Real LLM answer generated from retrieved context.",
        "llm_provider": result.llm_provider,
        "llm_model": result.llm_model,
        "has_openai_key": bool(settings.openai_api_key),
        "result": result.model_dump(),
        "why_this_matters": (
            "The LLM only sees retrieved context. Good parsing/chunking/retrieval "
            "is what makes answers correct — the prompt enforces grounding."
        ),
        "pipeline_complete": True,
    }


@router.get("/health")
async def ask_health():
    key = (settings.openai_api_key or "").strip()
    return {
        "status": "ok",
        "step": 8,
        "feature": "RAG answer generation (OpenAI API)",
        "openai_configured": bool(key),
        "llm_model": settings.llm_model,
        "require_real_llm": settings.require_real_llm,
        "base_url": settings.openai_base_url,
    }
