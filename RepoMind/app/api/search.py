
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.chat import RepositorySearchRequest
from app.services.answer_generation_service import (
    generate_repository_answer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/repository")
async def search_repository(request: RepositorySearchRequest):
    try:
        result = await generate_repository_answer(
            repository_id=request.repository_id,
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "repository_id": request.repository_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Repository question answering failed")

        raise HTTPException(
            status_code=500,
            detail="Repository question answering failed",
        ) from exc
