
import logging

import inngest
from fastapi import APIRouter, HTTPException, status

from app.inngest.client import inngest_client
from app.schemas.chat import ChatRequest, ChatAcceptedResponse
from app.services.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_chat_job(request: ChatRequest):
    job = job_store.create_job(
        message="Repository question queued for processing"
    )

    try:
        await inngest_client.send(
            inngest.Event(
                name="chat/question.requested",
                data={
                    "job_id": str(job.job_id),
                    "repository_id": request.repository_id,
                    "question": request.question,
                    "top_k": request.top_k,
                },
            )
        )

    except Exception as exc:
        logger.exception("Failed to enqueue repository chat job")

        job_store.update_job(
            job_id=job.job_id,
            status="failed",
            message="Failed to enqueue repository question",
            error="Unable to enqueue chat job",
        )

        raise HTTPException(
            status_code=503,
            detail="Unable to enqueue chat job",
        ) from exc

    return ChatAcceptedResponse(
        job_id=job.job_id,
        status="queued",
        message="Repository question accepted for processing",
    )


# This endpoint will:
# 1. Validate the user's question.
# 2. Create a job with queued status.
# 3. Send an event to Inngest.
# 4. Return the job ID without waiting for the LLM.