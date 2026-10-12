from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    JobAcceptedResponse,
    JobCreateRequest,
    JobStatusResponse,
)
from app.services.job_store import job_store

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(request: JobCreateRequest) -> JobAcceptedResponse:
    job = job_store.create_job(request.message)

    return JobAcceptedResponse(
        job_id=job.job_id,
        status="queued",
        message=job.message,
    )

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID):
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
    )