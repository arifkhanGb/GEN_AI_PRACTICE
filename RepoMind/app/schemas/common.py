from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "completed", "failed"]


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=200)


class JobAcceptedResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    message: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    result: dict | None = None
    error: str | None = None
