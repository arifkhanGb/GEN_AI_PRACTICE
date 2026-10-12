
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.schemas.common import JobStatus


class JobRecord(BaseModel):
    job_id: UUID
    status: JobStatus
    message: str
    created_at: datetime
    updated_at: datetime

    result: dict | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[UUID, JobRecord] = {}
        self._lock = RLock()

    def create_job(self, message: str) -> JobRecord:
        now = datetime.now(UTC)

        job = JobRecord(
            job_id=uuid4(),
            status="queued",
            message=message,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._jobs[job.job_id] = job

        return job.model_copy(deep=True)

    def get_job(self, job_id: UUID) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def update_job(
        self,
        job_id: UUID,
        status: JobStatus,
        message: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            updated = job.model_copy(
                update={
                    "status": status,
                    "message": message,
                    "updated_at": datetime.now(UTC),
                    "result": result if result is not None else job.result,
                    "error": error,
                }
            )

            self._jobs[job_id] = updated
            return updated.model_copy(deep=True)


job_store = JobStore()
