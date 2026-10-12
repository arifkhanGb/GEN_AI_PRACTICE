from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

class RepositorySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_id: str = Field(
        min_length=1,
        description="Repository identifier, e.g. owner/repo",
    )
    question: str = Field(
        min_length=1,
        description="Question to search for in the repository",
    )
    top_k: int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_id: str = Field(
        min_length=1,
        max_length=200,
        description="Repository namespace used during indexing, e.g. owner/repo",
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask about the repository",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class ChatAcceptedResponse(BaseModel):
    job_id: UUID
    status: str
    message: str


class ChatJobResult(BaseModel):
    answer: str
    sources: list[dict]


class ChatJobStatusResponse(BaseModel):
    job_id: UUID
    status: str
    message: str
    result: ChatJobResult | None = None
    error: str | None = None
