import inngest
from fastapi import APIRouter, status

from app.inngest.client import inngest_client
from app.schemas.repository import RepositoryIndexRequest

router = APIRouter(tags=["Repository Indexing"])


@router.post("/index", status_code=status.HTTP_202_ACCEPTED)
async def index_repository(
    request: RepositoryIndexRequest,
) -> dict[str, str]:
    event = inngest.Event(
        name="repo/index.requested",
        data={
            "repository_url": str(request.repository_url),
        },
    )

    await inngest_client.send(event)

    return {
        "status": "accepted",
        "repository_url": str(request.repository_url),
        "message": "Repository indexing request accepted.",
    }
