from fastapi import FastAPI
import inngest.fast_api

from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.config import get_settings
from app.inngest.client import inngest_client
from app.api.index import router as index_router
from app.inngest.functions import hello_world , index_repository,  answer_repository_question
from app.api.search import router as search_router
from app.api.chat import router as chat_router

settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    description="AI-powered GitHub repository documentation and chat.",
    version=settings.app_version,
)

app.include_router(
    health_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    jobs_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    index_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    search_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    chat_router,
    prefix=settings.api_v1_prefix,
)

# Register Inngest functions
inngest.fast_api.serve(
    app,
    inngest_client,
    [
        hello_world,
        index_repository,
        answer_repository_question,
    ],
)

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "application": settings.app_name,
        "message": "Welcome to RepoMind API",
        "docs": "/docs",
    }
