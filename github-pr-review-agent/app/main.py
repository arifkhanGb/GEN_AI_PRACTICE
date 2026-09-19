from fastapi import FastAPI
from app.inngest.client import inngest_client
from app.inngest.function import hello_world
from app.inngest.function import github_pr_review
import inngest
from inngest.fast_api import serve

app = FastAPI(
    title="GitHub PR Review Agent",
    description="AI-powered GitHub Pull Request Review Agent",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "message": "GitHub PR Review Agent is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }
    
    
# Expose Inngest functions to Inngest
serve(
    app,
    inngest_client,
    [
        hello_world,
        github_pr_review
        ],
)    
