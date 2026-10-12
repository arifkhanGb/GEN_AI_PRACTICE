
import inngest
from urllib.parse import urlparse
from app.inngest.client import inngest_client
from app.services.github import fetch_repository_files
from app.services.file_filter import filter_repository_files
from app.services.repository_loader import load_repository_files
from app.services.chunking_service import chunk_repository_files
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import upsert_chunks
from uuid import UUID


@inngest_client.create_function(
    fn_id="hello-world",
    trigger=inngest.TriggerEvent(event="demo/hello.world"),
)
async def hello_world(ctx: inngest.Context) -> dict[str, str]:
    return {
        "message": "Hello World from RepoMind!",
    }


@inngest_client.create_function(
    fn_id="index-repository",
    trigger=inngest.TriggerEvent(event="repo/index.requested"),
)
async def index_repository(ctx: inngest.Context) -> dict[str, str]:
    repository_url = str(ctx.event.data["repository_url"])
    
     # Extract owner and repository name from the URL.
    parsed_url = urlparse(repository_url)
    parts = parsed_url.path.strip("/").split("/")
    
    if (
    parsed_url.hostname not in {"github.com", "www.github.com"}
    or len(parts) != 2
    ):
        
        raise ValueError("Invalid GitHub repository URL")   
    
    owner = parts[0]
    repository = parts[1].removesuffix(".git")

     # Stage 1: Fetch the repository file tree.
    files = await fetch_repository_files(owner, repository)
    
     # Stage 2: Filter relevant files.
    filtered_files = filter_repository_files(files)
    
      # Stage 3: Download the first 10 filtered files.
    loaded_files = await load_repository_files(
        owner=owner,
        repository=repository,
        files=filtered_files,
        limit=10,
    )

    repository_chunks = chunk_repository_files(loaded_files)
    
    # Generate embeddings in batches to avoid oversized API requests.
    embedding_batch_size = 50
    total_vectors_stored = 0

    repository_id = f"{owner}/{repository}"

    for start in range(0, len(repository_chunks), embedding_batch_size):
        batch = repository_chunks[start : start + embedding_batch_size]

        texts = [chunk["content"] for chunk in batch]
        embeddings = await generate_embeddings(texts)

        stored_count = upsert_chunks(
            repository_id=repository_id,
            chunks=batch,
            embeddings=embeddings,
        )

        total_vectors_stored += stored_count

    return {
        "status": "completed",
        "repository_url": repository_url,
        "owner": owner,
        "repository": repository,
        "total_files": len(files),
        "filtered_files": len(filtered_files),
        "downloaded_files": len(loaded_files),
        "total_chunks": len(repository_chunks),
        "repository_id": repository_id,
        "vectors_stored": total_vectors_stored,
        "sample_chunks": [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "content_preview": chunk["content"][:100],
            }
            for chunk in repository_chunks[:5]
        ],
    }
    
    
from app.services.answer_generation_service import (
    generate_repository_answer,
)
from app.services.job_store import job_store


@inngest_client.create_function(
    fn_id="answer-repository-question",
    trigger=inngest.TriggerEvent(
        event="chat/question.requested"
    ),
)
async def answer_repository_question(ctx: inngest.Context):
    event_data = ctx.event.data

    job_id = UUID(event_data["job_id"])
    repository_id = event_data["repository_id"]
    question = event_data["question"]
    top_k = event_data.get("top_k", 5)

    job = job_store.update_job(
        job_id=job_id,
        status="running",
        message="Retrieving repository context and generating answer",
    )

    if job is None:
        raise ValueError(f"Chat job not found: {job_id}")

    try:
        result = await ctx.step.run(
            "generate-repository-answer",
            lambda: generate_repository_answer(
                repository_id=repository_id,
                question=question,
                top_k=top_k,
            ),
        )

        job_store.update_job(
            job_id=job_id,
            status="completed",
            message="Repository answer generated successfully",
            result=result,
        )

        return {
            "job_id": str(job_id),
            "status": "completed",
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception:
        job_store.update_job(
            job_id=job_id,
            status="failed",
            message="Repository question processing failed",
            error="Unable to generate an answer. Check the server logs.",
        )
        raise    