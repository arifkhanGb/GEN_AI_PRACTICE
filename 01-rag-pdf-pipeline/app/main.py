"""
app/main.py — FastAPI application entrypoint

Full pipeline:
  Parse → Tables → OCR → Clean → Chunk → Embed → Retrieve → Ask
"""

from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.chunk import router as chunk_router
from app.api.embed import router as embed_router
from app.api.parse import router as parse_router
from app.api.retrieve import router as retrieve_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-style RAG pipeline built step-by-step.\n\n"
        "Pipeline order: Document Parsing → Chunking → Embeddings → Retrieval → LLM"
    ),
)

app.include_router(parse_router)
app.include_router(chunk_router)
app.include_router(embed_router)
app.include_router(retrieve_router)
app.include_router(ask_router)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "current_step": 8,
        "current_focus": "RAG with Qdrant vector DB + OpenAI answers",
        "pipeline": [
            "1. Document Parsing (done)",
            "2. Table Extraction (done)",
            "3. Image OCR (done)",
            "4. Header/Footer Cleaning (done)",
            "5. Chunking (done)",
            "6. Embeddings + Qdrant upsert (done)",
            "7. Retrieval via Qdrant ANN (done)",
            "8. LLM Answer Generation (done)",
        ],
        "vector_db": {
            "name": "qdrant",
            "mode": settings.qdrant_mode,
            "collection": settings.qdrant_collection,
        },
        "endpoints": {
            "parse": "POST /parse/pdf",
            "chunk": "POST /chunk/pdf",
            "embed": "POST /embed/pdf",
            "retrieve": "POST /retrieve/query",
            "ask": "POST /ask",
        },
        "docs": "/docs",
    }
