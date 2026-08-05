"""
app/api/embed.py — Step 6 endpoints

Upload PDF → Steps 1–5 → embed chunks → upsert into Qdrant
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.embeddings.qdrant_store import collection_info
from app.embeddings.vector_store import list_indexes
from app.parsers.pipeline import ingest_chunk_and_embed_pdf

router = APIRouter(prefix="/embed", tags=["Step 6 — Embeddings + Qdrant"])


@router.post("/pdf")
async def embed_uploaded_pdf(
    file: UploadFile = File(...),
    extract_tables: bool = Query(True),
    run_ocr: bool = Query(False, description="OCR is slower; enable when images matter"),
    clean_boilerplate: bool = Query(True),
    chunk_size: int = Query(800, ge=200, le=4000),
    chunk_overlap: int = Query(120, ge=0, le=1000),
):
    """
    Full ingest through embeddings, then upsert vectors into Qdrant.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    document_id = str(uuid.uuid4())
    dest = settings.upload_dir / f"{document_id}_{file.filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(content)

    try:
        document, bundle, index, saved = ingest_chunk_and_embed_pdf(
            dest,
            document_id=document_id,
            extract_tables=extract_tables,
            run_ocr=run_ocr,
            clean_boilerplate=clean_boilerplate,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    (settings.parsed_dir / f"{document_id}.json").write_text(
        json.dumps(document.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (settings.chunks_dir / f"{document_id}_chunks.json").write_text(
        json.dumps(bundle.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    preview = [
        {
            "chunk_id": item.chunk_id,
            "chunk_type": item.chunk_type,
            "pages": item.page_numbers,
            "embedding_dim": item.embedding_dim,
            "vector_preview_first_8": item.embedding[:8],
            "text_preview": item.text[:200],
        }
        for item in index.items[:8]
    ]

    return {
        "message": "Chunks embedded and upserted into Qdrant vector DB.",
        "document_id": document_id,
        "chunk_count": index.chunk_count,
        "embedding_model": index.embedding_model,
        "embedding_dim": index.embedding_dim,
        "vector_db": "qdrant",
        "persist_result": saved,
        "qdrant_collection": collection_info(),
        "preview": preview,
        "why_this_matters": (
            "Qdrant stores vectors + payload and can ANN-search at scale. "
            "JSON dumps remain only for human debugging."
        ),
    }


@router.get("/indexes")
async def get_indexes():
    return {
        "indexes": list_indexes(),
        "qdrant": collection_info(),
    }


@router.get("/health")
async def embed_health():
    return {
        "status": "ok",
        "step": 6,
        "feature": "fastembed + Qdrant upsert",
        "model": settings.embedding_model,
        "qdrant_mode": settings.qdrant_mode,
        "qdrant": collection_info(),
    }
