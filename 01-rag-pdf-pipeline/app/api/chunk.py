"""
app/api/chunk.py — Step 5 endpoints

WHY a separate /chunk router?
-----------------------------
You often want to:
  1) Re-chunk with different sizes without re-uploading / re-OCRing
  2) Inspect chunks independently before spending on embeddings
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.parsers.header_footer_cleaner import header_footer_stats
from app.parsers.ocr_extractor import image_ocr_stats
from app.parsers.pipeline import ingest_and_chunk_pdf

router = APIRouter(prefix="/chunk", tags=["Step 5 — Chunking"])


@router.post("/pdf")
async def chunk_uploaded_pdf(
    file: UploadFile = File(...),
    extract_tables: bool = Query(True),
    run_ocr: bool = Query(True),
    clean_boilerplate: bool = Query(True),
    chunk_size: int = Query(800, ge=200, le=4000),
    chunk_overlap: int = Query(120, ge=0, le=1000),
):
    """
    Upload PDF → Steps 1–4 → structure-aware chunking (Step 5).

    Inspect each chunk's text + page_numbers + chunk_type before Step 6.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    document_id = str(uuid.uuid4())
    dest: Path = settings.upload_dir / f"{document_id}_{file.filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(content)

    try:
        document, bundle = ingest_and_chunk_pdf(
            dest,
            document_id=document_id,
            extract_tables=extract_tables,
            run_ocr=run_ocr,
            clean_boilerplate=clean_boilerplate,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}") from exc

    parsed_path = settings.parsed_dir / f"{document_id}.json"
    parsed_path.write_text(
        json.dumps(document.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chunks_path = settings.chunks_dir / f"{document_id}_chunks.json"
    chunks_path.write_text(
        json.dumps(bundle.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    preview = [
        {
            "chunk_id": c.chunk_id,
            "chunk_type": c.chunk_type,
            "pages": c.page_numbers,
            "chars": c.char_count,
            "token_estimate": c.token_estimate,
            "text_preview": c.text[:240],
        }
        for c in bundle.chunks
    ]

    return {
        "message": "Step 5 complete: structure-aware chunks ready for embedding.",
        "document_id": document_id,
        "page_count": document.page_count,
        "block_count": document.block_count(),
        "chunk_count": bundle.chunk_count,
        "chunk_size": bundle.chunk_size,
        "chunk_overlap": bundle.chunk_overlap,
        "strategy": bundle.strategy,
        "header_footer_stats": header_footer_stats(document),
        "ocr_stats": image_ocr_stats(document),
        "chunks_preview": preview,
        "saved_parsed_json": str(parsed_path),
        "saved_chunks_json": str(chunks_path),
        "chunks": bundle.model_dump(),
        "why_this_matters": (
            "Chunk boundaries decide what the retriever can find. "
            "Bad splits = missing answers even with a strong LLM."
        ),
        "next_step_hint": (
            "Open the chunks JSON and read a few chunks aloud. "
            "If a chunk feels incomplete or mixed-up, retune size/overlap. "
            "When ready, say CONTINUE for Step 6: Embeddings."
        ),
    }


@router.get("/health")
async def chunk_health():
    return {
        "status": "ok",
        "step": 5,
        "feature": "structure-aware chunking with overlap",
        "defaults": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    }
