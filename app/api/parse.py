"""
app/api/parse.py — FastAPI routes for document parsing (Steps 1–4)

Ingestion:
  Upload → Layout → Tables → OCR → Header/Footer clean → (next: chunk)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.parsers.header_footer_cleaner import header_footer_stats
from app.parsers.ocr_extractor import image_ocr_stats
from app.parsers.pipeline import ingest_pdf

router = APIRouter(prefix="/parse", tags=["Steps 1–4 — Parse through cleaning"])


@router.post("/pdf")
async def parse_uploaded_pdf(
    file: UploadFile = File(...),
    extract_tables: bool = Query(True, description="Step 2: structured tables"),
    run_ocr: bool = Query(True, description="Step 3: OCR images"),
    clean_boilerplate: bool = Query(
        True,
        description="Step 4: remove repeated headers/footers before chunking",
    ),
):
    """
    Upload a PDF and run Steps 1–4.

    WHY Step 4 exists:
    Repeated headers/footers would otherwise appear in almost every chunk,
    polluting embeddings and wasting the LLM context window.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    document_id = str(uuid.uuid4())
    dest: Path = settings.upload_dir / f"{document_id}_{file.filename}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(content)

    try:
        parsed = ingest_pdf(
            dest,
            document_id=document_id,
            extract_tables=extract_tables,
            run_ocr=run_ocr,
            clean_boilerplate=clean_boilerplate,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse failed: {exc}") from exc

    out_path = settings.parsed_dir / f"{document_id}.json"
    out_path.write_text(
        json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table_blocks = [
        {
            "block_id": b.block_id,
            "page": b.page_number,
            "headers": b.table.headers if b.table else [],
            "row_count": b.table.row_count if b.table else 0,
            "markdown_preview": (b.text[:500] if b.text else ""),
        }
        for page in parsed.pages
        for b in page.blocks
        if b.block_type.value == "table"
    ]

    return {
        "message": "Steps 1–4 complete (layout + tables + OCR + header/footer clean).",
        "document_id": parsed.document_id,
        "source_filename": parsed.source_filename,
        "page_count": parsed.page_count,
        "block_count": parsed.block_count(),
        "table_count": parsed.table_count(),
        "ocr_stats": image_ocr_stats(parsed),
        "header_footer_stats": header_footer_stats(parsed),
        "parser_version": parsed.parser_version,
        "tables_summary": table_blocks,
        "saved_json": str(out_path),
        "document": parsed.model_dump(),
        "why_this_matters": (
            "Cleaning headers/footers before chunking prevents boilerplate from "
            "dominating embeddings and retrieval rankings."
        ),
        "next_step_hint": (
            "Parsing/cleaning done. Use POST /chunk/pdf for Step 5, "
            "or say CONTINUE if you already finished chunking."
        ),
    }


@router.get("/health")
async def parse_health():
    return {
        "status": "ok",
        "steps_enabled": [1, 2, 3, 4],
        "features": [
            "layout-aware PDF parsing",
            "structured table extraction",
            "image OCR",
            "repeated header/footer cleaning",
        ],
    }
