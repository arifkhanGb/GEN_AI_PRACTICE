"""
app/parsers/pipeline.py

Orchestrates Steps 1–6:
  parse → tables → OCR → clean → chunk → embed + persist
"""

from __future__ import annotations

from pathlib import Path

from app.chunking.chunker import chunk_document
from app.core.config import settings
from app.embeddings.embedder import embed_chunk_bundle
from app.embeddings.vector_store import save_index
from app.models.chunk import ChunkBundle
from app.models.document import ParsedDocument
from app.models.embedding import EmbeddingIndex
from app.parsers.header_footer_cleaner import clean_headers_footers
from app.parsers.layout_parser import parse_pdf
from app.parsers.ocr_extractor import enrich_document_with_ocr
from app.parsers.table_extractor import enrich_document_with_tables


def ingest_pdf(
    file_path: str | Path,
    document_id: str | None = None,
    *,
    extract_tables: bool = True,
    run_ocr: bool = True,
    clean_boilerplate: bool = True,
) -> ParsedDocument:
    """Steps 1–4: produce a cleaned ParsedDocument."""
    path = Path(file_path)

    document = parse_pdf(path, document_id=document_id)

    if extract_tables:
        document = enrich_document_with_tables(document, path)

    if run_ocr:
        document = enrich_document_with_ocr(document, path)

    if clean_boilerplate:
        document = clean_headers_footers(document)

    return document


def ingest_and_chunk_pdf(
    file_path: str | Path,
    document_id: str | None = None,
    *,
    extract_tables: bool = True,
    run_ocr: bool = True,
    clean_boilerplate: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[ParsedDocument, ChunkBundle]:
    """Steps 1–5 end-to-end."""
    document = ingest_pdf(
        file_path,
        document_id=document_id,
        extract_tables=extract_tables,
        run_ocr=run_ocr,
        clean_boilerplate=clean_boilerplate,
    )
    bundle = chunk_document(
        document,
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        ),
    )
    return document, bundle


def ingest_chunk_and_embed_pdf(
    file_path: str | Path,
    document_id: str | None = None,
    *,
    extract_tables: bool = True,
    run_ocr: bool = True,
    clean_boilerplate: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist: bool = True,
) -> tuple[ParsedDocument, ChunkBundle, EmbeddingIndex, dict | None]:
    """
    Steps 1–6 end-to-end.

    Persist = upsert vectors into Qdrant (+ JSON debug dump).
    """
    document, bundle = ingest_and_chunk_pdf(
        file_path,
        document_id=document_id,
        extract_tables=extract_tables,
        run_ocr=run_ocr,
        clean_boilerplate=clean_boilerplate,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    index = embed_chunk_bundle(bundle)
    saved: dict | None = save_index(index) if persist else None
    return document, bundle, index, saved
