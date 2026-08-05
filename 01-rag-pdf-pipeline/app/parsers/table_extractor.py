"""
app/parsers/table_extractor.py  —  STEP 2 CORE

============================================================
LESSON: Extract tables as STRUCTURE, not scrambled text
============================================================

The LinkedIn lesson you shared is exactly this problem:

  Bad extract:  "Q1 100 Q2 150 Q3 200 Q4 250"
  Good extract: a real table with headers + rows

WHY a dedicated table step?
- Step 1 gives layout blocks (great for paragraphs)
- Table CELL boundaries are a different problem
- We need rows × columns preserved for retrieval quality

Library choice (important for your machine):
------------------------------------------------
Many tutorials use pdfplumber. That is a solid choice.
On THIS environment, pdfplumber → pdfminer → cryptography hit a
Windows Application Control DLL block, so we use PyMuPDF's native
table finder instead:

    page.find_tables()

Same production idea. Different engine. Structure still preserved.

CRITICAL RAG bug beginners hit:
  If you keep BOTH the scrambled text blocks AND the table block,
  the same numbers get embedded twice → noisy retrieval.

So we:
  - Detect tables
  - Insert TABLE ContentBlocks (markdown text + TableData)
  - REMOVE overlapping plain TEXT blocks that live inside the table bbox
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.models.document import (
    BlockType,
    BoundingBox,
    ContentBlock,
    ParsedDocument,
    TableData,
)


def _clean_cell(value: Any) -> str:
    """Normalize a single cell to a clean string."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    return text


def _bbox_overlap_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    """
    How much of `inner`'s area sits inside `outer`?

    WHY overlap ratio (not just "touches")?
    - A caption near a table should stay as TEXT
    - Only remove blocks that are MOSTLY inside the table region
    """
    x0 = max(inner.x0, outer.x0)
    y0 = max(inner.y0, outer.y0)
    x1 = min(inner.x1, outer.x1)
    y1 = min(inner.y1, outer.y1)

    iw = max(0.0, x1 - x0)
    ih = max(0.0, y1 - y0)
    intersection = iw * ih
    inner_area = max(inner.width * inner.height, 1e-6)
    return intersection / inner_area


def _rows_to_table_data(raw_table: list[list[Any]]) -> TableData:
    """
    Convert list-of-lists into our TableData contract.

    Convention (industry-common default):
    - First non-empty row → headers
    - Remaining rows → data

    Enterprise PDFs can have multi-row headers / merged cells.
    Start simple; harden later when you see real messy docs.
    """
    cleaned: list[list[str]] = []
    for row in raw_table or []:
        cleaned.append([_clean_cell(c) for c in (row or [])])

    cleaned = [r for r in cleaned if any(cell for cell in r)]
    if not cleaned:
        return TableData()

    headers = cleaned[0]
    data_rows = cleaned[1:] if len(cleaned) > 1 else []
    col_count = max((len(r) for r in cleaned), default=0)

    return TableData(
        headers=headers,
        rows=data_rows,
        column_count=col_count,
        row_count=len(data_rows),
    )


def extract_tables_from_pdf(
    file_path: str | Path,
) -> dict[int, list[tuple[BoundingBox, TableData]]]:
    """
    Find all tables in a PDF using PyMuPDF find_tables().

    Returns
    -------
    dict keyed by 1-based page number:
        page_number -> list of (table_bbox, TableData)
    """
    path = Path(file_path)
    found: dict[int, list[tuple[BoundingBox, TableData]]] = {}

    doc = fitz.open(path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            page_tables: list[tuple[BoundingBox, TableData]] = []

            # PyMuPDF table finder (available in modern pymupdf)
            try:
                detector = page.find_tables()
            except Exception:
                # Older builds / odd pages — fail soft so Step 1 text still works
                continue

            tables = getattr(detector, "tables", None) or []
            for table_obj in tables:
                bbox_vals = table_obj.bbox  # (x0, y0, x1, y1)
                bbox = BoundingBox(
                    x0=float(bbox_vals[0]),
                    y0=float(bbox_vals[1]),
                    x1=float(bbox_vals[2]),
                    y1=float(bbox_vals[3]),
                )

                raw = table_obj.extract()
                table_data = _rows_to_table_data(raw)

                # Skip tiny/empty false positives
                if table_data.column_count < 2 and table_data.row_count < 1:
                    continue
                if not table_data.headers and not table_data.rows:
                    continue

                page_tables.append((bbox, table_data))

            if page_tables:
                found[page_number] = page_tables
    finally:
        doc.close()

    return found


def enrich_document_with_tables(
    document: ParsedDocument,
    file_path: str | Path,
    overlap_threshold: float = 0.55,
) -> ParsedDocument:
    """
    STEP 2 public API: merge structured tables into a Step-1 ParsedDocument.

    1. Detect tables
    2. Drop TEXT blocks that mostly sit inside a table bbox (dedupe)
    3. Insert TABLE ContentBlocks with markdown + TableData
    4. Re-sort by reading order
    """
    tables_by_page = extract_tables_from_pdf(file_path)

    for page in document.pages:
        page_tables = tables_by_page.get(page.page_number, [])
        if not page_tables:
            continue

        table_bboxes = [bbox for bbox, _ in page_tables]

        kept_blocks: list[ContentBlock] = []
        for block in page.blocks:
            if block.block_type != BlockType.TEXT:
                kept_blocks.append(block)
                continue

            overlaps_table = any(
                _bbox_overlap_ratio(block.bbox, tb) >= overlap_threshold
                for tb in table_bboxes
            )
            if overlaps_table:
                continue  # represented better by the TABLE block
            kept_blocks.append(block)

        for t_index, (bbox, table_data) in enumerate(page_tables):
            markdown = table_data.to_markdown()
            kv_lines = table_data.to_key_value_lines()

            table_block = ContentBlock(
                block_id=f"p{page.page_number}_table{t_index}",
                block_type=BlockType.TABLE,
                page_number=page.page_number,
                bbox=bbox,
                text=markdown,  # what chunking/embeddings will primarily see
                table=table_data,
                metadata={
                    "format": "markdown_table",
                    "extractor": "pymupdf_find_tables",
                    "key_value_text": kv_lines,
                    "header_count": len(table_data.headers),
                    "data_row_count": table_data.row_count,
                    "column_count": table_data.column_count,
                    # Teaching breadcrumb: naive vs structured
                    "naive_flat_example": " ".join(
                        table_data.headers
                        + [c for row in table_data.rows for c in row]
                    ),
                },
            )
            kept_blocks.append(table_block)

        page.blocks = sorted(kept_blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

    document.parser_name = "pymupdf_layout+pymupdf_tables"
    document.parser_version = "step2"
    return document
