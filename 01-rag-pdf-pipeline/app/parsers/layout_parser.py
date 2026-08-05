"""
app/parsers/layout_parser.py  —  STEP 1 CORE

============================================================
LESSON: Layout-aware PDF parsing
============================================================

Naive approach (what beginners do):
    text = page.get_text()          # or pdfplumber.extract_text()
    chunks = text.split(...)
    embed(chunks)

Why this FAILS on enterprise PDFs:
1. Multi-column layouts get read left-to-right across columns → nonsense
2. Tables become "Q1 100 Q2 150 Q3 200" with no structure
3. You lose WHERE text lived on the page (needed to kill headers/footers)
4. Fonts/headings are discarded → weaker chunk boundaries later

Production approach (what we do here):
1. Open PDF with PyMuPDF (fitz)
2. Extract "dict" structure: blocks → lines → spans (with bbox + fonts)
3. Sort blocks in READING ORDER (top-to-bottom, then left-to-right)
4. Keep BoundingBox + font metadata for later cleaning/chunking
5. Return a ParsedDocument (typed contract), NOT a giant string

Remember the mantra:
    The quality of your RAG system is determined long before the prompt.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import fitz  # PyMuPDF

from app.models.document import (
    BlockType,
    BoundingBox,
    ContentBlock,
    PageContent,
    ParsedDocument,
    TextSpan,
)


def _is_bold_font(font_name: str | None) -> bool:
    """Heuristic: many PDF fonts encode bold in the font name."""
    if not font_name:
        return False
    name = font_name.lower()
    return "bold" in name or "black" in name or name.endswith("-bd")


def _sort_blocks_reading_order(
    blocks: list[ContentBlock],
    y_tolerance: float = 8.0,
) -> list[ContentBlock]:
    """
    Sort blocks into human reading order.

    WHY not trust PDF order?
    PDF internal order is the order objects were written, NOT reading order.
    Multi-column pages especially break if you trust raw order.

    Algorithm (simple + effective for learning / many enterprise docs):
    1. Group blocks that share roughly the same vertical position (a "row")
    2. Within a row, sort left → right
    3. Rows sorted top → bottom

    y_tolerance: how close in Y two blocks must be to count as same row.
    """
    if not blocks:
        return []

    # Sort roughly by Y first so grouping is stable
    tentative = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

    rows: list[list[ContentBlock]] = []
    current_row: list[ContentBlock] = [tentative[0]]
    current_y = tentative[0].bbox.y0

    for block in tentative[1:]:
        if abs(block.bbox.y0 - current_y) <= y_tolerance:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b.bbox.x0))
            current_row = [block]
            current_y = block.bbox.y0

    rows.append(sorted(current_row, key=lambda b: b.bbox.x0))

    ordered: list[ContentBlock] = []
    for row in rows:
        ordered.extend(row)
    return ordered


def _extract_blocks_from_page(
    page: fitz.Page,
    page_number: int,
) -> list[ContentBlock]:
    """
    Extract text blocks WITH layout from one page.

    PyMuPDF page.get_text("dict") returns:
      blocks → lines → spans
    Each has a bbox [x0, y0, x1, y1] and font info.

    We flatten lines inside a block into one ContentBlock,
    but KEEP span-level font metadata for later heading detection.
    """
    # "dict" mode is the layout-preserving mode we care about
    page_dict = page.get_text("dict")
    raw_blocks = page_dict.get("blocks", [])

    content_blocks: list[ContentBlock] = []

    for idx, raw in enumerate(raw_blocks):
        # type 0 = text, type 1 = image (we note images here; OCR in Step 3)
        block_type_code = raw.get("type", 0)
        bbox_vals = raw.get("bbox", (0, 0, 0, 0))
        bbox = BoundingBox(
            x0=float(bbox_vals[0]),
            y0=float(bbox_vals[1]),
            x1=float(bbox_vals[2]),
            y1=float(bbox_vals[3]),
        )

        if block_type_code == 1:
            # IMAGE block — preserve placeholder so Step 3 can OCR it
            content_blocks.append(
                ContentBlock(
                    block_id=f"p{page_number}_b{idx}",
                    block_type=BlockType.IMAGE,
                    page_number=page_number,
                    bbox=bbox,
                    text="",  # filled after OCR in Step 3
                    metadata={
                        "image_index": idx,
                        "note": "Image detected in Step 1; OCR deferred to Step 3",
                    },
                )
            )
            continue

        # TEXT block
        spans: list[TextSpan] = []
        line_texts: list[str] = []

        for line in raw.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = (span.get("text") or "").strip()
                if not span_text:
                    continue
                font_name = span.get("font")
                spans.append(
                    TextSpan(
                        text=span_text,
                        font=font_name,
                        font_size=float(span.get("size") or 0) or None,
                        is_bold=_is_bold_font(font_name),
                    )
                )
                line_parts.append(span_text)
            if line_parts:
                line_texts.append(" ".join(line_parts))

        full_text = "\n".join(line_texts).strip()
        if not full_text:
            continue  # skip empty noise blocks

        content_blocks.append(
            ContentBlock(
                block_id=f"p{page_number}_b{idx}",
                block_type=BlockType.TEXT,
                page_number=page_number,
                bbox=bbox,
                text=full_text,
                spans=spans,
                metadata={
                    # Useful later for heading-aware chunking
                    "avg_font_size": (
                        sum(s.font_size or 0 for s in spans) / len(spans)
                        if spans
                        else None
                    ),
                    "has_bold": any(s.is_bold for s in spans),
                },
            )
        )

    # CRITICAL: fix reading order before we ever chunk/embed
    return _sort_blocks_reading_order(content_blocks)


def parse_pdf(file_path: str | Path, document_id: str | None = None) -> ParsedDocument:
    """
    Public API for Step 1: parse a PDF into a layout-aware ParsedDocument.

    Parameters
    ----------
    file_path : path to the uploaded PDF
    document_id : optional id; we generate a UUID if not provided
                  (stable IDs help track docs through the whole RAG pipeline)

    Returns
    -------
    ParsedDocument — structured pages/blocks ready for Steps 2–5
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    doc_id = document_id or str(uuid.uuid4())

    # Open with PyMuPDF — this does NOT yet extract tables as grids (Step 2)
    # and does NOT OCR images (Step 3). We intentionally stage the pipeline.
    pdf = fitz.open(path)

    pages: list[PageContent] = []
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1  # humans think in 1-based pages

            blocks = _extract_blocks_from_page(page, page_number)
            pages.append(
                PageContent(
                    page_number=page_number,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=blocks,
                )
            )
    finally:
        # Always close file handles — production hygiene
        pdf.close()

    return ParsedDocument(
        document_id=doc_id,
        source_filename=path.name,
        page_count=len(pages),
        pages=pages,
        parser_name="pymupdf_layout",
        parser_version="step1",
    )
