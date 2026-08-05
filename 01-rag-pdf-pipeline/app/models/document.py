"""
app/models/document.py

WHY Pydantic models for parsed documents?
------------------------------------------------
When you parse a PDF, you get messy nested dicts.
If you pass raw dicts into chunking/embedding later, bugs hide for weeks.

Typed models give you:
1. Clear schema of what "a page" and "a block" mean
2. Automatic validation
3. Easy JSON serialization for debugging
4. A contract between Parser → Chunker → Embedder

This is how production RAG teams think: DATA CONTRACTS first.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """
    What kind of content is this block?

    WHY classify blocks early?
    - Tables need different chunking than paragraphs
    - Images need OCR (Step 3)
    - Headers/footers need filtering (Step 4)
    If you treat everything as plain text, retrieval quality collapses.
    """

    TEXT = "text"
    TABLE = "table"          # filled in Step 2 as structured rows/columns
    IMAGE = "image"          # OCR text filled in Step 3
    HEADER = "header"        # detected later in Step 4
    FOOTER = "footer"        # detected later in Step 4


class BoundingBox(BaseModel):
    """
    Position of a content block on the page.

    WHY keep coordinates (x0, y0, x1, y1)?
    - Layout matters: left column vs right column
    - Headers/footers are usually at top/bottom — we need Y position
    - Reading order is NOT always the PDF's raw order; we sort by position
    Without bbox, you cannot clean headers or reconstruct multi-column text.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class TextSpan(BaseModel):
    """
    A small run of text with font metadata.

    WHY keep font size / font name?
    - Larger fonts often = headings → useful for smarter chunking later
    - Bold/title styles help structure the document hierarchy
    Production chunkers use this to avoid splitting mid-section poorly.
    """

    text: str
    font: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False


class TableData(BaseModel):
    """
    STEP 2 — Structured table (rows × columns), NOT scrambled text.

    WHY this model exists:
    ------------------------------------------------
    Naive parsers flatten a sales table into:
        "Q1 100 Q2 150 Q3 200 Q4 250"

    That destroys meaning. Retrieval cannot answer:
        "What was Q3 revenue?" reliably from scrambled text.

    Production approach:
    - Keep headers + rows as lists
    - Also keep a markdown rendering in `text` for embedding/LLM context
    - Keep both: STRUCTURE for accuracy, TEXT for vector search
    """

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    # How many columns pdfplumber detected (useful for debugging bad extracts)
    column_count: int = 0
    row_count: int = 0

    def to_markdown(self) -> str:
        """
        Render table as Markdown — LLMs understand this format very well.

        Example:
        | Quarter | Revenue |
        | --- | --- |
        | Q1 | 100 |
        """
        if not self.headers and not self.rows:
            return ""

        headers = self.headers or [f"col_{i+1}" for i in range(self.column_count)]
        # Normalize row widths so markdown stays rectangular
        width = len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for row in self.rows:
            padded = list(row) + [""] * max(0, width - len(row))
            lines.append("| " + " | ".join(padded[:width]) + " |")
        return "\n".join(lines)

    def to_key_value_lines(self) -> str:
        """
        Alternate text form: one row → 'Header: value; Header: value'
        Sometimes retrieves better for row-centric questions.
        """
        if not self.rows:
            return ""
        headers = self.headers or [f"col_{i+1}" for i in range(self.column_count)]
        lines: list[str] = []
        for row in self.rows:
            pairs: list[str] = []
            for i, cell in enumerate(row):
                key = headers[i] if i < len(headers) else f"col_{i+1}"
                pairs.append(f"{key}: {cell}")
            lines.append("; ".join(pairs))
        return "\n".join(lines)


class ContentBlock(BaseModel):
    """
    One coherent piece of content on a page (a paragraph, title, table, etc.).

    This is the ATOM of our pipeline before chunking.
    Bad blocks → bad chunks → bad embeddings → bad answers.
    """

    block_id: str
    block_type: BlockType = BlockType.TEXT
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: BoundingBox
    text: str = ""
    spans: list[TextSpan] = Field(default_factory=list)

    # STEP 2: when block_type == TABLE, this holds rows/columns
    table: Optional[TableData] = None

    # Extra structured payload (image metadata added in later steps)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageContent(BaseModel):
    """All extracted blocks for a single PDF page."""

    page_number: int
    width: float
    height: float
    blocks: list[ContentBlock] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """
    Full structured representation of a PDF after layout-aware parsing.

    WHY this object exists?
    This becomes the INPUT to:
    - Step 2: table extraction enrichment
    - Step 3: OCR enrichment
    - Step 4: header/footer cleaning
    - Step 5: chunking

    Never skip straight from PDF bytes → embeddings in production.
    """

    document_id: str
    source_filename: str
    page_count: int
    pages: list[PageContent] = Field(default_factory=list)

    # Pipeline provenance — critical for debugging RAG quality issues
    parser_name: str = "pymupdf_layout"
    parser_version: str = "step1"

    # Step summaries (OCR counts, removed headers, etc.) for API/debugging
    # WHY a dedicated field? So we can drop HEADER/FOOTER blocks from pages
    # (clean chunking input) but still report WHAT was removed.
    pipeline_stats: dict[str, Any] = Field(default_factory=dict)

    def all_text(self) -> str:
        """Convenience: concatenate all text (for quick inspection only)."""
        parts: list[str] = []
        for page in self.pages:
            for block in page.blocks:
                if block.text.strip():
                    parts.append(block.text.strip())
        return "\n\n".join(parts)

    def block_count(self) -> int:
        return sum(len(p.blocks) for p in self.pages)

    def table_count(self) -> int:
        """STEP 2 helper: how many TABLE blocks survived enrichment."""
        return sum(
            1
            for page in self.pages
            for block in page.blocks
            if block.block_type == BlockType.TABLE
        )
