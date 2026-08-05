"""
app/chunking/chunker.py  —  STEP 5 CORE

============================================================
LESSON: Why chunking decides retrieval quality
============================================================

After Steps 1–4 you have CLEAN blocks.
Now you must decide the RETRIEVAL UNIT size.

Bad chunking (what beginners do):
  text = entire_pdf_as_string
  chunks = [text[i:i+500] for i in range(0, len(text), 500)]

Problems:
  1. Splits mid-sentence / mid-table → broken meaning
  2. No overlap → answer straddling two chunks is missed
  3. Ignores structure (tables/images treated like paragraphs)
  4. No metadata → cannot cite page/source later

Production approach we implement:
---------------------------------
A) Structure-aware:
   - TABLE block  → prefer ONE chunk (markdown table stays whole)
   - IMAGE+OCR    → own chunk (don't mix with unrelated paragraphs)
   - TEXT         → merge small neighbors, split long ones carefully

B) Size targets:
   - chunk_size   ≈ how much text per embedding (chars for learning)
   - chunk_overlap ≈ repeated tail/head so ideas on boundaries survive

C) Sentence-aware splits when possible (not blind character cuts)

D) Rich metadata on every chunk for Step 6–8 (embed / retrieve / cite)

WHY overlap?
  Query: "pressure test result for Well-12"
  If the sentence starts at the end of chunk A and finishes in chunk B,
  without overlap NEITHER chunk may embed the full meaning well.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.models.chunk import Chunk, ChunkBundle
from app.models.document import BlockType, ContentBlock, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _overlap_seed(text: str, overlap: int) -> str:
    """
    Take the last `overlap` chars, but snap to a word boundary.

    WHY?
    Blind character overlap produces junk like "t, the test fails..."
    which hurts embedding quality. Word-safe overlap keeps meaning intact.
    """
    if overlap <= 0 or not text:
        return ""
    seed = text[-overlap:]
    # If we started mid-word, drop the partial first token
    if len(text) > overlap and not text[-overlap - 1].isspace():
        parts = seed.split()
        if len(parts) > 1:
            seed = " ".join(parts[1:])
        # else keep as-is (single long token)
    return seed.strip()


def _split_long_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split long text into overlapping windows, preferring sentence boundaries.

    Algorithm:
    1. Break into sentences
    2. Pack sentences until chunk_size
    3. Start next window with a word-safe overlap seed from previous end
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        parts: list[str] = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(text), step):
            piece = text[i : i + chunk_size].strip()
            if piece:
                # word-safe trim at start for non-first windows
                if i > 0:
                    toks = piece.split()
                    piece = " ".join(toks[1:]) if len(toks) > 1 else piece
                if piece:
                    parts.append(piece)
            if i + chunk_size >= len(text):
                break
        return parts

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if len(sent) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            step = max(1, chunk_size - overlap)
            for i in range(0, len(sent), step):
                piece = sent[i : i + chunk_size].strip()
                if i > 0 and piece:
                    toks = piece.split()
                    piece = " ".join(toks[1:]) if len(toks) > 1 else piece
                if piece:
                    chunks.append(piece)
                if i + chunk_size >= len(sent):
                    break
            continue

        extra = len(sent) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            packed = " ".join(current).strip()
            chunks.append(packed)

            seed = _overlap_seed(packed, overlap)
            current = [seed, sent] if seed else [sent]
            current_len = len(" ".join(current))
        else:
            current.append(sent)
            current_len += extra

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def _block_chunk_type(block: ContentBlock) -> str:
    if block.block_type == BlockType.TABLE:
        return "table"
    if block.block_type == BlockType.IMAGE:
        return "image_ocr"
    return "text"


def _iter_content_blocks(document: ParsedDocument) -> Iterable[ContentBlock]:
    """Yield blocks that still have embeddable text after Steps 1–4."""
    for page in document.pages:
        for block in page.blocks:
            if block.block_type in (BlockType.HEADER, BlockType.FOOTER):
                continue
            if not (block.text or "").strip():
                continue
            yield block


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> ChunkBundle:
    """
    STEP 5 public API: turn a cleaned ParsedDocument into Chunks.

    Defaults (chars, not tokens):
      chunk_size=800  ≈ ~200 tokens — good learning default for small docs
      overlap=120     ≈ ~30 tokens — enough to bridge boundaries

    Tune later based on your embedding model context and query style.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    index = 0

    # Buffer for merging small adjacent TEXT blocks on the SAME page
    text_buffer: list[ContentBlock] = []

    def flush_text_buffer() -> None:
        nonlocal index
        if not text_buffer:
            return

        combined = "\n\n".join(b.text.strip() for b in text_buffer if b.text.strip())
        pages = sorted({b.page_number for b in text_buffer})
        block_ids = [b.block_id for b in text_buffer]
        pieces = _split_long_text(combined, chunk_size, chunk_overlap)

        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_c{index}",
                    document_id=document.document_id,
                    source_filename=document.source_filename,
                    text=piece,
                    chunk_index=index,
                    page_numbers=pages,
                    source_block_ids=block_ids,
                    chunk_type="text",
                    metadata={
                        "merged_block_count": len(text_buffer),
                        "why": (
                            "Adjacent text blocks merged then split with overlap "
                            "so retrieval units are coherent and boundary-safe"
                        ),
                    },
                )
            )
            index += 1
        text_buffer.clear()

    for block in _iter_content_blocks(document):
        btype = _block_chunk_type(block)

        # --- TABLE: keep structure together when possible ---
        if btype == "table":
            flush_text_buffer()
            table_text = block.text.strip()
            # If a huge table exceeds size, split by rows but keep header line
            if len(table_text) <= chunk_size:
                pieces = [table_text]
            else:
                lines = table_text.splitlines()
                header = "\n".join(lines[:2]) if len(lines) >= 2 else lines[0]
                body_lines = lines[2:] if len(lines) > 2 else []
                pieces = []
                buf = [header]
                buf_len = len(header)
                for line in body_lines:
                    if buf_len + len(line) + 1 > chunk_size and len(buf) > 1:
                        pieces.append("\n".join(buf))
                        # overlap: repeat header so each piece stays meaningful
                        buf = [header, line]
                        buf_len = len(header) + len(line) + 1
                    else:
                        buf.append(line)
                        buf_len += len(line) + 1
                if buf:
                    pieces.append("\n".join(buf))

            for piece in pieces:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}_c{index}",
                        document_id=document.document_id,
                        source_filename=document.source_filename,
                        text=piece,
                        chunk_index=index,
                        page_numbers=[block.page_number],
                        source_block_ids=[block.block_id],
                        chunk_type="table",
                        metadata={
                            "table_headers": block.table.headers if block.table else [],
                            "why": (
                                "Tables stay as structured markdown chunks so "
                                "row/column meaning is not scrambled"
                            ),
                        },
                    )
                )
                index += 1
            continue

        # --- IMAGE OCR: own chunk(s), don't mix with unrelated text ---
        if btype == "image_ocr":
            flush_text_buffer()
            for piece in _split_long_text(block.text.strip(), chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}_c{index}",
                        document_id=document.document_id,
                        source_filename=document.source_filename,
                        text=piece,
                        chunk_index=index,
                        page_numbers=[block.page_number],
                        source_block_ids=[block.block_id],
                        chunk_type="image_ocr",
                        metadata={
                            "ocr_confidence": (block.metadata or {}).get(
                                "ocr_avg_confidence"
                            ),
                            "why": (
                                "OCR text kept as its own chunk so equipment-plate "
                                "facts are retrievable independently"
                            ),
                        },
                    )
                )
                index += 1
            continue

        # --- TEXT: merge small neighbors on same page ---
        if text_buffer and text_buffer[-1].page_number != block.page_number:
            flush_text_buffer()

        text_buffer.append(block)
        buffered_len = sum(len(b.text or "") for b in text_buffer)
        # Flush when buffer grows past ~1.5x chunk size (avoid huge merges)
        if buffered_len >= int(chunk_size * 1.5):
            flush_text_buffer()

    flush_text_buffer()

    return ChunkBundle(
        document_id=document.document_id,
        source_filename=document.source_filename,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunks,
        strategy="structure_aware_overlap",
    )
