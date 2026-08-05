"""
app/parsers/header_footer_cleaner.py  —  STEP 4 CORE

============================================================
LESSON: Why headers/footers destroy RAG quality
============================================================

Enterprise PDFs repeat the SAME lines on EVERY page:

  Top:    "ACME Corp Confidential — Q4 Report"
  Bottom: "Page 3 of 48 | Internal Use Only"

If you chunk without cleaning:
  • Almost EVERY chunk contains that boilerplate
  • Embeddings get pulled toward "Confidential" / "Page N of M"
  • Retrieval ranks noisy chunks highly for unrelated queries
  • The LLM wastes context window on junk

That is exactly the LinkedIn point:
  "Every chunk contains repeated headers and footers."
  "Clean the document before chunking."

WHY we clean BEFORE chunking (not after):
-----------------------------------------
Chunking/embedding are expensive and sticky.
Once noise is inside vectors, you cannot "prompt it away" reliably.
Clean upstream → cleaner chunks → better retrieval → better answers.

Detection strategy (production-practical heuristic):
----------------------------------------------------
We combine THREE signals (any one alone is too weak):

1. POSITION
   - Header band = top ~12% of page height
   - Footer band = bottom ~12% of page height
   (This is why Step 1 kept bbox.y0 / y1!)

2. REPETITION across pages
   - Same normalized text appears on many pages
   - True body paragraphs rarely repeat verbatim page-to-page

3. PAGE-NUMBER patterns
   - "Page 1 of 10", "3/10", "- 4 -" etc. are almost always footers

We MARK blocks as HEADER/FOOTER (don't silently delete history),
then optionally DROP them from the active content list used later
for chunking. Provenance stays in metadata for debugging.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from app.models.document import BlockType, ContentBlock, ParsedDocument

# Normalize "Page 3 of 48" / "3 / 48" / "- 3 -" style noise
_PAGE_NUM_RE = re.compile(
    r"""
    (
        page\s*\d+(\s*of\s*\d+)?     # Page 3 of 48
      | \b\d+\s*/\s*\d+\b            # 3/48
      | ^\s*[-–—]?\s*\d+\s*[-–—]?\s*$  # - 3 -
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_text(text: str) -> str:
    """
    Normalize for cross-page comparison.

    WHY normalize?
    "Page 1 of 10" and "Page 2 of 10" should count as the SAME footer pattern,
    otherwise repetition detection fails.
    """
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Replace page numbers with a placeholder so variants collapse
    t = re.sub(r"\bpage\s*\d+(\s*of\s*\d+)?\b", "page # of #", t, flags=re.I)
    t = re.sub(r"\b\d+\s*/\s*\d+\b", "#/#", t)
    t = re.sub(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "#", t)
    return t


def _in_header_band(block: ContentBlock, page_height: float, band_ratio: float) -> bool:
    """True if block sits in the top band of the page."""
    if page_height <= 0:
        return False
    return block.bbox.y1 <= page_height * band_ratio


def _in_footer_band(block: ContentBlock, page_height: float, band_ratio: float) -> bool:
    """True if block sits in the bottom band of the page."""
    if page_height <= 0:
        return False
    return block.bbox.y0 >= page_height * (1.0 - band_ratio)


def _is_page_number_like(text: str) -> bool:
    return bool(_PAGE_NUM_RE.search((text or "").strip()))


def _candidate_kind(
    block: ContentBlock,
    page_height: float,
    band_ratio: float,
) -> str | None:
    """
    Return 'header', 'footer', or None for body content.

    WHY only TEXT/IMAGE candidates?
    We usually do NOT strip TABLE blocks just because they are near the edge —
    a table at the top can be real content. Be conservative.
    """
    if block.block_type not in (BlockType.TEXT, BlockType.IMAGE):
        return None
    if not (block.text or "").strip() and block.block_type == BlockType.TEXT:
        return None

    if _in_header_band(block, page_height, band_ratio):
        return "header"
    if _in_footer_band(block, page_height, band_ratio) or _is_page_number_like(block.text):
        return "footer"
    return None


def clean_headers_footers(
    document: ParsedDocument,
    *,
    band_ratio: float = 0.12,
    min_pages_for_repetition: int = 2,
    repetition_ratio: float = 0.5,
    drop_from_content: bool = True,
) -> ParsedDocument:
    """
    STEP 4 public API: detect + mark (+ optionally drop) headers/footers.

    Parameters
    ----------
    band_ratio
        Top/bottom fraction of page treated as header/footer zone.
    min_pages_for_repetition
        Need at least this many pages before trusting repetition signal.
        (On a 1-page PDF we still remove page-number-like footers + band text
         that looks boilerplate, but more carefully.)
    repetition_ratio
        Fraction of pages a normalized string must appear on to count as repeated.
        Example: 0.5 on a 4-page doc → must appear on >= 2 pages.
    drop_from_content
        If True, remove HEADER/FOOTER blocks from page.blocks (chunking input).
        Detection details stay recoverable via returned stats / metadata snapshot.
    """
    page_count = max(document.page_count, 1)

    # Pass 1: gather position candidates + frequency of normalized text
    # page -> list[(block, kind)]
    candidates: dict[int, list[tuple[ContentBlock, str]]] = defaultdict(list)
    norm_page_hits: Counter[str] = Counter()
    # Track which pages each normalized string appears on (set size = page coverage)
    norm_to_pages: dict[str, set[int]] = defaultdict(set)

    for page in document.pages:
        for block in page.blocks:
            kind = _candidate_kind(block, page.height, band_ratio)
            if not kind:
                continue
            candidates[page.page_number].append((block, kind))
            if (block.text or "").strip():
                norm = _normalize_text(block.text)
                if norm:
                    norm_to_pages[norm].add(page.page_number)

    for norm, pages in norm_to_pages.items():
        norm_page_hits[norm] = len(pages)

    required_hits = max(
        min_pages_for_repetition,
        int(page_count * repetition_ratio + 0.999),  # ceil
    )
    # On short docs, still allow strong page-number footers
    if page_count == 1:
        required_hits = 1

    repeated_norms = {
        norm for norm, hits in norm_page_hits.items() if hits >= required_hits
    }

    removed: list[dict[str, Any]] = []
    marked = 0

    # Pass 2: mark / drop
    for page in document.pages:
        kept: list[ContentBlock] = []
        cand_map = {id(b): kind for b, kind in candidates.get(page.page_number, [])}

        for block in page.blocks:
            kind = cand_map.get(id(block))
            if kind is None:
                kept.append(block)
                continue

            text = (block.text or "").strip()
            norm = _normalize_text(text) if text else ""
            is_repeated = bool(norm) and norm in repeated_norms
            is_page_num = _is_page_number_like(text)

            # Decision policy:
            # - Multi-page: must be repeated OR page-number-like
            # - Single-page: band position + (short boilerplate OR page-number)
            should_strip = False
            if page_count >= 2:
                should_strip = is_repeated or is_page_num
            else:
                # Single page: only strip obvious page labels / very short band lines
                should_strip = is_page_num or (len(text) <= 80 and bool(text))

            if not should_strip:
                kept.append(block)
                continue

            # Mark type for transparency
            block.block_type = BlockType.HEADER if kind == "header" else BlockType.FOOTER
            block.metadata = {
                **(block.metadata or {}),
                "cleaned_as": kind,
                "normalized_text": norm,
                "was_repeated_across_pages": is_repeated,
                "page_number_like": is_page_num,
                "note": (
                    "Removed from chunking input so repeated boilerplate "
                    "does not pollute embeddings/retrieval"
                ),
            }
            marked += 1
            removed.append(
                {
                    "block_id": block.block_id,
                    "page": page.page_number,
                    "kind": kind,
                    "text": text[:200],
                    "normalized_text": norm,
                }
            )

            if not drop_from_content:
                kept.append(block)
            # else: drop — do not append to kept

        page.blocks = kept

    document.parser_name = document.parser_name + "+hf_clean"
    document.parser_version = "step4"
    document.pipeline_stats["header_footer_cleaning"] = {
        "headers_footers_removed": removed,
        "marked_count": marked,
        "drop_from_content": drop_from_content,
        "band_ratio": band_ratio,
        "repeated_norms": sorted(repeated_norms),
    }
    return document


def header_footer_stats(document: ParsedDocument) -> dict[str, Any]:
    """API/CLI helper — works even after blocks were dropped."""
    summary = (document.pipeline_stats or {}).get("header_footer_cleaning")
    if isinstance(summary, dict) and "headers_footers_removed" in summary:
        removed = summary["headers_footers_removed"]
        return {
            "removed_count": len(removed),
            "removed": removed,
            "repeated_norms": summary.get("repeated_norms", []),
            "band_ratio": summary.get("band_ratio"),
        }

    # Fallback: count remaining tagged blocks (if drop_from_content=False)
    tagged = [
        {
            "block_id": b.block_id,
            "page": b.page_number,
            "kind": b.block_type.value,
            "text": (b.text or "")[:200],
        }
        for page in document.pages
        for b in page.blocks
        if b.block_type in (BlockType.HEADER, BlockType.FOOTER)
    ]
    return {"removed_count": 0, "still_tagged_in_content": tagged}
