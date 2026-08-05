"""
app/parsers/ocr_extractor.py  —  STEP 3 CORE

============================================================
LESSON: Why OCR belongs in a production RAG pipeline
============================================================

Enterprise PDFs are NOT "text files with a .pdf extension".

They often contain:
  • Scanned pages (entire page is an image)
  • Charts / screenshots with labels
  • Stamped text ("CONFIDENTIAL", "APPROVED")
  • Photos of whiteboards / equipment plates

If you only extract the digital text layer:
  → image text NEVER reaches embeddings
  → retrieval cannot find that knowledge
  → the LLM looks "dumb" even though the PDF had the answer

That is why the LinkedIn post said:
  "If an image contains text, extract it.
   Otherwise, valuable context never reaches your embeddings."

WHY we implement OCR as its OWN step (not mixed into Step 1):
-------------------------------------------------------------
1. OCR is SLOW and CPU-heavy — run it only when images exist
2. You can toggle OCR off for born-digital docs (cost control)
3. You can swap engines (RapidOCR / Tesseract / cloud Vision API)
   without rewriting layout parsing or table logic
4. Debugging stays clear: "Was the failure parse, table, or OCR?"

Engine choice on this project:
  RapidOCR (ONNX) — pip-installable, no system Tesseract required.
  (Tesseract is also industry-standard; install later if you prefer.)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.core.config import settings
from app.models.document import BlockType, ParsedDocument


# Save extracted images so YOU can open them and verify OCR input
IMAGE_DIR = settings.upload_dir.parent / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_ocr_engine():
    """
    Lazy-load RapidOCR once (model load is expensive).

    WHY cache?
    Loading ONNX models on every page would make ingestion painfully slow.
    In production you keep the OCR engine warm in the worker process.
    """
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_numpy_image(image_array: np.ndarray) -> tuple[str, float | None, list[dict[str, Any]]]:
    """
    Run OCR on an RGB numpy image.

    Returns
    -------
    text        : joined detected strings
    avg_conf    : average confidence (0..1) if available
    lines       : per-line detail for debugging
    """
    engine = _get_ocr_engine()
    result, _elapsed = engine(image_array)

    if not result:
        return "", None, []

    # RapidOCR result items look like:
    # [box_points, text, confidence]
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    confs: list[float] = []

    for item in result:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        conf = float(item[2]) if len(item) > 2 and item[2] is not None else None
        texts.append(text)
        if conf is not None:
            confs.append(conf)
        lines.append({"text": text, "confidence": conf})

    joined = "\n".join(texts)
    avg_conf = (sum(confs) / len(confs)) if confs else None
    return joined, avg_conf, lines


def _pix_to_rgb_array(pix: fitz.Pixmap) -> np.ndarray:
    """Convert PyMuPDF Pixmap → RGB numpy array for OCR."""
    # Handle alpha / CMYK by converting to RGB first
    if pix.n - pix.alpha > 3:  # CMYK etc.
        pix = fitz.Pixmap(fitz.csRGB, pix)
    elif pix.alpha:
        pix = fitz.Pixmap(pix, 0)  # drop alpha

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return np.array(img)


def _extract_image_region(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    zoom: float = 2.0,
) -> tuple[bytes, fitz.Pixmap]:
    """
    Render the IMAGE region (clip to bbox) at higher DPI for better OCR.

    WHY zoom=2.0?
    OCR accuracy rises with resolution. Native PDF image DPI is often low.
    2x render is a practical tradeoff: better text, still reasonable speed.

    WHY clip to bbox instead of whole page?
    - Faster
    - Less noise from surrounding content
    - Matches the IMAGE block we already detected in Step 1
    """
    mat = fitz.Matrix(zoom, zoom)
    clip = fitz.Rect(bbox)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png"), pix


def enrich_document_with_ocr(
    document: ParsedDocument,
    file_path: str | Path,
    *,
    min_image_side_px: int = 40,
    min_confidence: float = 0.40,
) -> ParsedDocument:
    """
    STEP 3 public API: fill IMAGE blocks with OCR text.

    WHY skip tiny images?
    Icons / bullets / decorative lines waste OCR time and add garbage tokens
    to embeddings. Production pipelines filter by size.

    WHY min_confidence?
    Low-confidence OCR is often noise. Better to leave text empty than
    poison retrieval with junk strings.
    """
    path = Path(file_path)
    pdf = fitz.open(path)

    ocr_success = 0
    ocr_skipped = 0

    try:
        for page_model in document.pages:
            page = pdf[page_model.page_number - 1]

            for block in page_model.blocks:
                if block.block_type != BlockType.IMAGE:
                    continue

                bbox = (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)

                # Skip tiny decorative images
                if block.bbox.width < min_image_side_px and block.bbox.height < min_image_side_px:
                    block.metadata.update(
                        {
                            "ocr_status": "skipped_too_small",
                            "note": "Decorative/tiny image — OCR not worth running",
                        }
                    )
                    ocr_skipped += 1
                    continue

                try:
                    png_bytes, pix = _extract_image_region(page, bbox)
                    # Save for human inspection (fresher learning habit!)
                    img_path = IMAGE_DIR / f"{document.document_id}_{block.block_id}.png"
                    img_path.write_bytes(png_bytes)

                    rgb = _pix_to_rgb_array(pix)
                    text, avg_conf, lines = _ocr_numpy_image(rgb)

                    # Filter weak OCR
                    if not text or (avg_conf is not None and avg_conf < min_confidence):
                        block.text = ""
                        block.metadata.update(
                            {
                                "ocr_status": "no_reliable_text",
                                "ocr_engine": "rapidocr_onnxruntime",
                                "ocr_avg_confidence": avg_conf,
                                "ocr_image_path": str(img_path),
                                "ocr_lines": lines,
                                "note": (
                                    "Image had no reliable text — "
                                    "this is OK for pure photos/charts without labels"
                                ),
                            }
                        )
                        ocr_skipped += 1
                        continue

                    # SUCCESS — text now can flow into chunking/embeddings
                    block.text = text
                    block.metadata.update(
                        {
                            "ocr_status": "ok",
                            "ocr_engine": "rapidocr_onnxruntime",
                            "ocr_avg_confidence": avg_conf,
                            "ocr_image_path": str(img_path),
                            "ocr_line_count": len(lines),
                            "ocr_lines": lines,
                            "note": (
                                "OCR text extracted so image content can be embedded "
                                "and retrieved like normal document text"
                            ),
                        }
                    )
                    ocr_success += 1

                except Exception as exc:  # keep pipeline alive if one image fails
                    block.metadata.update(
                        {
                            "ocr_status": "error",
                            "ocr_error": str(exc),
                            "note": "OCR failed for this image; other blocks still usable",
                        }
                    )
                    ocr_skipped += 1
    finally:
        pdf.close()

    # Provenance — always know which stages ran
    document.parser_name = document.parser_name + "+ocr"
    document.parser_version = "step3"
    # Stash summary on the document via a lightweight convention:
    # first page metadata is awkward; use a synthetic approach on doc itself
    # by attaching to each page is noisy — keep counts in a module-level style
    # by writing into a custom attribute via model_extra isn't ideal.
    # Instead, add summary onto document by updating parser fields only;
    # API layer will compute image stats from blocks.
    _ = (ocr_success, ocr_skipped)  # used by caller via block metadata
    return document


def image_ocr_stats(document: ParsedDocument) -> dict[str, Any]:
    """Helper for API / CLI summaries."""
    images = [
        b
        for page in document.pages
        for b in page.blocks
        if b.block_type == BlockType.IMAGE
    ]
    ok = [b for b in images if (b.metadata or {}).get("ocr_status") == "ok"]
    return {
        "image_block_count": len(images),
        "ocr_ok_count": len(ok),
        "ocr_with_text_preview": [
            {
                "block_id": b.block_id,
                "page": b.page_number,
                "confidence": (b.metadata or {}).get("ocr_avg_confidence"),
                "text": b.text[:300],
                "image_path": (b.metadata or {}).get("ocr_image_path"),
            }
            for b in ok
        ],
    }
