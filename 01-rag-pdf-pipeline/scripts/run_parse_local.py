"""
scripts/run_parse_local.py

Run Steps 1–6 (parse → chunk → embed):

    python scripts/run_parse_local.py path/to/file.pdf
    python scripts/run_parse_local.py path/to/file.pdf --embed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.parsers.pipeline import ingest_and_chunk_pdf, ingest_chunk_and_embed_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_parse_local.py <path-to-pdf> [--embed]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    do_embed = "--embed" in sys.argv or True  # Step 6 default: always embed now
    demo = "--demo-chunks" in sys.argv
    chunk_size = 350 if demo else settings.chunk_size
    overlap = 60 if demo else settings.chunk_overlap

    if do_embed:
        document, bundle, index, saved = ingest_chunk_and_embed_pdf(
            pdf_path,
            run_ocr=False,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    else:
        document, bundle = ingest_and_chunk_pdf(
            pdf_path,
            run_ocr=False,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        index, saved = None, None

    parsed_out = settings.parsed_dir / f"{document.document_id}.json"
    chunks_out = settings.chunks_dir / f"{document.document_id}_chunks.json"
    parsed_out.write_text(
        json.dumps(document.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chunks_out.write_text(
        json.dumps(bundle.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 60)
    print("STEPS 1-6 - Parse -> Chunk -> Embed")
    print("=" * 60)
    print(f"document_id : {document.document_id}")
    print(f"chunks      : {bundle.chunk_count}")
    if index:
        print(f"model       : {index.embedding_model}")
        print(f"dim         : {index.embedding_dim}")
        print(f"qdrant save : {saved}")
        print()
        for item in index.items:
            print(f"[{item.chunk_index}] dim={item.embedding_dim} pages={item.page_numbers}")
            print(f"    text: {item.text[:120]!r}")
            print(f"    vec[:5]: {item.embedding[:5]}")
            print()

    print("Vectors are now in Qdrant. Next: python scripts/run_retrieve.py \"your question\"")
    print("Or full RAG: python scripts/run_ask.py \"your question\"")


if __name__ == "__main__":
    main()
