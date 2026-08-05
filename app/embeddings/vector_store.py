"""
app/embeddings/vector_store.py

Persistence facade:
  1) Primary: Qdrant (production vector DB)
  2) Optional: JSON/npy debug dump (for learning/inspection)

Callers should prefer Qdrant for search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.embeddings.qdrant_store import (
    collection_info,
    list_indexed_document_ids,
    upsert_embedding_index,
)
from app.models.embedding import EmbeddingIndex


def save_index(index: EmbeddingIndex, directory: Path | None = None) -> dict[str, Any]:
    """
    Upsert into Qdrant + keep a JSON debug copy.

    WHY still write JSON?
    As a fresher you can open the file and inspect chunk text/metadata.
    Retrieval itself uses Qdrant, not this JSON.
    """
    qdrant_result = upsert_embedding_index(index)

    out_dir = directory or settings.vectors_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / f"{index.document_id}_index.json"
    # Store without giant vectors in the pretty JSON? Keep them for debugging once.
    full_path.write_text(
        json.dumps(index.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if index.items:
        matrix = np.array([item.embedding for item in index.items], dtype=np.float32)
        np.save(out_dir / f"{index.document_id}_vectors.npy", matrix)

        meta = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "source_filename": item.source_filename,
                "text": item.text,
                "chunk_index": item.chunk_index,
                "page_numbers": item.page_numbers,
                "chunk_type": item.chunk_type,
                "embedding_dim": item.embedding_dim,
                "embedding_model": item.embedding_model,
                "metadata": item.metadata,
            }
            for item in index.items
        ]
        (out_dir / f"{index.document_id}_meta.json").write_text(
            json.dumps(
                {
                    "document_id": index.document_id,
                    "source_filename": index.source_filename,
                    "embedding_model": index.embedding_model,
                    "embedding_dim": index.embedding_dim,
                    "chunk_count": index.chunk_count,
                    "items": meta,
                    "qdrant": qdrant_result,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {
        "debug_json": str(full_path),
        "qdrant": qdrant_result,
        "collection_info": collection_info(),
    }


def list_indexes(directory: Path | None = None) -> list[str]:
    """Prefer Qdrant document ids; fall back to local JSON names."""
    ids = list_indexed_document_ids()
    if ids:
        return ids

    out_dir = directory or settings.vectors_dir
    if not out_dir.exists():
        return []
    return sorted(p.name.replace("_index.json", "") for p in out_dir.glob("*_index.json"))


def load_index(document_id: str, directory: Path | None = None) -> EmbeddingIndex:
    """
    Debug helper: load the JSON dump for one document.
    Runtime retrieval should use Qdrant search, not this.
    """
    out_dir = directory or settings.vectors_dir
    path = out_dir / f"{document_id}_index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No debug JSON for document_id={document_id}. "
            "Re-run /embed/pdf so Qdrant + JSON are populated."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingIndex.model_validate(data)
