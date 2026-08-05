"""
app/embeddings/qdrant_store.py

============================================================
LESSON: Why use a Vector DB (Qdrant)?
============================================================

Earlier we stored vectors in JSON + NumPy and scanned all of them.
That is fine for learning / tiny demos.

It breaks at production scale because:
  - Loading every vector into RAM is slow/expensive
  - Brute-force cosine over millions of chunks is too slow
  - Filtering (by document, date, tenant) is awkward
  - Concurrent upserts/searches need a real service

Qdrant is a Vector Database specialized for this:
  - Stores vectors + payload (metadata/text)
  - ANN search (HNSW) for fast nearest-neighbor lookup
  - Payload filters (e.g. document_id == "...")
  - Persistent storage

Modes we support:
  1) local  — embedded Qdrant on disk (no Docker)  ← default for learning
  2) server — Qdrant running at QDRANT_URL (Docker/cloud)
"""

from __future__ import annotations

import atexit
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings
from app.models.embedding import EmbeddingIndex
from app.models.retrieval import RetrievalHit


def _point_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be UUID or unsigned int.
    We derive a stable UUID5 from chunk_id so re-upserts update the same point.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    One shared client per process.

    WHY cache?
    Opening local Qdrant path locks the storage directory.
    Creating many clients causes "already accessed by another instance" errors.
    """
    mode = (settings.qdrant_mode or "local").strip().lower()
    if mode == "server":
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )
    else:
        # Local persistent embedded mode (no Docker required)
        path = settings.qdrant_path
        path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(path))

    # Avoid noisy destructor errors on interpreter shutdown (local mode)
    def _close() -> None:
        try:
            client.close()
        except Exception:
            pass

    atexit.register(_close)
    return client


def ensure_collection(vector_size: int) -> None:
    """
    Create the collection if missing.

    WHY cosine?
    Our MiniLM embeddings are compared by direction (semantic similarity).
    Cosine is the matching metric for that.
    """
    client = get_qdrant_client()
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )


def upsert_embedding_index(index: EmbeddingIndex) -> dict[str, Any]:
    """
    Upsert all embedded chunks from one document into Qdrant.

    Each point =
      vector  → embedding floats
      payload → text + citation metadata (what the LLM needs later)
    """
    if not index.items:
        return {
            "collection": settings.qdrant_collection,
            "upserted": 0,
            "document_id": index.document_id,
        }

    dim = index.embedding_dim or len(index.items[0].embedding)
    ensure_collection(dim)
    client = get_qdrant_client()

    points: list[qm.PointStruct] = []
    for item in index.items:
        points.append(
            qm.PointStruct(
                id=_point_id(item.chunk_id),
                vector=item.embedding,
                payload={
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "source_filename": item.source_filename,
                    "text": item.text,
                    "chunk_index": item.chunk_index,
                    "page_numbers": item.page_numbers,
                    "chunk_type": item.chunk_type,
                    "embedding_model": item.embedding_model,
                    "embedding_dim": item.embedding_dim,
                    "metadata": item.metadata or {},
                },
            )
        )

    # Batch upsert (production habit: never one-by-one if you can avoid it)
    client.upsert(collection_name=settings.qdrant_collection, points=points)

    return {
        "collection": settings.qdrant_collection,
        "upserted": len(points),
        "document_id": index.document_id,
        "vector_size": dim,
        "mode": settings.qdrant_mode,
    }


def search_qdrant(
    query_vector: list[float],
    *,
    top_k: int = 3,
    document_ids: list[str] | None = None,
    min_score: float = 0.0,
) -> list[RetrievalHit]:
    """
    ANN search in Qdrant (cosine similarity).

    Optional filter: only search points whose payload.document_id is in document_ids.
    """
    client = get_qdrant_client()
    name = settings.qdrant_collection

    # Collection might not exist yet if user never embedded
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        raise FileNotFoundError(
            f"Qdrant collection '{name}' not found. Run POST /embed/pdf first."
        )

    query_filter = None
    if document_ids:
        query_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchAny(any=document_ids),
                )
            ]
        )

    # Modern qdrant-client uses query_points (search() was removed)
    response = client.query_points(
        collection_name=name,
        query=query_vector,
        query_filter=query_filter,
        limit=max(top_k * 3, top_k),  # over-fetch a bit for dedupe
        score_threshold=min_score if min_score > 0 else None,
        with_payload=True,
    )
    results = response.points

    hits: list[RetrievalHit] = []
    seen_text: set[str] = set()

    for point in results:
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        norm = " ".join(text.lower().split())
        if not norm or norm in seen_text:
            continue
        seen_text.add(norm)

        hits.append(
            RetrievalHit(
                chunk_id=str(payload.get("chunk_id") or point.id),
                document_id=str(payload.get("document_id") or ""),
                source_filename=str(payload.get("source_filename") or ""),
                text=text,
                score=round(float(point.score or 0.0), 6),
                page_numbers=list(payload.get("page_numbers") or []),
                chunk_type=str(payload.get("chunk_type") or "text"),
                chunk_index=int(payload.get("chunk_index") or 0),
                metadata=dict(payload.get("metadata") or {}),
            )
        )
        if len(hits) >= top_k:
            break

    return hits


def list_indexed_document_ids() -> list[str]:
    """Distinct document_ids currently stored in Qdrant."""
    client = get_qdrant_client()
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        return []

    doc_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=100,
            offset=offset,
            with_payload=["document_id"],
            with_vectors=False,
        )
        for p in points:
            did = (p.payload or {}).get("document_id")
            if did:
                doc_ids.add(str(did))
        if offset is None:
            break
    return sorted(doc_ids)


def collection_info() -> dict[str, Any]:
    client = get_qdrant_client()
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        return {
            "exists": False,
            "collection": name,
            "mode": settings.qdrant_mode,
            "points_count": 0,
        }
    info = client.get_collection(name)
    return {
        "exists": True,
        "collection": name,
        "mode": settings.qdrant_mode,
        "points_count": info.points_count,
        "status": str(info.status),
    }
