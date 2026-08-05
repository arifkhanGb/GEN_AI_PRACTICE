"""
app/embeddings/embedder.py  —  STEP 6 CORE

============================================================
LESSON: What embeddings are (and why RAG needs them)
============================================================

An embedding turns text into a list of numbers (a vector) such that:
  similar MEANING → vectors point in similar directions

Example intuition:
  "pressure test passed at 8500 psi"
  and
  "what was the wellhead pressure test result?"
  should be CLOSE in vector space — even if words differ.

WHY not keyword search alone?
  Users ask in their own words. Exact string match misses paraphrases.
  Semantic embeddings fix that (and keyword search can still help — hybrid
  retrieval comes later in real production systems).

WHY embed CHUNKS (Step 5), not whole PDFs?
  Retrieval must return a focused snippet for the LLM.
  Embedding a 50-page PDF as one vector loses all local detail.

WHY this comes AFTER cleaning/chunking:
  Garbage in → garbage vectors.
  Headers, scrambled tables, missing OCR = polluted embedding space.

Engine choice for this learning project:
  fastembed + a small open model (runs locally, no API key).
  In production you might use OpenAI text-embedding-3-*, Cohere, Voyage, etc.
  The INTERFACE stays the same: texts in → vectors out.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.models.chunk import Chunk, ChunkBundle
from app.models.embedding import EmbeddedChunk, EmbeddingIndex


@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str):
    """
    Lazy-load the embedding model once.

    WHY cache?
    Model download/load is expensive. In a FastAPI worker you load once
    and reuse for every request (same idea as the OCR engine in Step 3).
    """
    from fastembed import TextEmbedding

    # first run downloads model weights into a local cache
    return TextEmbedding(model_name=model_name)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """
    Embed a batch of strings.

    WHY batch?
    Embedding one-by-one is slow. Production always batches when possible.
    """
    if not texts:
        return []

    name = model_name or settings.embedding_model
    model = _get_embedding_model(name)

    # fastembed returns a generator of numpy arrays
    vectors: list[list[float]] = []
    for vec in model.embed(texts):
        vectors.append(vec.tolist())
    return vectors


def embed_chunk_bundle(
    bundle: ChunkBundle,
    *,
    model_name: str | None = None,
) -> EmbeddingIndex:
    """
    STEP 6 public API: turn ChunkBundle → EmbeddingIndex.

    We embed chunk.text (the retrieval unit), and keep ALL metadata
    needed for Step 7 (retrieve) and Step 8 (LLM citations).
    """
    name = model_name or settings.embedding_model
    chunks: list[Chunk] = bundle.chunks
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, model_name=name)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
        )

    dim = len(vectors[0]) if vectors else 0
    items: list[EmbeddedChunk] = []

    for chunk, vector in zip(chunks, vectors):
        items.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_filename=chunk.source_filename,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                page_numbers=chunk.page_numbers,
                chunk_type=chunk.chunk_type,
                embedding=vector,
                embedding_model=name,
                embedding_dim=len(vector),
                metadata={
                    **(chunk.metadata or {}),
                    "why_embedded": (
                        "Semantic vector so paraphrased questions can still "
                        "retrieve this chunk in Step 7"
                    ),
                },
            )
        )

    return EmbeddingIndex(
        document_id=bundle.document_id,
        source_filename=bundle.source_filename,
        embedding_model=name,
        embedding_dim=dim,
        chunk_count=len(items),
        items=items,
    )
