"""
app/retrieval/retriever.py  —  STEP 7 CORE (Qdrant-backed)

============================================================
LESSON: Retrieval with a Vector DB
============================================================

Old (learning) path:
  load all vectors from JSON → NumPy cosine over everything

New (production-style) path:
  embed query → Qdrant ANN search → top_k payloads (text + citations)

WHY this is better:
  - Fast at large scale (HNSW index inside Qdrant)
  - Filter by document_id / metadata without custom code
  - Persistence + concurrency handled by the DB
"""

from __future__ import annotations

from app.core.config import settings
from app.embeddings.embedder import embed_texts
from app.embeddings.qdrant_store import collection_info, list_indexed_document_ids, search_qdrant
from app.models.retrieval import RetrievalResult


def retrieve(
    query: str,
    *,
    top_k: int = 3,
    document_ids: list[str] | None = None,
    min_score: float = 0.0,
) -> RetrievalResult:
    """
    Semantic search via Qdrant.

    1) Embed question with SAME model used at index time
    2) Ask Qdrant for nearest vectors (cosine)
    3) Build LLM context string from payloads
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    info = collection_info()
    if not info.get("exists") or int(info.get("points_count") or 0) == 0:
        raise FileNotFoundError(
            "Qdrant has no points yet. Run POST /embed/pdf (or scripts/run_parse_local.py) first."
        )

    model_name = settings.embedding_model
    query_vec = embed_texts([query], model_name=model_name)[0]

    hits = search_qdrant(
        query_vec,
        top_k=top_k,
        document_ids=document_ids,
        min_score=min_score,
    )

    context_parts: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        pages = ",".join(str(p) for p in hit.page_numbers) or "?"
        context_parts.append(
            f"[Source {rank} | file={hit.source_filename} | pages={pages} | score={hit.score}]\n"
            f"{hit.text}"
        )

    searched_ids = document_ids or list_indexed_document_ids()

    return RetrievalResult(
        query=query,
        top_k=top_k,
        embedding_model=model_name,
        hit_count=len(hits),
        hits=hits,
        context_for_llm="\n\n".join(context_parts),
        searched_document_ids=searched_ids,
    )
